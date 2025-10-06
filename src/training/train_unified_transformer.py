"""
Training script for Unified Transformer Model.
Trains end-to-end detection and damage classification in a single model.
"""

import sys
from pathlib import Path
import json

import hydra
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from loguru import logger
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.unified_transformer import (
    UnifiedTransformerModel,
    UnifiedTransformerLoss,
    create_unified_transformer
)


class UnifiedTransformerDataset(torch.utils.data.Dataset):
    """
    Dataset for unified transformer training.
    Returns images with detection and damage classification labels.
    """
    
    def __init__(
        self,
        images_dir: Path,
        annotations_file: Path,
        split: str = 'train',
        img_size: int = 640,
        max_objects: int = 100
    ):
        """
        Initialize dataset.
        
        Args:
            images_dir: Directory containing images
            annotations_file: COCO format annotations with damage attributes
            split: Dataset split ('train', 'val', 'test')
            img_size: Image size for resizing
            max_objects: Maximum number of objects per image
        """
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        self.max_objects = max_objects
        self.split = split
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build image index
        self.images = {img['id']: img for img in self.coco_data['images']}
        
        # Group annotations by image
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        # Get image IDs with annotations
        self.image_ids = list(self.image_annotations.keys())
        
        # Split dataset
        self._split_dataset()
        
        logger.info(f"Loaded {len(self.split_image_ids)} images for {split} split")
    
    def _split_dataset(self):
        """Split dataset into train/val/test."""
        np.random.seed(42)
        indices = np.random.permutation(len(self.image_ids))
        
        train_ratio = 0.7
        val_ratio = 0.15
        
        train_size = int(len(indices) * train_ratio)
        val_size = int(len(indices) * val_ratio)
        
        if self.split == 'train':
            split_indices = indices[:train_size]
        elif self.split == 'val':
            split_indices = indices[train_size:train_size + val_size]
        else:  # test
            split_indices = indices[train_size + val_size:]
        
        self.split_image_ids = [self.image_ids[i] for i in split_indices]
    
    def __len__(self):
        return len(self.split_image_ids)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        from PIL import Image
        import torchvision.transforms as T
        
        image_id = self.split_image_ids[idx]
        image_info = self.images[image_id]
        annotations = self.image_annotations[image_id]
        
        # Load image
        image_path = self.images_dir / image_info['file_name']
        image = Image.open(image_path).convert('RGB')
        
        # Resize image
        transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image)
        
        # Prepare targets
        num_objects = min(len(annotations), self.max_objects)
        
        bboxes = torch.zeros((self.max_objects, 4))
        class_labels = torch.zeros(self.max_objects, dtype=torch.long)
        location_labels = torch.zeros(self.max_objects, dtype=torch.long)
        severity_labels = torch.zeros(self.max_objects)
        type_labels = torch.zeros(self.max_objects, dtype=torch.long)
        
        for i, ann in enumerate(annotations[:self.max_objects]):
            # Normalize bbox to [0, 1]
            x, y, w, h = ann['bbox']
            img_w, img_h = image_info['width'], image_info['height']
            
            bboxes[i] = torch.tensor([
                (x + w/2) / img_w,  # center x
                (y + h/2) / img_h,  # center y
                w / img_w,          # width
                h / img_h           # height
            ])
            
            class_labels[i] = ann.get('category_id', 0)
            location_labels[i] = ann.get('damage_location_id', 0)
            severity_labels[i] = ann.get('damage_severity', 0) / 10.0  # Normalize to [0, 1]
            type_labels[i] = ann.get('damage_type_id', 0)
        
        return {
            'image': image_tensor,
            'bboxes': bboxes,
            'class_labels': class_labels,
            'location_labels': location_labels,
            'severity_labels': severity_labels,
            'type_labels': type_labels,
            'num_objects': num_objects
        }


class UnifiedTransformerTrainer:
    """Trainer for unified transformer model."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize trainer."""
        self.cfg = cfg
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Setup MLflow
        self._setup_mlflow()
        
        # Setup datasets
        self._setup_datasets()
        
        # Create model
        self.model = self._create_model()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Loss function
        self.criterion = UnifiedTransformerLoss(
            num_classes=cfg.data.num_classes,
            bbox_loss_weight=cfg.train.loss_weights.get('bbox', 5.0),
            class_loss_weight=cfg.train.loss_weights.get('class', 1.0),
            location_loss_weight=cfg.train.loss_weights.get('location', 0.5),
            severity_loss_weight=cfg.train.loss_weights.get('severity', 0.3),
            type_loss_weight=cfg.train.loss_weights.get('damage_type', 0.5)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)
        mlflow.start_run(run_name=f"{self.cfg.mlflow.run_name}_unified_transformer")
        
        # Log config
        mlflow.log_params({
            "model_type": "unified_transformer",
            "model_size": self.cfg.train.get('model_size', 'base'),
            "img_size": self.cfg.train.image.size,
            "batch_size": self.cfg.train.training.batch_size,
            "learning_rate": self.cfg.train.training.learning_rate,
            "num_epochs": self.cfg.train.training.epochs
        })
    
    def _setup_datasets(self):
        """Setup train, val, test datasets."""
        # Determine paths
        if hasattr(self.cfg.paths, 'coco_annotations'):
            annotations_file = Path(self.cfg.paths.coco_annotations)
        else:
            # Fallback to labelme annotations
            annotations_file = Path(self.cfg.paths.data_dir) / "labelme" / "annotations.json"
        
        if hasattr(self.cfg.paths, 'images_dir'):
            images_dir = Path(self.cfg.paths.images_dir)
        else:
            images_dir = Path(self.cfg.paths.processed_data.yolo_dir) / "images"
        
        logger.info(f"Loading datasets from: {images_dir}")
        logger.info(f"Using annotations: {annotations_file}")
        
        # Create datasets
        self.train_dataset = UnifiedTransformerDataset(
            images_dir=images_dir,
            annotations_file=annotations_file,
            split='train',
            img_size=self.cfg.train.image.size
        )
        
        self.val_dataset = UnifiedTransformerDataset(
            images_dir=images_dir,
            annotations_file=annotations_file,
            split='val',
            img_size=self.cfg.train.image.size
        )
        
        self.test_dataset = UnifiedTransformerDataset(
            images_dir=images_dir,
            annotations_file=annotations_file,
            split='test',
            img_size=self.cfg.train.image.size
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.train.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.train.training.batch_size,
            shuffle=False,
            num_workers=4
        )
    
    def _create_model(self):
        """Create unified transformer model."""
        model_size = self.cfg.train.get('model_size', 'base')
        
        model = create_unified_transformer(
            num_vehicle_parts=self.cfg.data.num_classes,
            num_damage_locations=21,  # From config
            num_damage_types=11,       # From config
            model_size=model_size
        )
        
        model = model.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created: {model_size}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        mlflow.log_params({
            "total_params": total_params,
            "trainable_params": trainable_params
        })
        
        return model
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        lr = self.cfg.train.training.learning_rate
        weight_decay = self.cfg.train.training.weight_decay
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.train.training.epochs
        )
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in pbar:
            # Move to device
            images = batch['image'].to(self.device)
            targets = {
                'bboxes': batch['bboxes'].to(self.device),
                'class_labels': batch['class_labels'].to(self.device),
                'location_labels': batch['location_labels'].to(self.device),
                'severity_labels': batch['severity_labels'].to(self.device),
                'type_labels': batch['type_labels'].to(self.device)
            }
            
            # Forward pass
            predictions = self.model(images)
            
            # Compute loss
            loss, loss_dict = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track loss
            epoch_losses.append(loss_dict['total_loss'])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'bbox': f"{loss_dict['bbox_loss']:.4f}",
                'cls': f"{loss_dict['class_loss']:.4f}"
            })
        
        return np.mean(epoch_losses)
    
    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        epoch_losses = []
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move to device
            images = batch['image'].to(self.device)
            targets = {
                'bboxes': batch['bboxes'].to(self.device),
                'class_labels': batch['class_labels'].to(self.device),
                'location_labels': batch['location_labels'].to(self.device),
                'severity_labels': batch['severity_labels'].to(self.device),
                'type_labels': batch['type_labels'].to(self.device)
            }
            
            # Forward pass
            predictions = self.model(images)
            
            # Compute loss
            loss, loss_dict = self.criterion(predictions, targets)
            
            epoch_losses.append(loss_dict['total_loss'])
        
        return np.mean(epoch_losses)
    
    def train(self):
        """Main training loop."""
        num_epochs = self.cfg.train.training.epochs
        patience = self.cfg.train.training.get('patience', 10)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # Print summary
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = Path(self.cfg.paths.output_dir) / "unified_transformer_checkpoint.pth"
            self._save_checkpoint(checkpoint_path, val_loss)
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                best_model_path = Path(self.cfg.paths.output_dir) / "unified_transformer_best.pth"
                self._save_checkpoint(best_model_path, val_loss)
                logger.info(f"✓ New best model! Val Loss: {val_loss:.4f}")
                
                # Log to MLflow
                mlflow.pytorch.log_model(self.model, "unified_transformer_model")
            else:
                self.patience_counter += 1
                logger.info(f"No improvement. Patience: {self.patience_counter}/{patience}")
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        logger.info(f"Training completed! Best val loss: {self.best_val_loss:.4f}")
    
    def _save_checkpoint(self, path: Path, val_loss: float):
        """Save model checkpoint."""
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': dict(self.cfg)
        }, path)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    try:
        # Create trainer
        trainer = UnifiedTransformerTrainer(cfg)
        
        # Train model
        trainer.train()
        
    finally:
        # End MLflow run
        mlflow.end_run()
    
    logger.info("Unified transformer training completed!")


if __name__ == "__main__":
    main()
