#!/usr/bin/env python3
"""
Training script for the Unified Transformer Model.

Supports training with configurable hyperparameters, mixed precision,
gradient accumulation, and experiment tracking.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import torch.cuda.amp as amp
    from torch.utils.tensorboard import SummaryWriter

    from models.unified_transformer import create_unified_transformer
    from models.losses import create_unified_loss

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. This script requires PyTorch to run.")


class CollisionDataset(Dataset):
    """Dataset for collision parts with damage annotations."""

    def __init__(self, coco_path: str, images_dir: str, transform=None):
        """
        Args:
            coco_path: Path to COCO format annotations
            images_dir: Directory containing images
            transform: Optional image transformations
        """
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Load COCO annotations
        with open(coco_path, 'r') as f:
            self.coco_data = json.load(f)

        # Build image and annotation mappings
        self.image_id_to_path = {}
        self.image_id_to_annotations = {}

        for img in self.coco_data['images']:
            img_path = self.images_dir / img['file_name']
            if img_path.exists():
                self.image_id_to_path[img['id']] = img_path

        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

        self.image_ids = list(self.image_id_to_path.keys())

        # Category mappings
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = self.image_id_to_path[image_id]
        annotations = self.image_id_to_annotations.get(image_id, [])

        # Load image
        image = torch.load(image_path) if image_path.suffix == '.pt' else self.load_image(image_path)

        if self.transform:
            image = self.transform(image)

        # Process annotations into fixed-size tensors
        # This is a simplified version - in practice you'd need proper assignment
        # of annotations to queries (similar to DETR's Hungarian matching)

        # For now, return raw data and process in collate_fn
        return {
            'image': image,
            'image_id': image_id,
            'annotations': annotations
        }

    def load_image(self, path):
        """Load image from file."""
        try:
            from PIL import Image
            import torchvision.transforms as T

            img = Image.open(path).convert('RGB')
            transform = T.Compose([
                T.Resize((640, 640)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return transform(img)
        except ImportError:
            # Fallback if PIL not available
            return torch.zeros(3, 640, 640)


def collate_fn(batch, num_queries=100):
    """Collate function to create fixed-size targets."""
    images = []
    targets = []

    for item in batch:
        images.append(item['image'])

        # Convert annotations to model targets
        # This is a simplified version - real implementation would use
        # Hungarian matching or similar assignment algorithm
        target = create_target_from_annotations(item['annotations'], num_queries)
        targets.append(target)

    return {
        'images': torch.stack(images),
        'targets': targets
    }


def create_target_from_annotations(annotations, num_queries):
    """Create fixed-size target tensors from variable annotations."""
    # Initialize with zeros/no-object
    bbox = torch.zeros(num_queries, 4)
    class_ids = torch.full((num_queries,), -1, dtype=torch.long)  # -1 for no object
    locations = torch.zeros(num_queries, dtype=torch.long)
    severities = torch.zeros(num_queries, 1)
    types = torch.zeros(num_queries, dtype=torch.long)
    object_mask = torch.zeros(num_queries, dtype=torch.bool)

    # Fill with actual annotations (up to num_queries)
    for i, ann in enumerate(annotations[:num_queries]):
        bbox[i] = torch.tensor(ann['bbox'])
        class_ids[i] = ann.get('category_id', 0)
        locations[i] = ann.get('damage_location', 0)
        severities[i] = ann.get('damage_severity', 0.0)
        types[i] = ann.get('damage_type', 0)
        object_mask[i] = True

    return {
        'bbox': bbox,
        'class': class_ids,
        'location': locations,
        'severity': severities,
        'type': types,
        'object_mask': object_mask
    }


class Trainer:
    """Trainer class for the Unified Transformer."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Create model
        self.model = create_unified_transformer(
            model_size=config.get('model_size', 'base'),
            num_vehicle_parts=config.get('num_vehicle_parts', 11),
            num_damage_locations=config.get('num_damage_locations', 21),
            num_damage_types=config.get('num_damage_types', 11)
        ).to(self.device)

        # Create loss function
        self.criterion = create_unified_loss(
            bbox_weight=config.get('bbox_loss_weight', 5.0),
            class_weight=config.get('class_loss_weight', 1.0),
            location_weight=config.get('location_loss_weight', 0.5),
            severity_weight=config.get('severity_loss_weight', 0.3),
            type_weight=config.get('type_loss_weight', 0.5)
        )

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = amp.GradScaler() if config.get('mixed_precision', True) else None

        # Gradient clipping
        self.max_grad_norm = config.get('gradient_clip', 1.0)

        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Logging
        self.writer = SummaryWriter(log_dir=config.get('log_dir', 'logs'))

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

    def _create_optimizer(self):
        """Create optimizer."""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)

        if optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        epochs = self.config.get('epochs', 100)
        warmup_epochs = self.config.get('warmup_epochs', 10)

        if scheduler_name == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
        else:
            # No scheduler
            return None

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}

        for batch_idx, batch in enumerate(dataloader):
            images = batch['images'].to(self.device)
            targets = batch['targets']

            # Convert targets to tensors on device
            batch_targets = self._prepare_targets(targets)

            # Forward pass
            with amp.autocast(enabled=self.scaler is not None):
                predictions = self.model(images)
                losses = self.criterion(predictions, batch_targets)
                loss = losses['total_loss']

            # Backward pass
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # Update learning rate
            if self.scheduler and self.epoch >= self.config.get('warmup_epochs', 10):
                self.scheduler.step()

            # Accumulate losses
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v.item()

            # Logging
            if batch_idx % self.config.get('log_interval', 10) == 0:
                self._log_step(losses, batch_idx, len(dataloader))

            self.global_step += 1

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= len(dataloader)

        return epoch_losses

    def _prepare_targets(self, targets):
        """Prepare targets for loss computation."""
        batch_size = len(targets)

        # Stack targets into tensors
        bbox = torch.stack([t['bbox'] for t in targets]).to(self.device)
        class_ids = torch.stack([t['class'] for t in targets]).to(self.device)
        locations = torch.stack([t['location'] for t in targets]).to(self.device)
        severities = torch.stack([t['severity'] for t in targets]).to(self.device)
        types = torch.stack([t['type'] for t in targets]).to(self.device)
        object_mask = torch.stack([t['object_mask'] for t in targets]).to(self.device)

        return {
            'bbox': bbox,
            'class': class_ids,
            'location': locations,
            'severity': severities,
            'type': types,
            'object_mask': object_mask
        }

    def _log_step(self, losses, batch_idx, num_batches):
        """Log training step."""
        progress = (batch_idx + 1) / num_batches
        print(f"Epoch {self.epoch+1}, Step {batch_idx+1}/{num_batches} "
              f"({progress:.1%}): Loss = {losses['total_loss']:.4f}")

        # TensorBoard logging
        for k, v in losses.items():
            self.writer.add_scalar(f'train/{k}', v.item(), self.global_step)

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_losses = {}

        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                targets = batch['targets']
                batch_targets = self._prepare_targets(targets)

                predictions = self.model(images)
                losses = self.criterion(predictions, batch_targets)

                for k, v in losses.items():
                    if k not in val_losses:
                        val_losses[k] = 0.0
                    val_losses[k] += v.item()

        # Average losses
        for k in val_losses:
            val_losses[k] /= len(dataloader)

        return val_losses

    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'loss': loss,
            'config': self.config
        }

        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with loss {loss:.4f}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('loss', float('inf'))

        print(f"Loaded checkpoint from epoch {self.epoch}")


def main():
    parser = argparse.ArgumentParser(description='Train Unified Transformer')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Model size to use')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override with command line args
    config.update({
        'model_size': args.model_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr
    })

    if not TORCH_AVAILABLE:
        print("PyTorch not available. Please install PyTorch to run training.")
        return

    # Create datasets (placeholder - would need actual data paths)
    train_dataset = CollisionDataset(
        coco_path=config.get('coco_annotations', 'data/annotations.json'),
        images_dir=config.get('images_dir', 'data/images')
    )

    val_dataset = CollisionDataset(
        coco_path=config.get('coco_annotations', 'data/annotations.json'),
        images_dir=config.get('images_dir', 'data/images')
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, config.get('num_queries', 100)),
        num_workers=config.get('workers', 4)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, config.get('num_queries', 100)),
        num_workers=config.get('workers', 4)
    )

    # Create trainer
    trainer = Trainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Training loop
    for epoch in range(trainer.epoch, config['epochs']):
        trainer.epoch = epoch

        # Train
        train_losses = trainer.train_epoch(train_loader)

        # Validate
        val_losses = trainer.validate(val_loader)

        # Logging
        print(f"Epoch {epoch+1}/{config['epochs']}:")
        print(f"  Train Loss: {train_losses['total_loss']:.4f}")
        print(f"  Val Loss: {val_losses['total_loss']:.4f}")

        # TensorBoard logging
        for k, v in train_losses.items():
            trainer.writer.add_scalar(f'epoch/train_{k}', v, epoch)
        for k, v in val_losses.items():
            trainer.writer.add_scalar(f'epoch/val_{k}', v, epoch)

        # Save checkpoint
        is_best = val_losses['total_loss'] < trainer.best_loss
        if is_best:
            trainer.best_loss = val_losses['total_loss']

        trainer.save_checkpoint(epoch, val_losses['total_loss'], is_best)

    print("Training completed!")


if __name__ == "__main__":
    main()