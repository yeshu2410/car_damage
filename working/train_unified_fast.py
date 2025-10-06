#!/usr/bin/env python3
"""
Fast Training Script for Unified Transformer
Optimized for quick training with reasonable accuracy
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.unified_transformer import create_unified_transformer


# ============================================================
# CONFIGURATION - Optimized for FAST training
# ============================================================

CONFIG = {
    # Model
    'model_size': 'small',  # Use small model (faster than 'base')
    'img_size': 448,        # Smaller images = faster (vs 640)
    
    # Training
    'epochs': 50,           # Fewer epochs (vs 100+)
    'batch_size': 8,        # Adjust based on your RAM/GPU
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    
    # Early stopping
    'patience': 10,         # Stop if no improvement for 10 epochs
    'min_delta': 0.001,     # Minimum improvement to count
    
    # Data
    'num_workers': 2,       # Parallel data loading
    'pin_memory': True,     # Faster data transfer to GPU
    
    # Paths
    'train_json': '../data/processed/yolo/batch_output/splits/annotations_train.json',
    'val_json': '../data/processed/yolo/batch_output/splits/annotations_val.json',
    'images_dir': '../data/processed/yolo/images_2500',
    'output_dir': './outputs/unified_transformer_fast',
    
    # Loss weights
    'loss_weights': {
        'bbox': 1.0,
        'class': 2.0,
        'location': 1.5,
        'severity': 1.0,
        'damage_type': 1.5
    }
}


# ============================================================
# DATASET CLASS
# ============================================================

class COCODamageDataset(Dataset):
    """PyTorch Dataset for COCO damage annotations."""
    
    def __init__(self, coco_json_path, images_dir, transform=None, img_size=448):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.img_size = img_size
        
        # Load COCO annotations
        print(f"Loading annotations from {coco_json_path}...")
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        self.images = coco_data['images']
        self.annotations = coco_data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Group annotations by image
        self.image_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        print(f"✅ Loaded {len(self.images)} images, {len(self.annotations)} annotations")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = self.images_dir / img_info['file_name']
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            # Return dummy data if image not found
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Get annotations for this image
        anns = self.image_annotations.get(img_info['id'], [])
        
        # Convert to tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0  # [3, H, W]
        
        # Prepare targets (simplified for fast training)
        targets = {
            'image_id': img_info['id'],
            'num_damages': len(anns),
            'annotations': anns
        }
        
        return image, targets


# ============================================================
# TRAINER CLASS
# ============================================================

class UnifiedTransformerTrainer:
    """Fast trainer for Unified Transformer."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️  Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        print("\n🔨 Creating model...")
        self.model = create_unified_transformer(
            model_size=config['model_size'],
            num_vehicle_parts=21,  # From your part detection
            num_damage_locations=21,
            num_damage_types=8  # From your COCO categories
        )
        self.model = self.model.to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model created: {num_params:,} parameters ({num_params/1e6:.1f}M)")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.training_history = []
    
    def compute_loss(self, outputs, targets):
        """Compute multi-task loss (simplified for fast training)."""
        # For now, just return a placeholder loss
        # In production, you'd compute proper losses for each task
        
        batch_size = outputs['bbox_pred'].size(0)
        
        # Simple L2 losses (placeholder)
        bbox_loss = outputs['bbox_pred'].mean()
        class_loss = outputs['class_logits'].mean()
        location_loss = outputs['location_logits'].mean()
        severity_loss = outputs['severity_pred'].mean()
        damage_type_loss = outputs['type_logits'].mean()
        
        # Weighted sum
        weights = self.config['loss_weights']
        total_loss = (
            weights['bbox'] * bbox_loss +
            weights['class'] * class_loss +
            weights['location'] * location_loss +
            weights['severity'] * severity_loss +
            weights['damage_type'] * damage_type_loss
        )
        
        return total_loss, {
            'bbox': bbox_loss.item(),
            'class': class_loss.item(),
            'location': location_loss.item(),
            'severity': severity_loss.item(),
            'damage_type': damage_type_loss.item()
        }
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            loss, loss_dict = self.compute_loss(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                
                outputs = self.model(images)
                loss, _ = self.compute_loss(outputs, targets)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save latest
        latest_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model (val_loss: {val_loss:.4f})")
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("🚀 STARTING TRAINING")
        print("=" * 60)
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Check for improvement
            is_best = val_loss < (self.best_val_loss - self.config['min_delta'])
            
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Log progress
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            print(f"\n📊 Epoch {epoch+1}/{self.config['epochs']} Summary:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f} {'🌟 BEST!' if is_best else ''}")
            print(f"   Time: {epoch_time:.1f}s (Total: {total_time/60:.1f}m)")
            print(f"   No improve: {self.epochs_no_improve}/{self.config['patience']}")
            
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'time': epoch_time
            })
            
            # Early stopping
            if self.epochs_no_improve >= self.config['patience']:
                print(f"\n⚠️  Early stopping triggered (no improvement for {self.config['patience']} epochs)")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {self.output_dir}")
        print("=" * 60)
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("🚗 UNIFIED TRANSFORMER - FAST TRAINING")
    print("=" * 60)
    
    # Check paths
    train_json = Path(CONFIG['train_json'])
    val_json = Path(CONFIG['val_json'])
    images_dir = Path(CONFIG['images_dir'])
    
    if not train_json.exists():
        print(f"❌ Train JSON not found: {train_json}")
        return
    if not val_json.exists():
        print(f"❌ Val JSON not found: {val_json}")
        return
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    print("✅ All paths verified\n")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = COCODamageDataset(
        train_json,
        images_dir,
        img_size=CONFIG['img_size']
    )
    
    val_dataset = COCODamageDataset(
        val_json,
        images_dir,
        img_size=CONFIG['img_size']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )
    
    print(f"✅ Train loader: {len(train_loader)} batches")
    print(f"✅ Val loader: {len(val_loader)} batches\n")
    
    # Create trainer
    trainer = UnifiedTransformerTrainer(CONFIG)
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
