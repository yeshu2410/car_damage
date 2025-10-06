"""
ResNet50 training script with Hydra configuration and MLflow logging.
Trains end-to-end ResNet50 model for collision parts classification.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import (
    f1_score, hamming_loss, accuracy_score, precision_score, recall_score,
    classification_report, multilabel_confusion_matrix
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset_resnet import CollisionPartsDataset, create_data_loaders
from ..models.resnet_endtoend import DamageNet
from ..models.losses import create_loss_function, calculate_class_weights


class ModelTrainer:
    """Trainer class for ResNet50 collision parts classification."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize trainer with configuration."""
        self.cfg = cfg
        self.device = self._get_device()
        
        # Setup logging
        logger.info(f"Using device: {self.device}")
        logger.info(f"Configuration: {cfg}")
        
        # Initialize MLflow
        self._setup_mlflow()
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize data loaders
        self.train_loader, self.val_loader, self.test_loader = self._create_data_loaders()
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.epoch = 0
        self.best_val_f1 = 0.0
        self.best_thresholds = None
        self.training_history = {
            "train_loss": [], 
            "val_loss": [], 
            "val_f1_micro": [], 
            "val_f1_macro": [], 
            "val_hamming_loss": [], 
            "val_exact_match": [],
            "val_precision_micro": [],
            "val_recall_micro": []
        }
    
    def _get_device(self) -> torch.device:
        """Get training device."""
        if self.cfg.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.cfg.device)
        return device
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)
        
        # Start MLflow run
        mlflow.start_run(run_name=self.cfg.mlflow.run_name)
        
        # Log configuration
        mlflow.log_params({
            "model_name": self.cfg.train.model.name,
            "learning_rate": self.cfg.train.training.learning_rate,
            "batch_size": self.cfg.train.training.batch_size,
            "epochs": self.cfg.train.training.epochs,
            "optimizer": self.cfg.train.optimizer.type,
            "loss_type": self.cfg.train.loss.type,
            "num_classes": self.cfg.data.num_classes,
            "image_size": self.cfg.data.image_size
        })
    
    def _create_model(self) -> DamageNet:
        """Create and initialize model."""
        model = DamageNet(
            num_classes=self.cfg.data.num_classes,
            pretrained=self.cfg.train.model.pretrained,
            hidden_dims=[512, 256],  # Can be made configurable
            dropout=self.cfg.train.model.dropout
        )
        
        model.to(self.device)
        
        # Log model parameters
        param_count = model.get_parameter_count()
        logger.info(f"Model parameters: {param_count}")
        mlflow.log_params(param_count)
        
        return model
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders."""
        # Paths
        train_annotations = Path(self.cfg.data.processed_data.train_annotations)
        val_annotations = Path(self.cfg.data.processed_data.val_annotations)
        test_annotations = Path(self.cfg.data.processed_data.test_annotations)
        image_base_dir = Path(self.cfg.data.raw_data.vehide_images_dir)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_annotations=train_annotations,
            val_annotations=val_annotations,
            test_annotations=test_annotations,
            image_base_dir=image_base_dir,
            class_names=self.cfg.data.classes,
            batch_size=self.cfg.train.training.batch_size,
            image_size=self.cfg.data.image_size,
            num_workers=4
        )
        
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function with class weights."""
        # Calculate class weights if needed
        pos_weight = None
        if self.cfg.train.loss.get("pos_weight") == "auto":
            # Calculate from training data
            class_weights = self.train_loader.dataset.get_class_weights()
            pos_weight = class_weights.to(self.device)
            logger.info(f"Calculated class weights: {class_weights}")
        
        criterion = create_loss_function(
            loss_type=self.cfg.train.loss.type,
            num_classes=self.cfg.data.num_classes,
            pos_weight=pos_weight
        )
        
        return criterion
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        if self.cfg.train.optimizer.type == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.cfg.train.optimizer.lr,
                weight_decay=self.cfg.train.optimizer.weight_decay,
                betas=self.cfg.train.optimizer.betas
            )
        elif self.cfg.train.optimizer.type == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.cfg.train.optimizer.lr,
                weight_decay=self.cfg.train.optimizer.weight_decay,
                momentum=self.cfg.train.training.momentum
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.train.optimizer.type}")
        
        return optimizer
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if self.cfg.train.scheduler.type == "cosine_annealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.train.scheduler.T_max,
                eta_min=self.cfg.train.scheduler.eta_min
            )
        elif self.cfg.train.scheduler.type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=10,
                verbose=True
            )
        
        return scheduler
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (images, targets, metadata) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # Log batch metrics to MLflow
            if batch_idx % 100 == 0:
                mlflow.log_metric("batch_loss", loss.item(), step=self.epoch * num_batches + batch_idx)
        
        avg_epoch_loss = total_loss / num_batches
        return avg_epoch_loss
    
    def validate_epoch(self) -> Tuple[Dict[str, float], np.ndarray]:
        """Validate for one epoch."""
        # Handle empty validation set
        if len(self.val_loader.dataset) == 0:
            logger.warning("No validation samples available. Skipping validation.")
            empty_metrics = {
                "val_loss": 0.0,
                "val_f1_micro": 0.0,
                "val_f1_macro": 0.0,
                "val_hamming_loss": 0.0,
                "val_exact_match": 0.0,
                "val_precision_micro": 0.0,
                "val_recall_micro": 0.0
            }
            empty_thresholds = np.full(self.cfg.data.num_classes, 0.5)
            return empty_metrics, empty_thresholds
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets, metadata in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Collect predictions and targets
                predictions = torch.sigmoid(outputs)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                total_loss += loss.item()
        
        # Calculate metrics only if we have predictions
        if not all_predictions:
            logger.warning("No predictions collected during validation.")
            return {
                "val_loss": 0.0,
                "val_accuracy": 0.0,
                "val_f1_macro": 0.0,
                "val_f1_micro": 0.0,
                "val_precision": 0.0,
                "val_recall": 0.0
            }
        
        # Calculate metrics
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # Find optimal thresholds
        optimal_thresholds = self._find_optimal_thresholds(all_predictions, all_targets)
        
        # Calculate metrics with optimal thresholds
        binary_predictions = (all_predictions >= optimal_thresholds).astype(int)
        
        metrics = {
            "val_loss": total_loss / len(self.val_loader),
            "val_f1_micro": f1_score(all_targets, binary_predictions, average="micro"),
            "val_f1_macro": f1_score(all_targets, binary_predictions, average="macro"),
            "val_hamming_loss": hamming_loss(all_targets, binary_predictions),
            "val_exact_match": accuracy_score(all_targets, binary_predictions),
            "val_precision_micro": precision_score(all_targets, binary_predictions, average="micro"),
            "val_recall_micro": recall_score(all_targets, binary_predictions, average="micro")
        }
        
        return metrics, optimal_thresholds
    
    def _find_optimal_thresholds(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Find optimal thresholds for each class using F1 score."""
        optimal_thresholds = np.zeros(self.cfg.data.num_classes)
        
        for class_idx in range(self.cfg.data.num_classes):
            best_f1 = 0.0
            best_threshold = 0.5
            
            # Try different thresholds
            for threshold in np.arange(0.1, 0.9, 0.05):
                class_predictions = (predictions[:, class_idx] >= threshold).astype(int)
                class_targets = targets[:, class_idx]
                
                if class_targets.sum() > 0:  # Only if there are positive samples
                    f1 = f1_score(class_targets, class_predictions, average="binary", zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
            
            optimal_thresholds[class_idx] = best_threshold
        
        return optimal_thresholds
    
    def save_checkpoint(self, metrics: Dict[str, float], thresholds: np.ndarray, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "thresholds": thresholds.tolist(),
            "training_history": self.training_history,
            "config": self.cfg
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.cfg.paths.output_dir) / f"checkpoint_epoch_{self.epoch}.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_model_path = Path(self.cfg.paths.output_dir) / "best_model.pth"
            torch.save(checkpoint, best_model_path)
            
            # Save thresholds separately
            thresholds_path = Path(self.cfg.paths.output_dir) / "thresholds.json"
            thresholds_dict = {
                "thresholds": thresholds.tolist(),
                "class_names": self.cfg.data.classes
            }
            with open(thresholds_path, "w") as f:
                json.dump(thresholds_dict, f, indent=2)
            
            # Log to MLflow
            mlflow.log_artifact(str(best_model_path))
            mlflow.log_artifact(str(thresholds_path))
            
            logger.info(f"Saved best model with F1 score: {metrics['val_f1_macro']:.4f}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.cfg.train.training.epochs):
            self.epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_metrics, optimal_thresholds = self.validate_epoch()
            
            # Update learning rate
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics["val_f1_macro"])
            else:
                self.scheduler.step()
            
            # Update training history
            self.training_history["train_loss"].append(train_loss)
            for key, value in val_metrics.items():
                self.training_history[key].append(value)
            
            # Log metrics
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                       f"Val F1 Macro: {val_metrics['val_f1_macro']:.4f}, "
                       f"Val Hamming Loss: {val_metrics['val_hamming_loss']:.4f}")
            
            # Log to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for key, value in val_metrics.items():
                mlflow.log_metric(key, value, step=epoch)
            
            # Save checkpoint
            is_best = val_metrics["val_f1_macro"] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics["val_f1_macro"]
                self.best_thresholds = optimal_thresholds
            
            self.save_checkpoint(val_metrics, optimal_thresholds, is_best)
            
            # Early stopping
            if hasattr(self.cfg.train, "early_stopping"):
                patience = self.cfg.train.early_stopping.patience
                if epoch > patience:
                    recent_f1_scores = self.training_history["val_f1_macro"][-patience:]
                    if max(recent_f1_scores) <= self.best_val_f1 - self.cfg.train.early_stopping.min_delta:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
        
        # Final evaluation on test set
        self.evaluate_test()
        
        # End MLflow run
        mlflow.end_run()
        
        logger.info("Training completed!")
    
    def evaluate_test(self):
        """Evaluate on test set with best model."""
        # Load best model
        best_model_path = Path(self.cfg.paths.output_dir) / "best_model.pth"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets, metadata in tqdm(self.test_loader, desc="Test Evaluation"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                predictions = torch.sigmoid(outputs)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # Use best thresholds
        if self.best_thresholds is not None:
            binary_predictions = (all_predictions >= self.best_thresholds).astype(int)
        else:
            binary_predictions = (all_predictions >= 0.5).astype(int)
        
        # Calculate test metrics
        test_metrics = {
            "test_f1_micro": f1_score(all_targets, binary_predictions, average="micro"),
            "test_f1_macro": f1_score(all_targets, binary_predictions, average="macro"),
            "test_hamming_loss": hamming_loss(all_targets, binary_predictions),
            "test_exact_match": accuracy_score(all_targets, binary_predictions),
            "test_precision_micro": precision_score(all_targets, binary_predictions, average="micro"),
            "test_recall_micro": recall_score(all_targets, binary_predictions, average="micro")
        }
        
        # Log test metrics
        for key, value in test_metrics.items():
            mlflow.log_metric(key, value)
        
        logger.info(f"Test Results: {test_metrics}")
        
        # Generate classification report
        report = classification_report(
            all_targets, binary_predictions, 
            target_names=self.cfg.data.classes, 
            output_dict=True, zero_division=0
        )
        
        # Save test results
        results_path = Path(self.cfg.paths.output_dir) / "test_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "test_metrics": test_metrics,
                "classification_report": report,
                "thresholds": self.best_thresholds.tolist() if self.best_thresholds is not None else None
            }, f, indent=2)
        
        mlflow.log_artifact(str(results_path))


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Create trainer and start training
    trainer = ModelTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()