"""
YOLO11m training script using Ultralytics.
Trains YOLO model for collision parts object detection.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import hydra
import mlflow
import yaml
from loguru import logger
from omegaconf import DictConfig
from ultralytics import YOLO


class YOLOTrainer:
    """Trainer class for YOLO collision parts detection."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize YOLO trainer with configuration."""
        self.cfg = cfg
        
        # Setup logging
        logger.info(f"Configuration: {cfg}")
        
        # Initialize MLflow
        self._setup_mlflow()
        
        # Prepare dataset
        self.dataset_path = self._prepare_dataset()
        
        # Initialize YOLO model
        self.model = self._create_model()
        
        # Training results storage
        self.results = None
        self.best_model_path = None
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)
        
        # Start MLflow run
        mlflow.start_run(run_name=f"{self.cfg.mlflow.run_name}_yolo")
        
        # Log configuration
        mlflow.log_params({
            "model_name": self.cfg.train.model.name,
            "epochs": self.cfg.train.training.epochs,
            "batch_size": self.cfg.train.training.batch_size,
            "learning_rate": self.cfg.train.training.learning_rate,
            "image_size": self.cfg.train.image.size,
            "optimizer": self.cfg.train.optimizer.type,
            "num_classes": self.cfg.data.num_classes
        })
    
    def _prepare_dataset(self) -> Path:
        """Prepare YOLO dataset structure and configuration."""
        yolo_dir = Path(self.cfg.data.processed_data.yolo_dir)
        dataset_yaml_path = yolo_dir / "dataset.yaml"
        
        if not dataset_yaml_path.exists():
            raise FileNotFoundError(f"YOLO dataset not found at {dataset_yaml_path}. "
                                   "Please run data preparation scripts first.")
        
        # Load and update dataset configuration
        with open(dataset_yaml_path, "r") as f:
            dataset_config = yaml.safe_load(f)
        
        # Update paths to be absolute
        dataset_config["path"] = str(yolo_dir.absolute())
        dataset_config["train"] = "images"
        dataset_config["val"] = "images"  # Will split during training
        
        # Create train/val split if needed
        self._create_train_val_split(yolo_dir)
        
        # Save updated dataset configuration
        updated_yaml_path = yolo_dir / "dataset_train.yaml"
        with open(updated_yaml_path, "w") as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Dataset prepared at: {updated_yaml_path}")
        return updated_yaml_path
    
    def _create_train_val_split(self, yolo_dir: Path):
        """Create train/val split for YOLO training."""
        images_dir = yolo_dir / "images"
        labels_dir = yolo_dir / "labels"
        
        # Create train/val directories
        train_images_dir = yolo_dir / "train" / "images"
        val_images_dir = yolo_dir / "val" / "images"
        train_labels_dir = yolo_dir / "train" / "labels"
        val_labels_dir = yolo_dir / "val" / "labels"
        
        for directory in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        if len(image_files) == 0:
            logger.warning("No images found in dataset")
            return
        
        # Split ratio from config
        val_ratio = self.cfg.data.split_ratios.val / (
            self.cfg.data.split_ratios.train + self.cfg.data.split_ratios.val
        )
        
        # Shuffle and split
        import random
        random.seed(self.cfg.data.split_seed)
        random.shuffle(image_files)
        
        split_idx = int(len(image_files) * (1 - val_ratio))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Copy files
        for img_file in train_files:
            # Copy image
            shutil.copy2(img_file, train_images_dir / img_file.name)
            
            # Copy corresponding label
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, train_labels_dir / label_file.name)
        
        for img_file in val_files:
            # Copy image
            shutil.copy2(img_file, val_images_dir / img_file.name)
            
            # Copy corresponding label
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, val_labels_dir / label_file.name)
        
        logger.info(f"Created train/val split: {len(train_files)} train, {len(val_files)} val")
        
        # Update dataset configuration to use split directories
        dataset_config = {
            "path": str(yolo_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "nc": self.cfg.data.num_classes,
            "names": self.cfg.data.classes
        }
        
        dataset_yaml_path = yolo_dir / "dataset_train.yaml"
        with open(dataset_yaml_path, "w") as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
    
    def _create_model(self) -> YOLO:
        """Create YOLO model."""
        model_name = self.cfg.train.model.name
        
        if self.cfg.train.model.pretrained:
            # Load pretrained model
            model = YOLO(f"{model_name}.pt")
            logger.info(f"Loaded pretrained {model_name} model")
        else:
            # Load model architecture only
            model = YOLO(f"{model_name}.yaml")
            logger.info(f"Created {model_name} model from scratch")
        
        return model
    
    def train(self):
        """Train YOLO model."""
        logger.info("Starting YOLO training...")
        
        # Training arguments
        train_args = {
            "data": str(self.dataset_path),
            "epochs": self.cfg.train.training.epochs,
            "imgsz": self.cfg.train.image.size,
            "batch": self.cfg.train.training.batch_size,
            "lr0": self.cfg.train.training.learning_rate,
            "weight_decay": self.cfg.train.training.weight_decay,
            "momentum": self.cfg.train.training.momentum,
            "warmup_epochs": self.cfg.train.training.warmup_epochs,
            "warmup_momentum": self.cfg.train.training.warmup_momentum,
            "warmup_bias_lr": self.cfg.train.training.warmup_bias_lr,
            "box": self.cfg.train.loss.box_loss_gain,
            "cls": self.cfg.train.loss.cls_loss_gain,
            "dfl": self.cfg.train.loss.dfl_loss_gain,
            "hsv_h": self.cfg.train.augmentation.hsv_h,
            "hsv_s": self.cfg.train.augmentation.hsv_s,
            "hsv_v": self.cfg.train.augmentation.hsv_v,
            "degrees": self.cfg.train.augmentation.degrees,
            "translate": self.cfg.train.augmentation.translate,
            "scale": self.cfg.train.augmentation.scale,
            "shear": self.cfg.train.augmentation.shear,
            "perspective": self.cfg.train.augmentation.perspective,
            "flipud": self.cfg.train.augmentation.flipud,
            "fliplr": self.cfg.train.augmentation.fliplr,
            "mosaic": self.cfg.train.augmentation.mosaic,
            "mixup": self.cfg.train.augmentation.mixup,
            "copy_paste": self.cfg.train.augmentation.copy_paste,
            "conf": self.cfg.train.nms.conf_threshold,
            "iou": self.cfg.train.nms.iou_threshold,
            "max_det": self.cfg.train.nms.max_detections,
            "save_json": self.cfg.train.validation.save_json,
            "save_hybrid": self.cfg.train.validation.save_hybrid,
            "project": str(Path(self.cfg.paths.output_dir) / "yolo_runs"),
            "name": "collision_parts_detection",
            "exist_ok": True,
            "pretrained": self.cfg.train.model.pretrained,
            "optimizer": self.cfg.train.optimizer.type,
            "verbose": True,
            "seed": self.cfg.seed,
            "deterministic": True,
            "single_cls": False,
            "rect": False,
            "cos_lr": self.cfg.train.scheduler.type == "cosine",
            "close_mosaic": 10,  # Disable mosaic in last 10 epochs
            "resume": False,
            "amp": True,  # Automatic Mixed Precision
            "fraction": 1.0,  # Use full dataset
            "profile": False,
            "freeze": None,
            "multi_scale": False,
            "overlap_mask": True,
            "mask_ratio": 4,
            "dropout": 0.0,
            "val": True,
            "split": "val",
            "save": True,
            "save_period": -1,
            "cache": False,
            "device": None,  # Auto-select device
            "workers": 8,
            "patience": 50,
            "plots": True,
            "overlap_mask": True,
            "mask_ratio": 4
        }
        
        # Start training
        try:
            self.results = self.model.train(**train_args)
            
            # Get best model path
            if hasattr(self.results, 'save_dir'):
                self.best_model_path = Path(self.results.save_dir) / "weights" / "best.pt"
            else:
                # Fallback to default path
                run_dir = Path(self.cfg.paths.output_dir) / "yolo_runs" / "collision_parts_detection"
                self.best_model_path = run_dir / "weights" / "best.pt"
            
            logger.info(f"Training completed. Best model saved at: {self.best_model_path}")
            
            # Log training results
            self._log_results()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _log_results(self):
        """Log training results to MLflow."""
        if self.results is None:
            logger.warning("No training results to log")
            return
        
        # Log final metrics
        if hasattr(self.results, 'results_dict'):
            results_dict = self.results.results_dict
            
            # Log key metrics
            metrics_to_log = {
                "final_mAP50": results_dict.get("metrics/mAP50(B)", 0.0),
                "final_mAP50_95": results_dict.get("metrics/mAP50-95(B)", 0.0),
                "final_precision": results_dict.get("metrics/precision(B)", 0.0),
                "final_recall": results_dict.get("metrics/recall(B)", 0.0),
                "final_train_loss": results_dict.get("train/box_loss", 0.0),
                "final_val_loss": results_dict.get("val/box_loss", 0.0)
            }
            
            for metric_name, value in metrics_to_log.items():
                if value is not None:
                    mlflow.log_metric(metric_name, float(value))
        
        # Log best model
        if self.best_model_path and self.best_model_path.exists():
            mlflow.log_artifact(str(self.best_model_path), "models")
            
            # Copy best model to output directory
            output_model_path = Path(self.cfg.paths.output_dir) / "best_yolo_model.pt"
            shutil.copy2(self.best_model_path, output_model_path)
            logger.info(f"Best model copied to: {output_model_path}")
        
        # Log results CSV if exists
        if hasattr(self.results, 'save_dir'):
            results_csv_path = Path(self.results.save_dir) / "results.csv"
            if results_csv_path.exists():
                mlflow.log_artifact(str(results_csv_path), "training_logs")
                
                # Copy results to output directory
                output_csv_path = Path(self.cfg.paths.output_dir) / "yolo_results.csv"
                shutil.copy2(results_csv_path, output_csv_path)
                logger.info(f"Training results saved to: {output_csv_path}")
        
        # Log training plots
        if hasattr(self.results, 'save_dir'):
            plots_dir = Path(self.results.save_dir)
            for plot_file in plots_dir.glob("*.png"):
                mlflow.log_artifact(str(plot_file), "plots")
    
    def evaluate(self):
        """Evaluate trained model."""
        if not self.best_model_path or not self.best_model_path.exists():
            logger.warning("No best model found for evaluation")
            return
        
        logger.info("Evaluating YOLO model...")
        
        # Load best model
        model = YOLO(str(self.best_model_path))
        
        # Validate on test set
        val_results = model.val(
            data=str(self.dataset_path),
            split="val",
            imgsz=self.cfg.train.image.size,
            batch=self.cfg.train.training.batch_size,
            conf=self.cfg.train.nms.conf_threshold,
            iou=self.cfg.train.nms.iou_threshold,
            max_det=self.cfg.train.nms.max_detections,
            save_json=True,
            plots=True,
            verbose=True
        )
        
        # Log evaluation metrics
        if hasattr(val_results, 'box'):
            box_metrics = val_results.box
            eval_metrics = {
                "eval_mAP50": float(box_metrics.map50),
                "eval_mAP50_95": float(box_metrics.map),
                "eval_precision": float(box_metrics.mp),
                "eval_recall": float(box_metrics.mr),
                "eval_f1": float(box_metrics.f1)
            }
            
            for metric_name, value in eval_metrics.items():
                mlflow.log_metric(metric_name, value)
            
            logger.info(f"Evaluation completed. mAP@0.5: {box_metrics.map50:.4f}, "
                       f"mAP@0.5:0.95: {box_metrics.map:.4f}")
            
            # Save evaluation results
            eval_results_path = Path(self.cfg.paths.output_dir) / "yolo_evaluation.json"
            with open(eval_results_path, "w") as f:
                json.dump(eval_metrics, f, indent=2)
            
            mlflow.log_artifact(str(eval_results_path))
    
    def export_model(self):
        """Export model to different formats."""
        if not self.best_model_path or not self.best_model_path.exists():
            logger.warning("No best model found for export")
            return
        
        logger.info("Exporting YOLO model...")
        
        # Load best model
        model = YOLO(str(self.best_model_path))
        
        # Export to different formats
        export_formats = self.cfg.train.export.format
        
        for format_name in export_formats:
            try:
                exported_path = model.export(
                    format=format_name,
                    imgsz=self.cfg.train.image.size,
                    optimize=self.cfg.train.export.optimize,
                    simplify=self.cfg.train.export.simplify
                )
                
                logger.info(f"Model exported to {format_name}: {exported_path}")
                
                # Log exported model
                if Path(exported_path).exists():
                    mlflow.log_artifact(str(exported_path), f"exported_models/{format_name}")
                
            except Exception as e:
                logger.error(f"Failed to export to {format_name}: {e}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Update config to use YOLO training config
    cfg.train = cfg.train.yolo11m if hasattr(cfg.train, 'yolo11m') else cfg.train
    
    # Create trainer and start training
    trainer = YOLOTrainer(cfg)
    
    try:
        # Train model
        trainer.train()
        
        # Evaluate model
        trainer.evaluate()
        
        # Export model
        trainer.export_model()
        
    finally:
        # End MLflow run
        mlflow.end_run()
    
    logger.info("YOLO training pipeline completed!")


if __name__ == "__main__":
    main()