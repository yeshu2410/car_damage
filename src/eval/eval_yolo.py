"""
YOLO model evaluation script.
Loads YOLO model, runs validation, and computes mAP metrics.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

import hydra
import yaml
from loguru import logger
from omegaconf import DictConfig
from ultralytics import YOLO


class YOLOEvaluator:
    """Evaluator for trained YOLO models."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize evaluator with configuration."""
        self.cfg = cfg
        
        # Setup paths
        self.model_path = Path(cfg.paths.output_dir) / "best_yolo_model.pt"
        self.dataset_path = Path(cfg.data.processed_data.yolo_dir) / "dataset_train.yaml"
        self.results_dir = Path(cfg.paths.output_dir) / "evaluation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        
        # Load dataset config
        self.dataset_config = self._load_dataset_config()
        
        logger.info("YOLO evaluator initialized")
    
    def _load_model(self) -> YOLO:
        """Load trained YOLO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        
        model = YOLO(str(self.model_path))
        logger.info(f"Loaded YOLO model from: {self.model_path}")
        
        return model
    
    def _load_dataset_config(self) -> Dict:
        """Load YOLO dataset configuration."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {self.dataset_path}")
        
        with open(self.dataset_path, "r") as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded dataset config from: {self.dataset_path}")
        return config
    
    def evaluate_validation_set(self) -> Dict:
        """Evaluate model on validation set."""
        logger.info("Evaluating YOLO model on validation set...")
        
        # Run validation
        val_results = self.model.val(
            data=str(self.dataset_path),
            split="val",
            imgsz=self.cfg.train.image.size,
            batch=self.cfg.train.training.batch_size,
            conf=self.cfg.train.nms.conf_threshold,
            iou=self.cfg.train.nms.iou_threshold,
            max_det=self.cfg.train.nms.max_detections,
            save_json=True,
            plots=True,
            verbose=True,
            project=str(self.results_dir),
            name="yolo_validation"
        )
        
        # Extract metrics
        if hasattr(val_results, 'box'):
            box_metrics = val_results.box
            
            metrics = {
                "mAP50": float(box_metrics.map50),
                "mAP50_95": float(box_metrics.map),
                "precision": float(box_metrics.mp),
                "recall": float(box_metrics.mr),
                "f1_score": float(box_metrics.f1),
                "conf_threshold": self.cfg.train.nms.conf_threshold,
                "iou_threshold": self.cfg.train.nms.iou_threshold
            }
            
            # Per-class metrics
            if hasattr(box_metrics, 'ap_class_index') and hasattr(box_metrics, 'ap'):
                per_class_metrics = {}
                class_names = self.dataset_config.get("names", [])
                
                for i, class_idx in enumerate(box_metrics.ap_class_index):
                    if i < len(class_names):
                        class_name = class_names[class_idx]
                        per_class_metrics[class_name] = {
                            "ap50": float(box_metrics.ap50[i]) if i < len(box_metrics.ap50) else 0.0,
                            "ap50_95": float(box_metrics.ap[i]) if i < len(box_metrics.ap) else 0.0
                        }
                
                metrics["per_class"] = per_class_metrics
            
        else:
            logger.warning("No box metrics found in validation results")
            metrics = {
                "mAP50": 0.0,
                "mAP50_95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
        
        return metrics
    
    def evaluate_test_set(self) -> Dict:
        """Evaluate model on test set."""
        logger.info("Evaluating YOLO model on test set...")
        
        # Create test dataset config
        test_dataset_config = self.dataset_config.copy()
        test_dataset_config["val"] = "test/images"  # Use test split
        
        test_dataset_path = self.results_dir / "test_dataset.yaml"
        with open(test_dataset_path, "w") as f:
            yaml.dump(test_dataset_config, f, default_flow_style=False)
        
        # Run evaluation on test set
        test_results = self.model.val(
            data=str(test_dataset_path),
            split="val",  # YOLO uses 'val' parameter for any validation split
            imgsz=self.cfg.train.image.size,
            batch=self.cfg.train.training.batch_size,
            conf=self.cfg.train.nms.conf_threshold,
            iou=self.cfg.train.nms.iou_threshold,
            max_det=self.cfg.train.nms.max_detections,
            save_json=True,
            plots=True,
            verbose=True,
            project=str(self.results_dir),
            name="yolo_test"
        )
        
        # Extract metrics (same structure as validation)
        if hasattr(test_results, 'box'):
            box_metrics = test_results.box
            
            metrics = {
                "mAP50": float(box_metrics.map50),
                "mAP50_95": float(box_metrics.map),
                "precision": float(box_metrics.mp),
                "recall": float(box_metrics.mr),
                "f1_score": float(box_metrics.f1),
                "conf_threshold": self.cfg.train.nms.conf_threshold,
                "iou_threshold": self.cfg.train.nms.iou_threshold
            }
            
            # Per-class metrics
            if hasattr(box_metrics, 'ap_class_index') and hasattr(box_metrics, 'ap'):
                per_class_metrics = {}
                class_names = self.dataset_config.get("names", [])
                
                for i, class_idx in enumerate(box_metrics.ap_class_index):
                    if i < len(class_names):
                        class_name = class_names[class_idx]
                        per_class_metrics[class_name] = {
                            "ap50": float(box_metrics.ap50[i]) if i < len(box_metrics.ap50) else 0.0,
                            "ap50_95": float(box_metrics.ap[i]) if i < len(box_metrics.ap) else 0.0
                        }
                
                metrics["per_class"] = per_class_metrics
            
        else:
            logger.warning("No box metrics found in test results")
            metrics = {
                "mAP50": 0.0,
                "mAP50_95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
        
        return metrics
    
    def benchmark_inference_speed(self) -> Dict:
        """Benchmark inference speed on sample images."""
        logger.info("Benchmarking YOLO inference speed...")
        
        # Get sample images from validation set
        val_images_dir = Path(self.dataset_config["path"]) / self.dataset_config["val"]
        sample_images = list(val_images_dir.glob("*.jpg"))[:100]  # Use first 100 images
        
        if not sample_images:
            logger.warning("No sample images found for benchmarking")
            return {"inference_time_ms": 0.0, "fps": 0.0}
        
        # Warm up
        for _ in range(5):
            self.model.predict(str(sample_images[0]), verbose=False)
        
        # Benchmark
        import time
        start_time = time.time()
        
        for img_path in sample_images:
            self.model.predict(str(img_path), verbose=False)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_image = total_time / len(sample_images)
        fps = 1.0 / avg_time_per_image
        
        speed_metrics = {
            "total_images": len(sample_images),
            "total_time_s": total_time,
            "avg_time_per_image_ms": avg_time_per_image * 1000,
            "fps": fps
        }
        
        logger.info(f"Inference speed: {avg_time_per_image*1000:.2f} ms/image, {fps:.2f} FPS")
        
        return speed_metrics
    
    def save_csv_reports(self, val_metrics: Dict, test_metrics: Dict, speed_metrics: Dict):
        """Save evaluation results to CSV files."""
        
        # Overall metrics
        overall_data = [
            {
                "split": "validation",
                "model": "yolo11m",
                "mAP50": val_metrics["mAP50"],
                "mAP50_95": val_metrics["mAP50_95"],
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "f1_score": val_metrics["f1_score"],
                "conf_threshold": val_metrics.get("conf_threshold", 0.25),
                "iou_threshold": val_metrics.get("iou_threshold", 0.45)
            },
            {
                "split": "test",
                "model": "yolo11m",
                "mAP50": test_metrics["mAP50"],
                "mAP50_95": test_metrics["mAP50_95"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1_score": test_metrics["f1_score"],
                "conf_threshold": test_metrics.get("conf_threshold", 0.25),
                "iou_threshold": test_metrics.get("iou_threshold", 0.45)
            }
        ]
        
        overall_df = pd.DataFrame(overall_data)
        overall_csv = self.results_dir / "yolo_overall_metrics.csv"
        overall_df.to_csv(overall_csv, index=False)
        
        # Per-class metrics
        per_class_data = []
        
        for split_name, metrics in [("validation", val_metrics), ("test", test_metrics)]:
            if "per_class" in metrics:
                for class_name, class_metrics in metrics["per_class"].items():
                    row = {
                        "split": split_name,
                        "model": "yolo11m",
                        "class": class_name,
                        "ap50": class_metrics["ap50"],
                        "ap50_95": class_metrics["ap50_95"]
                    }
                    per_class_data.append(row)
        
        if per_class_data:
            per_class_df = pd.DataFrame(per_class_data)
            per_class_csv = self.results_dir / "yolo_per_class_metrics.csv"
            per_class_df.to_csv(per_class_csv, index=False)
        
        # Speed metrics
        speed_df = pd.DataFrame([{
            "model": "yolo11m",
            "avg_inference_time_ms": speed_metrics["avg_time_per_image_ms"],
            "fps": speed_metrics["fps"],
            "total_images_tested": speed_metrics["total_images"]
        }])
        speed_csv = self.results_dir / "yolo_speed_metrics.csv"
        speed_df.to_csv(speed_csv, index=False)
        
        logger.info(f"CSV reports saved to: {self.results_dir}")
    
    def run_full_evaluation(self) -> Dict:
        """Run complete YOLO evaluation pipeline."""
        logger.info("Starting YOLO evaluation pipeline...")
        
        # Evaluate on validation set
        val_metrics = self.evaluate_validation_set()
        
        # Evaluate on test set
        test_metrics = self.evaluate_test_set()
        
        # Benchmark speed
        speed_metrics = self.benchmark_inference_speed()
        
        # Save CSV reports
        self.save_csv_reports(val_metrics, test_metrics, speed_metrics)
        
        # Create summary
        summary = {
            "model": "yolo11m",
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "speed_metrics": speed_metrics,
            "evaluation_completed": True
        }
        
        # Save detailed results
        results_file = self.results_dir / "yolo_evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save summary
        summary_file = self.results_dir / "yolo_evaluation_summary.json"
        summary_data = {
            "model": "yolo11m",
            "validation_mAP50": val_metrics["mAP50"],
            "validation_mAP50_95": val_metrics["mAP50_95"],
            "test_mAP50": test_metrics["mAP50"],
            "test_mAP50_95": test_metrics["mAP50_95"],
            "inference_fps": speed_metrics["fps"],
            "evaluation_completed": True
        }
        
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"YOLO evaluation completed. Results saved to: {results_file}")
        
        return summary


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    evaluator = YOLOEvaluator(cfg)
    summary = evaluator.run_full_evaluation()
    
    # Print summary
    logger.info("=== YOLO Evaluation Summary ===")
    logger.info(f"Validation mAP@0.5: {summary['validation_metrics']['mAP50']:.4f}")
    logger.info(f"Validation mAP@0.5:0.95: {summary['validation_metrics']['mAP50_95']:.4f}")
    logger.info(f"Test mAP@0.5: {summary['test_metrics']['mAP50']:.4f}")
    logger.info(f"Test mAP@0.5:0.95: {summary['test_metrics']['mAP50_95']:.4f}")
    logger.info(f"Inference Speed: {summary['speed_metrics']['fps']:.2f} FPS")


if __name__ == "__main__":
    main()