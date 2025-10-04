"""
ResNet model evaluation script.
Loads best ResNet model, runs validation, and generates classification reports.
"""

import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, precision_score, 
    recall_score, accuracy_score, hamming_loss, multilabel_confusion_matrix
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset_resnet import CollisionPartsDataset
from ..models.resnet_endtoend import DamageNet
from ..utils.thresholding import (
    load_thresholds, apply_thresholds, evaluate_with_thresholds,
    threshold_optimization_report
)


class ResNetEvaluator:
    """Evaluator for trained ResNet models."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize evaluator with configuration."""
        self.cfg = cfg
        self.device = self._get_device()
        
        # Setup paths
        self.model_path = Path(cfg.paths.output_dir) / "best_model.pth"
        self.thresholds_path = Path(cfg.paths.output_dir) / "thresholds.json"
        self.results_dir = Path(cfg.paths.output_dir) / "evaluation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        
        # Load thresholds
        self.thresholds = self._load_thresholds()
        
        # Create data loaders
        self.val_loader, self.test_loader = self._create_data_loaders()
        
        logger.info(f"ResNet evaluator initialized with device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """Get evaluation device."""
        if self.cfg.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.cfg.device)
        return device
    
    def _load_model(self) -> DamageNet:
        """Load trained ResNet model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model with same configuration
        model = DamageNet(
            num_classes=self.cfg.data.num_classes,
            pretrained=False,  # Don't need pretrained weights when loading checkpoint
            hidden_dims=[512, 256],
            dropout=0.5
        )
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded model from: {self.model_path}")
        return model
    
    def _load_thresholds(self) -> Dict[str, float]:
        """Load optimal thresholds."""
        if self.thresholds_path.exists():
            try:
                thresholds = load_thresholds(self.thresholds_path)
                logger.info(f"Loaded thresholds from: {self.thresholds_path}")
                return thresholds
            except Exception as e:
                logger.warning(f"Failed to load thresholds: {e}")
        
        # Use default thresholds
        default_thresholds = {class_name: 0.5 for class_name in self.cfg.data.classes}
        logger.info("Using default thresholds (0.5 for all classes)")
        return default_thresholds
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create validation and test data loaders."""
        # Validation loader
        val_dataset = CollisionPartsDataset(
            annotations_file=Path(self.cfg.data.processed_data.val_annotations),
            image_base_dir=Path(self.cfg.data.raw_data.vehide_images_dir),
            class_names=self.cfg.data.classes,
            image_size=self.cfg.data.image_size,
            split="val"
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        
        # Test loader
        test_dataset = CollisionPartsDataset(
            annotations_file=Path(self.cfg.data.processed_data.test_annotations),
            image_base_dir=Path(self.cfg.data.raw_data.vehide_images_dir),
            class_names=self.cfg.data.classes,
            image_size=self.cfg.data.image_size,
            split="test"
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"Created data loaders: {len(val_dataset)} val, {len(test_dataset)} test samples")
        return val_loader, test_loader
    
    def _predict_dataset(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Get predictions for entire dataset."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_metadata = []
        
        with torch.no_grad():
            for images, targets, metadata in tqdm(dataloader, desc="Predicting"):
                images = images.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                predictions = torch.sigmoid(outputs)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())
                all_metadata.extend(metadata)
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        return predictions, targets, all_metadata
    
    def evaluate_dataset(self, dataloader: DataLoader, split_name: str) -> Dict:
        """Evaluate model on dataset."""
        logger.info(f"Evaluating on {split_name} set...")
        
        # Get predictions
        predictions, targets, metadata = self._predict_dataset(dataloader)
        
        # Apply thresholds
        binary_predictions = apply_thresholds(predictions, self.thresholds, self.cfg.data.classes)
        
        # Calculate metrics
        metrics = evaluate_with_thresholds(predictions, targets, self.thresholds, self.cfg.data.classes)
        
        # Per-class metrics
        per_class_metrics = self._calculate_per_class_metrics(targets, binary_predictions)
        
        # Classification report
        report = classification_report(
            targets, binary_predictions,
            target_names=self.cfg.data.classes,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrices
        confusion_matrices = self._calculate_confusion_matrices(targets, binary_predictions)
        
        # Combine results
        results = {
            "split": split_name,
            "overall_metrics": metrics,
            "per_class_metrics": per_class_metrics,
            "classification_report": report,
            "confusion_matrices": confusion_matrices,
            "predictions": predictions.tolist(),
            "targets": targets.tolist(),
            "thresholds": self.thresholds
        }
        
        # Save detailed results
        self._save_detailed_results(results, split_name)
        
        # Save CSV report
        self._save_csv_report(results, split_name)
        
        return results
    
    def _calculate_per_class_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate per-class metrics."""
        per_class_metrics = {}
        
        for i, class_name in enumerate(self.cfg.data.classes):
            class_targets = targets[:, i]
            class_predictions = predictions[:, i]
            
            # Skip if no positive samples
            if class_targets.sum() == 0:
                per_class_metrics[class_name] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "support": 0,
                    "true_positives": 0,
                    "false_positives": int(class_predictions.sum()),
                    "false_negatives": 0,
                    "true_negatives": int((1 - class_predictions).sum())
                }
                continue
            
            # Calculate metrics
            tp = int(((class_targets == 1) & (class_predictions == 1)).sum())
            fp = int(((class_targets == 0) & (class_predictions == 1)).sum())
            fn = int(((class_targets == 1) & (class_predictions == 0)).sum())
            tn = int(((class_targets == 0) & (class_predictions == 0)).sum())
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_metrics[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": int(class_targets.sum()),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn
            }
        
        return per_class_metrics
    
    def _calculate_confusion_matrices(self, targets: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate confusion matrices."""
        # Multi-label confusion matrices
        cm_multilabel = multilabel_confusion_matrix(targets, predictions)
        
        confusion_matrices = {}
        for i, class_name in enumerate(self.cfg.data.classes):
            cm = cm_multilabel[i]
            confusion_matrices[class_name] = {
                "matrix": cm.tolist(),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1])
            }
        
        return confusion_matrices
    
    def _save_detailed_results(self, results: Dict, split_name: str):
        """Save detailed results to JSON."""
        results_file = self.results_dir / f"resnet_evaluation_{split_name}.json"
        
        # Create a serializable copy
        serializable_results = {
            "split": results["split"],
            "overall_metrics": results["overall_metrics"],
            "per_class_metrics": results["per_class_metrics"],
            "classification_report": results["classification_report"],
            "confusion_matrices": results["confusion_matrices"],
            "thresholds": results["thresholds"]
        }
        
        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Detailed results saved to: {results_file}")
    
    def _save_csv_report(self, results: Dict, split_name: str):
        """Save classification report to CSV."""
        # Overall metrics
        overall_df = pd.DataFrame([results["overall_metrics"]])
        overall_df.insert(0, "split", split_name)
        overall_df.insert(1, "model", "resnet50")
        
        overall_csv = self.results_dir / f"resnet_overall_metrics_{split_name}.csv"
        overall_df.to_csv(overall_csv, index=False)
        
        # Per-class metrics
        per_class_data = []
        for class_name, metrics in results["per_class_metrics"].items():
            row = {
                "split": split_name,
                "model": "resnet50",
                "class": class_name,
                "threshold": results["thresholds"][class_name],
                **metrics
            }
            per_class_data.append(row)
        
        per_class_df = pd.DataFrame(per_class_data)
        per_class_csv = self.results_dir / f"resnet_per_class_metrics_{split_name}.csv"
        per_class_df.to_csv(per_class_csv, index=False)
        
        # Classification report
        report_data = []
        for class_name, metrics in results["classification_report"].items():
            if class_name not in ["micro avg", "macro avg", "weighted avg", "samples avg"]:
                if isinstance(metrics, dict):
                    row = {
                        "split": split_name,
                        "model": "resnet50",
                        "class": class_name,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1-score"],
                        "support": metrics["support"]
                    }
                    report_data.append(row)
        
        # Add aggregate metrics
        for avg_type in ["micro avg", "macro avg", "weighted avg"]:
            if avg_type in results["classification_report"]:
                metrics = results["classification_report"][avg_type]
                row = {
                    "split": split_name,
                    "model": "resnet50",
                    "class": avg_type,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1-score"],
                    "support": metrics["support"]
                }
                report_data.append(row)
        
        report_df = pd.DataFrame(report_data)
        report_csv = self.results_dir / f"resnet_classification_report_{split_name}.csv"
        report_df.to_csv(report_csv, index=False)
        
        logger.info(f"CSV reports saved to: {self.results_dir}")
    
    def optimize_thresholds(self):
        """Optimize thresholds on validation set."""
        logger.info("Optimizing thresholds on validation set...")
        
        # Get validation predictions
        val_predictions, val_targets, _ = self._predict_dataset(self.val_loader)
        
        # Run threshold optimization
        optimization_results = threshold_optimization_report(
            val_predictions,
            val_targets,
            self.cfg.data.classes,
            self.results_dir / "threshold_optimization"
        )
        
        # Use the best method (highest F1 macro)
        best_method = max(
            optimization_results.keys(),
            key=lambda k: optimization_results[k]["metrics"]["f1_macro"]
        )
        
        logger.info(f"Best threshold optimization method: {best_method}")
        
        # Update thresholds
        self.thresholds = optimization_results[best_method]["thresholds"]
        
        # Save optimized thresholds
        optimized_thresholds_path = self.results_dir / "optimized_thresholds.json"
        with open(optimized_thresholds_path, "w") as f:
            json.dump({
                "thresholds": self.thresholds,
                "method": best_method,
                "optimization_results": optimization_results
            }, f, indent=2)
        
        logger.info(f"Optimized thresholds saved to: {optimized_thresholds_path}")
        
        return optimization_results
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        logger.info("Starting ResNet evaluation pipeline...")
        
        # Optimize thresholds on validation set
        threshold_results = self.optimize_thresholds()
        
        # Evaluate on validation set with optimized thresholds
        val_results = self.evaluate_dataset(self.val_loader, "validation")
        
        # Evaluate on test set with optimized thresholds
        test_results = self.evaluate_dataset(self.test_loader, "test")
        
        # Create summary
        summary = {
            "model": "resnet50",
            "threshold_optimization": threshold_results,
            "validation_results": val_results["overall_metrics"],
            "test_results": test_results["overall_metrics"],
            "evaluation_completed": True
        }
        
        # Save summary
        summary_file = self.results_dir / "resnet_evaluation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Evaluation completed. Summary saved to: {summary_file}")
        
        return summary


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    evaluator = ResNetEvaluator(cfg)
    summary = evaluator.run_full_evaluation()
    
    # Print summary
    logger.info("=== ResNet Evaluation Summary ===")
    logger.info(f"Validation F1 Macro: {summary['validation_results']['f1_macro']:.4f}")
    logger.info(f"Test F1 Macro: {summary['test_results']['f1_macro']:.4f}")
    logger.info(f"Test Exact Match: {summary['test_results']['exact_match_ratio']:.4f}")


if __name__ == "__main__":
    main()