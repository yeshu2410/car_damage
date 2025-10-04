"""
Model comparison and visualization utilities.
Loads evaluation results and generates comparison plots and reports.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

import hydra
from loguru import logger
from omegaconf import DictConfig


class ModelComparator:
    """Compare evaluation results from different models."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize model comparator."""
        self.cfg = cfg
        self.results_dir = Path(cfg.paths.output_dir) / "evaluation"
        self.comparison_dir = self.results_dir / "comparisons"
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Load evaluation results
        self.resnet_results = self._load_resnet_results()
        self.yolo_results = self._load_yolo_results()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info("Model comparator initialized")
    
    def _load_resnet_results(self) -> Dict:
        """Load ResNet evaluation results."""
        results = {}
        
        # Load validation results
        val_file = self.results_dir / "resnet_evaluation_validation.json"
        if val_file.exists():
            with open(val_file, "r") as f:
                results["validation"] = json.load(f)
        
        # Load test results
        test_file = self.results_dir / "resnet_evaluation_test.json"
        if test_file.exists():
            with open(test_file, "r") as f:
                results["test"] = json.load(f)
        
        # Load summary
        summary_file = self.results_dir / "resnet_evaluation_summary.json"
        if summary_file.exists():
            with open(summary_file, "r") as f:
                results["summary"] = json.load(f)
        
        logger.info(f"Loaded ResNet results: {list(results.keys())}")
        return results
    
    def _load_yolo_results(self) -> Dict:
        """Load YOLO evaluation results."""
        results = {}
        
        # Load evaluation results
        results_file = self.results_dir / "yolo_evaluation_results.json"
        if results_file.exists():
            with open(results_file, "r") as f:
                results = json.load(f)
        
        logger.info("Loaded YOLO results")
        return results
    
    def create_overall_comparison(self) -> pd.DataFrame:
        """Create overall model comparison DataFrame."""
        comparison_data = []
        
        # ResNet metrics
        if "test" in self.resnet_results:
            resnet_metrics = self.resnet_results["test"]["overall_metrics"]
            comparison_data.append({
                "model": "ResNet50",
                "type": "Classification",
                "f1_micro": resnet_metrics.get("f1_micro", 0.0),
                "f1_macro": resnet_metrics.get("f1_macro", 0.0),
                "precision_micro": resnet_metrics.get("precision_micro", 0.0),
                "recall_micro": resnet_metrics.get("recall_micro", 0.0),
                "hamming_loss": resnet_metrics.get("hamming_loss", 0.0),
                "exact_match": resnet_metrics.get("exact_match_ratio", 0.0),
                "primary_metric": resnet_metrics.get("f1_macro", 0.0)
            })
        
        # YOLO metrics
        if "test_metrics" in self.yolo_results:
            yolo_metrics = self.yolo_results["test_metrics"]
            comparison_data.append({
                "model": "YOLO11m",
                "type": "Detection",
                "mAP50": yolo_metrics.get("mAP50", 0.0),
                "mAP50_95": yolo_metrics.get("mAP50_95", 0.0),
                "precision": yolo_metrics.get("precision", 0.0),
                "recall": yolo_metrics.get("recall", 0.0),
                "f1_score": yolo_metrics.get("f1_score", 0.0),
                "primary_metric": yolo_metrics.get("mAP50", 0.0)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_csv = self.comparison_dir / "overall_results_final.csv"
        df.to_csv(comparison_csv, index=False)
        
        logger.info(f"Overall comparison saved to: {comparison_csv}")
        return df
    
    def create_per_class_comparison(self) -> pd.DataFrame:
        """Create per-class comparison DataFrame."""
        comparison_data = []
        
        # ResNet per-class metrics
        if "test" in self.resnet_results and "per_class_metrics" in self.resnet_results["test"]:
            resnet_per_class = self.resnet_results["test"]["per_class_metrics"]
            
            for class_name, metrics in resnet_per_class.items():
                comparison_data.append({
                    "model": "ResNet50",
                    "class": class_name,
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1_score": metrics.get("f1_score", 0.0),
                    "support": metrics.get("support", 0)
                })
        
        # YOLO per-class metrics
        if "test_metrics" in self.yolo_results and "per_class" in self.yolo_results["test_metrics"]:
            yolo_per_class = self.yolo_results["test_metrics"]["per_class"]
            
            for class_name, metrics in yolo_per_class.items():
                comparison_data.append({
                    "model": "YOLO11m",
                    "class": class_name,
                    "ap50": metrics.get("ap50", 0.0),
                    "ap50_95": metrics.get("ap50_95", 0.0)
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison
        per_class_csv = self.comparison_dir / "per_class_comparison.csv"
        df.to_csv(per_class_csv, index=False)
        
        logger.info(f"Per-class comparison saved to: {per_class_csv}")
        return df
    
    def plot_model_comparison(self, overall_df: pd.DataFrame):
        """Create comparison plots."""
        
        # Plot 1: Primary Metrics Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model Performance Comparison", fontsize=16, fontweight='bold')
        
        # Primary metrics bar plot
        if len(overall_df) >= 2:
            ax1 = axes[0, 0]
            models = overall_df["model"]
            primary_metrics = overall_df["primary_metric"]
            
            bars = ax1.bar(models, primary_metrics, color=['skyblue', 'lightcoral'])
            ax1.set_title("Primary Metrics\n(F1 Macro for ResNet, mAP@0.5 for YOLO)")
            ax1.set_ylabel("Score")
            ax1.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, primary_metrics):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # ResNet metrics breakdown
        if "test" in self.resnet_results:
            ax2 = axes[0, 1]
            resnet_metrics = self.resnet_results["test"]["overall_metrics"]
            
            metrics = ["f1_micro", "f1_macro", "precision_micro", "recall_micro"]
            values = [resnet_metrics.get(m, 0.0) for m in metrics]
            
            bars = ax2.bar(metrics, values, color='skyblue', alpha=0.7)
            ax2.set_title("ResNet Classification Metrics")
            ax2.set_ylabel("Score")
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # YOLO metrics breakdown
        if "test_metrics" in self.yolo_results:
            ax3 = axes[1, 0]
            yolo_metrics = self.yolo_results["test_metrics"]
            
            metrics = ["mAP50", "mAP50_95", "precision", "recall", "f1_score"]
            values = [yolo_metrics.get(m, 0.0) for m in metrics]
            
            bars = ax3.bar(metrics, values, color='lightcoral', alpha=0.7)
            ax3.set_title("YOLO Detection Metrics")
            ax3.set_ylabel("Score")
            ax3.set_ylim(0, 1)
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Speed comparison (if available)
        ax4 = axes[1, 1]
        speed_data = []
        
        if "speed_metrics" in self.yolo_results:
            speed_data.append({
                "model": "YOLO11m",
                "fps": self.yolo_results["speed_metrics"].get("fps", 0.0)
            })
        
        # Placeholder for ResNet speed (would need to be measured)
        speed_data.append({
            "model": "ResNet50",
            "fps": 25.0  # Estimated typical speed
        })
        
        if speed_data:
            speed_df = pd.DataFrame(speed_data)
            bars = ax4.bar(speed_df["model"], speed_df["fps"], 
                          color=['skyblue', 'lightcoral'], alpha=0.7)
            ax4.set_title("Inference Speed Comparison")
            ax4.set_ylabel("FPS")
            
            for bar, value in zip(bars, speed_df["fps"]):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.comparison_dir / "model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plot saved to: {plot_path}")
    
    def plot_per_class_comparison(self, per_class_df: pd.DataFrame):
        """Create per-class comparison plots."""
        
        if per_class_df.empty:
            logger.warning("No per-class data available for plotting")
            return
        
        # Get unique classes
        classes = per_class_df["class"].unique()
        
        if len(classes) == 0:
            return
        
        # Create subplots for each class
        n_classes = len(classes)
        n_cols = 3
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.suptitle("Per-Class Performance Comparison", fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, class_name in enumerate(classes):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            class_data = per_class_df[per_class_df["class"] == class_name]
            
            # Plot ResNet metrics if available
            resnet_data = class_data[class_data["model"] == "ResNet50"]
            if not resnet_data.empty:
                resnet_row = resnet_data.iloc[0]
                metrics = ["precision", "recall", "f1_score"]
                values = [resnet_row.get(m, 0.0) for m in metrics]
                
                x_pos = np.arange(len(metrics))
                bars1 = ax.bar(x_pos - 0.2, values, 0.4, label="ResNet50", color='skyblue', alpha=0.7)
                
                # Add value labels
                for bar, value in zip(bars1, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Plot YOLO metrics if available
            yolo_data = class_data[class_data["model"] == "YOLO11m"]
            if not yolo_data.empty:
                yolo_row = yolo_data.iloc[0]
                yolo_metrics = ["ap50", "ap50_95"]
                yolo_values = [yolo_row.get(m, 0.0) for m in yolo_metrics]
                
                x_pos_yolo = np.arange(len(yolo_metrics)) + len(metrics) + 0.5
                bars2 = ax.bar(x_pos_yolo, yolo_values, 0.4, label="YOLO11m", color='lightcoral', alpha=0.7)
                
                # Add value labels
                for bar, value in zip(bars2, yolo_values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title(f"Class: {class_name}", fontweight='bold')
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            
            # Set x-axis labels
            all_labels = ["precision", "recall", "f1_score", "ap50", "ap50_95"]
            ax.set_xticks(range(len(all_labels)))
            ax.set_xticklabels(all_labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_classes, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.comparison_dir / "per_class_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Per-class comparison plot saved to: {plot_path}")
    
    def create_performance_summary(self) -> Dict:
        """Create comprehensive performance summary."""
        summary = {
            "evaluation_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models_compared": [],
            "best_classification_model": None,
            "best_detection_model": None,
            "recommendations": []
        }
        
        # ResNet summary
        if "test" in self.resnet_results:
            resnet_summary = {
                "model": "ResNet50",
                "type": "Classification",
                "f1_macro": self.resnet_results["test"]["overall_metrics"].get("f1_macro", 0.0),
                "f1_micro": self.resnet_results["test"]["overall_metrics"].get("f1_micro", 0.0),
                "exact_match": self.resnet_results["test"]["overall_metrics"].get("exact_match_ratio", 0.0),
                "strengths": [],
                "weaknesses": []
            }
            
            # Analyze strengths and weaknesses
            f1_macro = resnet_summary["f1_macro"]
            if f1_macro > 0.8:
                resnet_summary["strengths"].append("High F1 macro score")
            elif f1_macro < 0.6:
                resnet_summary["weaknesses"].append("Low F1 macro score")
            
            summary["models_compared"].append(resnet_summary)
            summary["best_classification_model"] = "ResNet50"
        
        # YOLO summary
        if "test_metrics" in self.yolo_results:
            yolo_summary = {
                "model": "YOLO11m",
                "type": "Detection",
                "mAP50": self.yolo_results["test_metrics"].get("mAP50", 0.0),
                "mAP50_95": self.yolo_results["test_metrics"].get("mAP50_95", 0.0),
                "fps": self.yolo_results.get("speed_metrics", {}).get("fps", 0.0),
                "strengths": [],
                "weaknesses": []
            }
            
            # Analyze strengths and weaknesses
            map50 = yolo_summary["mAP50"]
            fps = yolo_summary["fps"]
            
            if map50 > 0.7:
                yolo_summary["strengths"].append("High mAP@0.5 score")
            elif map50 < 0.5:
                yolo_summary["weaknesses"].append("Low mAP@0.5 score")
            
            if fps > 30:
                yolo_summary["strengths"].append("Real-time inference speed")
            elif fps < 10:
                yolo_summary["weaknesses"].append("Slow inference speed")
            
            summary["models_compared"].append(yolo_summary)
            summary["best_detection_model"] = "YOLO11m"
        
        # Generate recommendations
        if len(summary["models_compared"]) >= 2:
            resnet_f1 = summary["models_compared"][0].get("f1_macro", 0.0)
            yolo_map = summary["models_compared"][1].get("mAP50", 0.0)
            
            if resnet_f1 > 0.75:
                summary["recommendations"].append(
                    "ResNet50 shows good classification performance for damage type identification"
                )
            
            if yolo_map > 0.6:
                summary["recommendations"].append(
                    "YOLO11m is suitable for real-time damage detection and localization"
                )
            
            summary["recommendations"].append(
                "Consider ensemble approach: YOLO for detection + ResNet for detailed classification"
            )
        
        # Save summary
        summary_file = self.comparison_dir / "performance_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Performance summary saved to: {summary_file}")
        return summary
    
    def run_full_comparison(self):
        """Run complete model comparison pipeline."""
        logger.info("Starting model comparison pipeline...")
        
        # Create comparison DataFrames
        overall_df = self.create_overall_comparison()
        per_class_df = self.create_per_class_comparison()
        
        # Create plots
        self.plot_model_comparison(overall_df)
        self.plot_per_class_comparison(per_class_df)
        
        # Create performance summary
        summary = self.create_performance_summary()
        
        logger.info("Model comparison completed successfully!")
        
        # Print key findings
        if overall_df is not None and len(overall_df) > 0:
            logger.info("=== Model Comparison Summary ===")
            for _, row in overall_df.iterrows():
                logger.info(f"{row['model']}: Primary Metric = {row['primary_metric']:.4f}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main comparison function."""
    comparator = ModelComparator(cfg)
    comparator.run_full_comparison()


if __name__ == "__main__":
    main()