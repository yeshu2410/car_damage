"""
Threshold optimization utilities for multi-label classification.
Sweeps thresholds per class to maximize F1 score and saves optimal values.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from sklearn.metrics import f1_score, precision_recall_curve
from loguru import logger
from tqdm import tqdm


def sweep_thresholds_per_class(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: List[str],
    threshold_range: Tuple[float, float, float] = (0.05, 0.95, 0.01),
    metric: str = "f1_score",
    average: str = "binary"
) -> Dict[str, float]:
    """
    Sweep thresholds per class to find optimal values.
    
    Args:
        predictions: Probability predictions [num_samples, num_classes]
        targets: Binary targets [num_samples, num_classes]
        class_names: List of class names
        threshold_range: (start, end, step) for threshold sweep
        metric: Metric to optimize ("f1_score", "precision", "recall")
        average: Averaging method for metric calculation
        
    Returns:
        Dictionary mapping class names to optimal thresholds
    """
    num_classes = predictions.shape[1]
    optimal_thresholds = {}
    
    # Generate threshold range
    start, end, step = threshold_range
    thresholds = np.arange(start, end + step, step)
    
    logger.info(f"Sweeping thresholds from {start} to {end} with step {step}")
    logger.info(f"Optimizing {metric} with {average} averaging")
    
    for class_idx in tqdm(range(num_classes), desc="Optimizing thresholds"):
        class_name = class_names[class_idx]
        class_predictions = predictions[:, class_idx]
        class_targets = targets[:, class_idx]
        
        # Skip if no positive samples
        if class_targets.sum() == 0:
            logger.warning(f"No positive samples for class {class_name}, using default threshold 0.5")
            optimal_thresholds[class_name] = 0.5
            continue
        
        best_score = -1.0
        best_threshold = 0.5
        
        # Sweep thresholds
        for threshold in thresholds:
            binary_predictions = (class_predictions >= threshold).astype(int)
            
            # Calculate metric
            if metric == "f1_score":
                score = f1_score(class_targets, binary_predictions, average=average, zero_division=0)
            elif metric == "precision":
                from sklearn.metrics import precision_score
                score = precision_score(class_targets, binary_predictions, average=average, zero_division=0)
            elif metric == "recall":
                from sklearn.metrics import recall_score
                score = recall_score(class_targets, binary_predictions, average=average, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        optimal_thresholds[class_name] = best_threshold
        logger.debug(f"Class {class_name}: optimal threshold = {best_threshold:.3f}, "
                    f"{metric} = {best_score:.4f}")
    
    return optimal_thresholds


def sweep_thresholds_global(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold_range: Tuple[float, float, float] = (0.05, 0.95, 0.01),
    metric: str = "f1_score",
    average: str = "macro"
) -> float:
    """
    Sweep single global threshold for all classes.
    
    Args:
        predictions: Probability predictions [num_samples, num_classes]
        targets: Binary targets [num_samples, num_classes]
        threshold_range: (start, end, step) for threshold sweep
        metric: Metric to optimize
        average: Averaging method for metric calculation
        
    Returns:
        Optimal global threshold
    """
    start, end, step = threshold_range
    thresholds = np.arange(start, end + step, step)
    
    best_score = -1.0
    best_threshold = 0.5
    
    logger.info(f"Sweeping global threshold from {start} to {end} with step {step}")
    
    for threshold in tqdm(thresholds, desc="Global threshold sweep"):
        binary_predictions = (predictions >= threshold).astype(int)
        
        # Calculate metric
        if metric == "f1_score":
            score = f1_score(targets, binary_predictions, average=average, zero_division=0)
        elif metric == "precision":
            from sklearn.metrics import precision_score
            score = precision_score(targets, binary_predictions, average=average, zero_division=0)
        elif metric == "recall":
            from sklearn.metrics import recall_score
            score = recall_score(targets, binary_predictions, average=average, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    logger.info(f"Optimal global threshold: {best_threshold:.3f}, {metric}: {best_score:.4f}")
    return best_threshold


def precision_recall_threshold_optimization(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Find optimal thresholds using precision-recall curves.
    
    Args:
        predictions: Probability predictions [num_samples, num_classes]
        targets: Binary targets [num_samples, num_classes]
        class_names: List of class names
        
    Returns:
        Dictionary with optimal thresholds and metrics for each class
    """
    num_classes = predictions.shape[1]
    threshold_results = {}
    
    for class_idx in range(num_classes):
        class_name = class_names[class_idx]
        class_predictions = predictions[:, class_idx]
        class_targets = targets[:, class_idx]
        
        # Skip if no positive samples
        if class_targets.sum() == 0:
            threshold_results[class_name] = {
                "optimal_threshold": 0.5,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
            continue
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(class_targets, class_predictions)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Find optimal threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        threshold_results[class_name] = {
            "optimal_threshold": float(optimal_threshold),
            "precision": float(precision[optimal_idx]),
            "recall": float(recall[optimal_idx]),
            "f1_score": float(f1_scores[optimal_idx])
        }
    
    return threshold_results


def save_thresholds(
    thresholds: Dict[str, float],
    output_path: Union[str, Path],
    metadata: Dict = None
) -> None:
    """
    Save optimal thresholds to JSON file.
    
    Args:
        thresholds: Dictionary mapping class names to thresholds
        output_path: Path to save the thresholds JSON file
        metadata: Optional metadata to include
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data to save
    threshold_data = {
        "thresholds": thresholds,
        "metadata": metadata or {},
        "num_classes": len(thresholds),
        "class_names": list(thresholds.keys())
    }
    
    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(threshold_data, f, indent=2)
    
    logger.info(f"Thresholds saved to: {output_path}")


def load_thresholds(input_path: Union[str, Path]) -> Dict[str, float]:
    """
    Load thresholds from JSON file.
    
    Args:
        input_path: Path to the thresholds JSON file
        
    Returns:
        Dictionary mapping class names to thresholds
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {input_path}")
    
    with open(input_path, "r") as f:
        threshold_data = json.load(f)
    
    return threshold_data["thresholds"]


def apply_thresholds(
    predictions: np.ndarray,
    thresholds: Union[Dict[str, float], List[float], float],
    class_names: List[str] = None
) -> np.ndarray:
    """
    Apply thresholds to predictions to get binary classifications.
    
    Args:
        predictions: Probability predictions [num_samples, num_classes]
        thresholds: Thresholds per class (dict, list, or single value)
        class_names: List of class names (required if thresholds is dict)
        
    Returns:
        Binary predictions [num_samples, num_classes]
    """
    if isinstance(thresholds, dict):
        if class_names is None:
            raise ValueError("class_names required when thresholds is a dictionary")
        
        # Convert dict to array
        threshold_array = np.array([thresholds[name] for name in class_names])
    elif isinstance(thresholds, list):
        threshold_array = np.array(thresholds)
    else:
        # Single threshold for all classes
        threshold_array = np.full(predictions.shape[1], thresholds)
    
    # Apply thresholds
    binary_predictions = (predictions >= threshold_array).astype(int)
    
    return binary_predictions


def evaluate_with_thresholds(
    predictions: np.ndarray,
    targets: np.ndarray,
    thresholds: Union[Dict[str, float], List[float], float],
    class_names: List[str] = None
) -> Dict[str, float]:
    """
    Evaluate predictions using specified thresholds.
    
    Args:
        predictions: Probability predictions [num_samples, num_classes]
        targets: Binary targets [num_samples, num_classes]
        thresholds: Thresholds per class
        class_names: List of class names
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Apply thresholds
    binary_predictions = apply_thresholds(predictions, thresholds, class_names)
    
    # Calculate metrics
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, accuracy_score, hamming_loss
    )
    
    metrics = {
        "f1_micro": f1_score(targets, binary_predictions, average="micro"),
        "f1_macro": f1_score(targets, binary_predictions, average="macro"),
        "f1_weighted": f1_score(targets, binary_predictions, average="weighted"),
        "precision_micro": precision_score(targets, binary_predictions, average="micro"),
        "precision_macro": precision_score(targets, binary_predictions, average="macro"),
        "recall_micro": recall_score(targets, binary_predictions, average="micro"),
        "recall_macro": recall_score(targets, binary_predictions, average="macro"),
        "accuracy": accuracy_score(targets, binary_predictions),
        "hamming_loss": hamming_loss(targets, binary_predictions),
        "exact_match_ratio": accuracy_score(targets, binary_predictions)
    }
    
    return metrics


def threshold_optimization_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: List[str],
    output_dir: Union[str, Path],
    methods: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Generate comprehensive threshold optimization report.
    
    Args:
        predictions: Probability predictions [num_samples, num_classes]
        targets: Binary targets [num_samples, num_classes]
        class_names: List of class names
        output_dir: Directory to save results
        methods: List of optimization methods to try
        
    Returns:
        Dictionary of results for each method
    """
    if methods is None:
        methods = ["per_class_f1", "global_f1", "precision_recall"]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for method in methods:
        logger.info(f"Running threshold optimization with method: {method}")
        
        if method == "per_class_f1":
            thresholds = sweep_thresholds_per_class(
                predictions, targets, class_names, metric="f1_score"
            )
            
        elif method == "global_f1":
            global_threshold = sweep_thresholds_global(
                predictions, targets, metric="f1_score"
            )
            thresholds = {name: global_threshold for name in class_names}
            
        elif method == "precision_recall":
            pr_results = precision_recall_threshold_optimization(
                predictions, targets, class_names
            )
            thresholds = {name: result["optimal_threshold"] 
                         for name, result in pr_results.items()}
        
        # Evaluate with optimized thresholds
        metrics = evaluate_with_thresholds(predictions, targets, thresholds, class_names)
        
        # Save thresholds
        threshold_file = output_dir / f"thresholds_{method}.json"
        save_thresholds(
            thresholds, 
            threshold_file,
            metadata={
                "method": method,
                "metrics": metrics,
                "optimization_range": "0.05-0.95",
                "step_size": 0.01
            }
        )
        
        results[method] = {
            "thresholds": thresholds,
            "metrics": metrics,
            "threshold_file": str(threshold_file)
        }
        
        logger.info(f"Method {method}: F1 Macro = {metrics['f1_macro']:.4f}, "
                   f"F1 Micro = {metrics['f1_micro']:.4f}")
    
    # Save comparison report
    comparison_file = output_dir / "threshold_optimization_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Threshold optimization report saved to: {comparison_file}")
    return results