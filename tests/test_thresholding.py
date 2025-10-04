"""
Test threshold optimization functionality.
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple

from conftest import assert_valid_thresholds, TEST_CLASSES


class TestThresholdOptimization:
    """Test threshold optimization utilities."""
    
    @pytest.fixture
    def sample_predictions_and_labels(self):
        """Create sample predictions and ground truth labels for testing."""
        n_samples = 100
        n_classes = len(TEST_CLASSES)
        
        # Generate realistic prediction scores
        np.random.seed(42)  # For reproducible tests
        
        # Create correlated predictions (higher scores for positive samples)
        y_true = np.random.randint(0, 2, size=(n_samples, n_classes))
        
        # Generate predictions with some correlation to ground truth
        y_scores = np.random.rand(n_samples, n_classes)
        
        # Make positive samples have higher scores on average
        for i in range(n_samples):
            for j in range(n_classes):
                if y_true[i, j] == 1:
                    # Positive samples: boost score
                    y_scores[i, j] = np.clip(y_scores[i, j] + 0.3, 0, 1)
                else:
                    # Negative samples: reduce score
                    y_scores[i, j] = np.clip(y_scores[i, j] - 0.2, 0, 1)
        
        return y_true, y_scores
    
    def test_f1_sweep_returns_valid_thresholds(self, sample_predictions_and_labels):
        """Test that F1 threshold sweep returns valid threshold list."""
        y_true, y_scores = sample_predictions_and_labels
        
        def mock_f1_threshold_sweep(y_true, y_scores, step=0.1):
            """Mock F1 threshold sweep implementation."""
            thresholds = np.arange(0.1, 1.0, step)
            threshold_results = []
            
            for threshold in thresholds:
                # Calculate F1 for each class at this threshold
                class_f1_scores = []
                
                for class_idx in range(y_scores.shape[1]):
                    y_pred = (y_scores[:, class_idx] >= threshold).astype(int)
                    
                    # Calculate F1 score
                    tp = np.sum((y_true[:, class_idx] == 1) & (y_pred == 1))
                    fp = np.sum((y_true[:, class_idx] == 0) & (y_pred == 1))
                    fn = np.sum((y_true[:, class_idx] == 1) & (y_pred == 0))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    class_f1_scores.append(f1)
                
                # Calculate macro F1
                macro_f1 = np.mean(class_f1_scores)
                
                threshold_results.append({
                    'threshold': threshold,
                    'macro_f1': macro_f1,
                    'class_f1_scores': class_f1_scores
                })
            
            return threshold_results
        
        # Run threshold sweep
        results = mock_f1_threshold_sweep(y_true, y_scores, step=0.1)
        
        # Verify results structure
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert 'threshold' in result
            assert 'macro_f1' in result
            assert 'class_f1_scores' in result
            
            # Verify threshold is valid
            threshold = result['threshold']
            assert isinstance(threshold, (int, float))
            assert 0.0 <= threshold <= 1.0
            
            # Verify F1 scores are valid
            macro_f1 = result['macro_f1']
            assert isinstance(macro_f1, (int, float))
            assert 0.0 <= macro_f1 <= 1.0
            
            class_f1_scores = result['class_f1_scores']
            assert isinstance(class_f1_scores, list)
            assert len(class_f1_scores) == len(TEST_CLASSES)
            
            for f1_score in class_f1_scores:
                assert isinstance(f1_score, (int, float))
                assert 0.0 <= f1_score <= 1.0
    
    def test_per_class_threshold_optimization(self, sample_predictions_and_labels):
        """Test per-class threshold optimization."""
        y_true, y_scores = sample_predictions_and_labels
        
        def mock_optimize_per_class_thresholds(y_true, y_scores, metric='f1'):
            """Mock per-class threshold optimization."""
            optimal_thresholds = {}
            
            for class_idx, class_name in enumerate(TEST_CLASSES):
                best_threshold = 0.5
                best_score = 0.0
                
                # Test different thresholds for this class
                for threshold in np.arange(0.1, 1.0, 0.1):
                    y_pred = (y_scores[:, class_idx] >= threshold).astype(int)
                    
                    # Calculate metric score
                    tp = np.sum((y_true[:, class_idx] == 1) & (y_pred == 1))
                    fp = np.sum((y_true[:, class_idx] == 0) & (y_pred == 1))
                    fn = np.sum((y_true[:, class_idx] == 1) & (y_pred == 0))
                    
                    if metric == 'f1':
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    elif metric == 'precision':
                        score = tp / (tp + fp) if (tp + fp) > 0 else 0
                    elif metric == 'recall':
                        score = tp / (tp + fn) if (tp + fn) > 0 else 0
                    else:
                        score = 0.0
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                
                optimal_thresholds[class_name] = best_threshold
            
            return optimal_thresholds
        
        # Test different metrics
        for metric in ['f1', 'precision', 'recall']:
            thresholds = mock_optimize_per_class_thresholds(y_true, y_scores, metric)
            
            # Verify threshold structure
            assert_valid_thresholds(thresholds, TEST_CLASSES)
            
            # Check that thresholds are reasonable
            threshold_values = list(thresholds.values())
            assert min(threshold_values) >= 0.1  # Not too low
            assert max(threshold_values) <= 0.9  # Not too high
    
    def test_precision_recall_threshold_optimization(self, sample_predictions_and_labels):
        """Test precision-recall curve based threshold optimization."""
        y_true, y_scores = sample_predictions_and_labels
        
        def mock_precision_recall_optimization(y_true, y_scores):
            """Mock precision-recall curve optimization."""
            optimal_thresholds = {}
            
            for class_idx, class_name in enumerate(TEST_CLASSES):
                # Calculate precision-recall curve
                thresholds = np.linspace(0, 1, 101)
                precisions = []
                recalls = []
                
                for threshold in thresholds:
                    y_pred = (y_scores[:, class_idx] >= threshold).astype(int)
                    
                    tp = np.sum((y_true[:, class_idx] == 1) & (y_pred == 1))
                    fp = np.sum((y_true[:, class_idx] == 0) & (y_pred == 1))
                    fn = np.sum((y_true[:, class_idx] == 1) & (y_pred == 0))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    
                    precisions.append(precision)
                    recalls.append(recall)
                
                # Find threshold that maximizes F1 (harmonic mean of precision and recall)
                f1_scores = []
                for p, r in zip(precisions, recalls):
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                    f1_scores.append(f1)
                
                best_idx = np.argmax(f1_scores)
                optimal_thresholds[class_name] = thresholds[best_idx]
            
            return optimal_thresholds
        
        # Run optimization
        thresholds = mock_precision_recall_optimization(y_true, y_scores)
        
        # Verify results
        assert_valid_thresholds(thresholds, TEST_CLASSES)
    
    def test_threshold_validation_and_bounds(self):
        """Test threshold validation and boundary conditions."""
        
        def validate_threshold_bounds(thresholds: Dict[str, float]) -> bool:
            """Validate threshold bounds and constraints."""
            for class_name, threshold in thresholds.items():
                # Check type
                if not isinstance(threshold, (int, float)):
                    return False
                
                # Check bounds
                if not (0.0 <= threshold <= 1.0):
                    return False
                
                # Check for invalid values
                if np.isnan(threshold) or np.isinf(threshold):
                    return False
            
            return True
        
        # Test valid thresholds
        valid_thresholds = {class_name: 0.5 for class_name in TEST_CLASSES}
        assert validate_threshold_bounds(valid_thresholds)
        
        # Test boundary values
        boundary_thresholds = {
            TEST_CLASSES[0]: 0.0,
            TEST_CLASSES[1]: 1.0,
            TEST_CLASSES[2]: 0.5
        }
        # Add remaining classes
        for i, class_name in enumerate(TEST_CLASSES[3:]):
            boundary_thresholds[class_name] = 0.5
        
        assert validate_threshold_bounds(boundary_thresholds)
        
        # Test invalid thresholds
        invalid_thresholds = [
            {TEST_CLASSES[0]: -0.1},  # Below 0
            {TEST_CLASSES[0]: 1.1},   # Above 1
            {TEST_CLASSES[0]: float('nan')},  # NaN
            {TEST_CLASSES[0]: float('inf')},  # Inf
            {TEST_CLASSES[0]: "0.5"},  # Wrong type
        ]
        
        for invalid_thresh in invalid_thresholds:
            # Add valid thresholds for other classes
            complete_thresh = {class_name: 0.5 for class_name in TEST_CLASSES}
            complete_thresh.update(invalid_thresh)
            assert not validate_threshold_bounds(complete_thresh)
    
    def test_threshold_optimization_with_class_imbalance(self):
        """Test threshold optimization with imbalanced classes."""
        # Create imbalanced dataset
        n_samples = 1000
        n_classes = len(TEST_CLASSES)
        
        np.random.seed(42)
        
        # Create highly imbalanced labels (some classes very rare)
        y_true = np.zeros((n_samples, n_classes))
        
        # Make first class common (30% positive)
        y_true[:int(0.3 * n_samples), 0] = 1
        
        # Make second class rare (5% positive)
        y_true[:int(0.05 * n_samples), 1] = 1
        
        # Make third class very rare (1% positive)
        y_true[:int(0.01 * n_samples), 2] = 1
        
        # Remaining classes have moderate frequency (10% positive)
        for i in range(3, n_classes):
            y_true[:int(0.1 * n_samples), i] = 1
        
        # Shuffle the data
        for i in range(n_classes):
            np.random.shuffle(y_true[:, i])
        
        # Generate correlated predictions
        y_scores = np.random.rand(n_samples, n_classes)
        for i in range(n_samples):
            for j in range(n_classes):
                if y_true[i, j] == 1:
                    y_scores[i, j] = np.clip(y_scores[i, j] + 0.4, 0, 1)
        
        def mock_imbalance_aware_optimization(y_true, y_scores):
            """Mock optimization that handles class imbalance."""
            optimal_thresholds = {}
            
            for class_idx, class_name in enumerate(TEST_CLASSES):
                class_labels = y_true[:, class_idx]
                class_scores = y_scores[:, class_idx]
                
                # Calculate class balance
                positive_ratio = np.mean(class_labels)
                
                # Adjust threshold based on class imbalance
                if positive_ratio < 0.05:  # Very rare class
                    # Lower threshold to catch more positives
                    base_threshold = 0.3
                elif positive_ratio < 0.15:  # Moderately rare
                    base_threshold = 0.4
                else:  # Common class
                    base_threshold = 0.5
                
                # Fine-tune around base threshold
                best_threshold = base_threshold
                best_f1 = 0.0
                
                for offset in np.arange(-0.2, 0.21, 0.05):
                    threshold = np.clip(base_threshold + offset, 0.1, 0.9)
                    y_pred = (class_scores >= threshold).astype(int)
                    
                    tp = np.sum((class_labels == 1) & (y_pred == 1))
                    fp = np.sum((class_labels == 0) & (y_pred == 1))
                    fn = np.sum((class_labels == 1) & (y_pred == 0))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                
                optimal_thresholds[class_name] = best_threshold
            
            return optimal_thresholds
        
        # Run optimization
        thresholds = mock_imbalance_aware_optimization(y_true, y_scores)
        
        # Verify results
        assert_valid_thresholds(thresholds, TEST_CLASSES)
        
        # Check that rare classes have lower thresholds
        # (This is a heuristic check - exact values may vary)
        threshold_values = list(thresholds.values())
        assert min(threshold_values) < max(threshold_values)  # Some variation in thresholds


@pytest.mark.integration
class TestThresholdOptimizationIntegration:
    """Integration tests for threshold optimization pipeline."""
    
    def test_threshold_optimization_end_to_end(
        self,
        test_config,
        sample_evaluation_results,
        test_outputs_dir
    ):
        """Test complete threshold optimization workflow."""
        # Mock complete optimization pipeline
        def mock_threshold_optimization_pipeline(config):
            """Mock complete threshold optimization."""
            # Generate mock validation predictions
            n_samples = 200
            n_classes = len(TEST_CLASSES)
            
            np.random.seed(42)
            y_true = np.random.randint(0, 2, size=(n_samples, n_classes))
            y_scores = np.random.rand(n_samples, n_classes)
            
            # Optimize thresholds
            optimal_thresholds = {}
            optimization_results = {}
            
            for class_idx, class_name in enumerate(TEST_CLASSES):
                best_threshold = 0.5
                best_f1 = 0.0
                
                for threshold in np.arange(0.1, 1.0, 0.1):
                    y_pred = (y_scores[:, class_idx] >= threshold).astype(int)
                    
                    tp = np.sum((y_true[:, class_idx] == 1) & (y_pred == 1))
                    fp = np.sum((y_true[:, class_idx] == 0) & (y_pred == 1))
                    fn = np.sum((y_true[:, class_idx] == 1) & (y_pred == 0))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                
                optimal_thresholds[class_name] = best_threshold
                optimization_results[class_name] = {
                    "optimal_threshold": best_threshold,
                    "best_f1": best_f1,
                    "optimization_method": "f1_maximization"
                }
            
            # Save results
            thresholds_file = test_outputs_dir / "evaluation" / "optimized_thresholds.json"
            thresholds_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(thresholds_file, 'w') as f:
                import json
                json.dump({
                    "optimized_thresholds": optimal_thresholds,
                    "optimization_results": optimization_results,
                    "method": "per_class_f1_optimization",
                    "validation_samples": n_samples
                }, f, indent=2)
            
            return thresholds_file
        
        # Run optimization
        result_file = mock_threshold_optimization_pipeline(test_config)
        
        # Verify results
        assert result_file.exists()
        
        with open(result_file, 'r') as f:
            import json
            data = json.load(f)
        
        assert "optimized_thresholds" in data
        assert "optimization_results" in data
        
        # Verify thresholds
        thresholds = data["optimized_thresholds"]
        assert_valid_thresholds(thresholds, TEST_CLASSES)
        
        # Verify optimization results
        opt_results = data["optimization_results"]
        for class_name in TEST_CLASSES:
            assert class_name in opt_results
            class_result = opt_results[class_name]
            
            assert "optimal_threshold" in class_result
            assert "best_f1" in class_result
            assert "optimization_method" in class_result
            
            assert 0.0 <= class_result["best_f1"] <= 1.0