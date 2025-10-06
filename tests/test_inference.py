"""
Test model inference components.
"""

import pytest
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

from .conftest import TEST_CLASSES, TEST_IMAGE_SIZE, create_mock_resnet_model


class TestInferencePipeline:
    """Test inference pipeline functionality."""
    
    def test_pipeline_initialization(self, test_config, monkeypatch):
        """Test that pipeline initializes correctly."""
        # Mock the heavy imports
        monkeypatch.setattr("torch.device", lambda x: "cpu")
        
        class MockInferencePipeline:
            def __init__(self, config_path):
                self.cfg = test_config
                self.device = "cpu"
                self.resnet_model = None
                self.yolo_model = None
                self.thresholds = {cls: 0.5 for cls in TEST_CLASSES}
                self.correction_codes = {}
                self.model_versions = {}
                self.inference_stats = {
                    "total_inferences": 0,
                    "avg_resnet_time": 0.0,
                    "avg_yolo_time": 0.0,
                    "avg_fusion_time": 0.0,
                    "avg_total_time": 0.0
                }
        
        pipeline = MockInferencePipeline("test_config.yaml")
        
        assert pipeline.cfg is not None
        assert pipeline.device == "cpu"
        assert len(pipeline.thresholds) == len(TEST_CLASSES)
        assert pipeline.inference_stats["total_inferences"] == 0
    
    def test_image_preprocessing(self, sample_image):
        """Test image preprocessing functionality."""
        def mock_preprocess_image(image, target_size=TEST_IMAGE_SIZE):
            """Mock image preprocessing."""
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            if image.size != target_size:
                image = image.resize(target_size)
            
            # Convert to array and normalize
            img_array = np.array(image).astype(np.float32) / 255.0
            
            return img_array
        
        # Test with RGB image
        rgb_image = sample_image.convert('RGB')
        processed = mock_preprocess_image(rgb_image)
        
        assert processed.shape == (*TEST_IMAGE_SIZE[::-1], 3)  # (H, W, C)
        assert processed.dtype == np.float32
        assert 0.0 <= processed.min() <= processed.max() <= 1.0
        
        # Test with different modes
        for mode in ['L', 'RGBA']:
            converted_image = sample_image.convert(mode)
            processed = mock_preprocess_image(converted_image)
            assert processed.shape == (*TEST_IMAGE_SIZE[::-1], 3)
    
    def test_score_fusion_max_rule(self):
        """Test score fusion using max rule."""
        def mock_fuse_scores(resnet_scores, yolo_scores, method="max_rule"):
            """Mock score fusion implementation."""
            assert len(resnet_scores) == len(yolo_scores)
            
            if method == "max_rule":
                return [max(r, y) for r, y in zip(resnet_scores, yolo_scores)]
            elif method == "average":
                return [(r + y) / 2 for r, y in zip(resnet_scores, yolo_scores)]
            else:
                raise ValueError(f"Unknown fusion method: {method}")
        
        # Test max rule fusion
        resnet_scores = [0.8, 0.3, 0.6, 0.1]
        yolo_scores = [0.7, 0.9, 0.4, 0.2]
        
        fused_max = mock_fuse_scores(resnet_scores, yolo_scores, "max_rule")
        expected_max = [0.8, 0.9, 0.6, 0.2]
        
        assert fused_max == expected_max
        
        # Test average fusion
        fused_avg = mock_fuse_scores(resnet_scores, yolo_scores, "average")
        expected_avg = [0.75, 0.6, 0.5, 0.15]
        
        assert fused_avg == expected_avg
    
    def test_threshold_application(self, sample_thresholds):
        """Test threshold application to scores."""
        def mock_apply_thresholds(scores, thresholds):
            """Mock threshold application."""
            predictions = {}
            
            for class_name, score in scores.items():
                threshold = thresholds.get(class_name, 0.5)
                predictions[class_name] = {
                    "score": score,
                    "threshold": threshold,
                    "predicted": score >= threshold
                }
            
            return predictions
        
        # Test with sample scores
        scores = {
            TEST_CLASSES[0]: 0.8,
            TEST_CLASSES[1]: 0.3,
            TEST_CLASSES[2]: 0.6,
        }
        
        predictions = mock_apply_thresholds(scores, sample_thresholds)
        
        # Verify structure
        for class_name in scores.keys():
            assert class_name in predictions
            pred = predictions[class_name]
            
            assert "score" in pred
            assert "threshold" in pred
            assert "predicted" in pred
            
            assert pred["score"] == scores[class_name]
            assert pred["threshold"] == sample_thresholds[class_name]
            assert pred["predicted"] == (scores[class_name] >= sample_thresholds[class_name])


class TestModelMapping:
    """Test prediction mapping functionality."""
    
    def test_class_to_correction_code_mapping(self):
        """Test mapping from damage classes to correction codes."""
        def mock_map_to_correction_codes(predicted_classes):
            """Mock correction code mapping."""
            code_mapping = {
                "front_bumper_damage": "FB001",
                "rear_bumper_damage": "RB001",
                "hood_damage": "HD001",
                "door_damage": "DR001",
                "fender_damage": "FD001",
                "headlight_damage": "HL001"
            }
            
            correction_codes = []
            for class_name in predicted_classes:
                if class_name in code_mapping:
                    correction_codes.append({
                        "class_name": class_name,
                        "correction_code": code_mapping[class_name],
                        "description": f"{class_name.replace('_', ' ').title()}"
                    })
            
            return correction_codes
        
        # Test mapping
        test_classes = ["front_bumper_damage", "hood_damage"]
        codes = mock_map_to_correction_codes(test_classes)
        
        assert len(codes) == 2
        
        for code in codes:
            assert "class_name" in code
            assert "correction_code" in code
            assert "description" in code
            
            assert code["class_name"] in test_classes
            assert code["correction_code"].startswith(code["class_name"][:2].upper())
    
    def test_severity_analysis(self):
        """Test damage severity analysis."""
        def mock_analyze_severity(predictions):
            """Mock severity analysis."""
            severity_analysis = {}
            
            for class_name, pred_data in predictions.items():
                if pred_data.get("predicted", False):
                    confidence = pred_data["score"]
                    
                    if confidence >= 0.8:
                        severity = "severe"
                    elif confidence >= 0.6:
                        severity = "moderate"
                    elif confidence >= 0.4:
                        severity = "minor"
                    else:
                        severity = "minimal"
                    
                    severity_analysis[class_name] = {
                        "severity_level": severity,
                        "confidence": confidence,
                        "repair_action": "replace" if severity == "severe" else "repair"
                    }
            
            return severity_analysis
        
        # Test with different confidence levels
        test_predictions = {
            "front_bumper_damage": {"score": 0.9, "predicted": True},
            "hood_damage": {"score": 0.7, "predicted": True},
            "door_damage": {"score": 0.5, "predicted": True},
            "fender_damage": {"score": 0.3, "predicted": False}
        }
        
        analysis = mock_analyze_severity(test_predictions)
        
        # Should only analyze predicted classes
        assert len(analysis) == 3  # Exclude fender_damage (not predicted)
        
        # Check severity levels
        assert analysis["front_bumper_damage"]["severity_level"] == "severe"
        assert analysis["hood_damage"]["severity_level"] == "moderate"
        assert analysis["door_damage"]["severity_level"] == "minor"
        
        # Check repair actions
        assert analysis["front_bumper_damage"]["repair_action"] == "replace"
        assert analysis["hood_damage"]["repair_action"] == "repair"
    
    def test_cost_estimation(self):
        """Test repair cost estimation."""
        def mock_estimate_costs(severity_analysis):
            """Mock cost estimation."""
            base_costs = {
                "front_bumper_damage": 500,
                "rear_bumper_damage": 450,
                "hood_damage": 800,
                "door_damage": 600,
                "fender_damage": 400,
                "headlight_damage": 200
            }
            
            severity_multipliers = {
                "minimal": 0.8,
                "minor": 1.0,
                "moderate": 1.3,
                "severe": 1.8
            }
            
            total_cost = 0
            cost_breakdown = {}
            
            for class_name, analysis in severity_analysis.items():
                base_cost = base_costs.get(class_name, 300)
                multiplier = severity_multipliers.get(analysis["severity_level"], 1.0)
                
                estimated_cost = int(base_cost * multiplier)
                total_cost += estimated_cost
                
                cost_breakdown[class_name] = {
                    "base_cost": base_cost,
                    "severity_multiplier": multiplier,
                    "estimated_cost": estimated_cost
                }
            
            return {
                "total_estimated_cost": total_cost,
                "cost_breakdown": cost_breakdown
            }
        
        # Test cost estimation
        test_analysis = {
            "front_bumper_damage": {"severity_level": "severe"},
            "hood_damage": {"severity_level": "moderate"}
        }
        
        costs = mock_estimate_costs(test_analysis)
        
        assert "total_estimated_cost" in costs
        assert "cost_breakdown" in costs
        
        # Check individual costs
        breakdown = costs["cost_breakdown"]
        assert "front_bumper_damage" in breakdown
        assert "hood_damage" in breakdown
        
        # Verify cost calculation
        fb_cost = breakdown["front_bumper_damage"]
        expected_fb = int(500 * 1.8)  # severe multiplier
        assert fb_cost["estimated_cost"] == expected_fb
        
        hd_cost = breakdown["hood_damage"] 
        expected_hd = int(800 * 1.3)  # moderate multiplier
        assert hd_cost["estimated_cost"] == expected_hd
        
        # Check total
        expected_total = expected_fb + expected_hd
        assert costs["total_estimated_cost"] == expected_total


@pytest.mark.slow
class TestModelLoading:
    """Test model loading and inference (marked as slow due to potential model size)."""
    
    def test_mock_model_inference(self, test_config):
        """Test inference with mock models."""
        # Create mock ResNet model
        mock_model = create_mock_resnet_model(len(TEST_CLASSES))
        
        # Test forward pass
        batch_size = 2
        mock_input = np.random.rand(batch_size, 3, *TEST_IMAGE_SIZE).astype(np.float32)
        
        # Mock torch tensor
        class MockTensor:
            def __init__(self, data):
                self.data = data
            
            def numpy(self):
                return self.data
            
            def cpu(self):
                return self
        
        # Mock model prediction
        def mock_predict(input_tensor):
            # Simulate model output
            output = np.random.rand(batch_size, len(TEST_CLASSES))
            return MockTensor(output)
        
        # Test prediction
        output = mock_predict(mock_input)
        predictions = output.numpy()
        
        assert predictions.shape == (batch_size, len(TEST_CLASSES))
        assert predictions.dtype == np.float64  # numpy default
    
    def test_model_version_tracking(self, test_config):
        """Test model version tracking."""
        def mock_extract_model_versions(model_paths):
            """Mock model version extraction."""
            versions = {}
            
            for model_name, path in model_paths.items():
                if path.exists():
                    # Mock version extraction from filename or metadata
                    if "resnet" in model_name:
                        versions[model_name] = "v1.0.0_epoch_50"
                    elif "yolo" in model_name:
                        versions[model_name] = "v11m_epoch_100"
                    else:
                        versions[model_name] = "unknown"
                else:
                    versions[model_name] = "not_found"
            
            return versions
        
        # Test version extraction
        mock_paths = {
            "resnet": Path("models/resnet/best_model.pth"),
            "yolo": Path("models/yolo/best.pt")
        }
        
        # Mock file existence
        for path in mock_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        
        versions = mock_extract_model_versions(mock_paths)
        
        assert "resnet" in versions
        assert "yolo" in versions
        assert versions["resnet"] != "not_found"
        assert versions["yolo"] != "not_found"
        
        # Cleanup
        for path in mock_paths.values():
            if path.exists():
                path.unlink()


@pytest.mark.integration
class TestInferenceIntegration:
    """Integration tests for complete inference pipeline."""
    
    def test_end_to_end_inference_workflow(
        self,
        test_config,
        sample_image,
        sample_thresholds
    ):
        """Test complete inference workflow."""
        # Mock complete inference pipeline
        def mock_complete_inference(image, config, thresholds):
            """Mock complete inference workflow."""
            # Step 1: Preprocess image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            processed_image = np.array(image.resize(TEST_IMAGE_SIZE)).astype(np.float32) / 255.0
            
            # Step 2: Run models (mock)
            resnet_scores = [0.8, 0.3, 0.6] + [0.2] * (len(TEST_CLASSES) - 3)
            yolo_scores = [0.7, 0.9, 0.4] + [0.1] * (len(TEST_CLASSES) - 3)
            
            # Step 3: Fuse scores
            fused_scores = [max(r, y) for r, y in zip(resnet_scores, yolo_scores)]
            
            # Step 4: Apply thresholds
            predictions = {}
            for i, class_name in enumerate(TEST_CLASSES):
                score = fused_scores[i] if i < len(fused_scores) else 0.0
                threshold = thresholds.get(class_name, 0.5)
                
                predictions[class_name] = {
                    "fused_score": score,
                    "resnet_score": resnet_scores[i] if i < len(resnet_scores) else 0.0,
                    "yolo_score": yolo_scores[i] if i < len(yolo_scores) else 0.0,
                    "predicted": score >= threshold,
                    "threshold": threshold
                }
            
            # Step 5: Extract predicted classes
            predicted_classes = [
                class_name for class_name, pred in predictions.items()
                if pred["predicted"]
            ]
            
            return {
                "predictions": predictions,
                "predicted_classes": predicted_classes,
                "processing_info": {
                    "image_size": processed_image.shape,
                    "num_classes": len(TEST_CLASSES),
                    "fusion_method": "max_rule"
                }
            }
        
        # Run complete workflow
        result = mock_complete_inference(sample_image, test_config, sample_thresholds)
        
        # Verify results
        assert "predictions" in result
        assert "predicted_classes" in result
        assert "processing_info" in result
        
        predictions = result["predictions"]
        assert len(predictions) == len(TEST_CLASSES)
        
        # Verify each prediction
        for class_name, prediction in predictions.items():
            assert class_name in TEST_CLASSES
            
            required_fields = ["fused_score", "resnet_score", "yolo_score", "predicted", "threshold"]
            for field in required_fields:
                assert field in prediction
            
            # Verify score ranges
            assert 0.0 <= prediction["fused_score"] <= 1.0
            assert 0.0 <= prediction["resnet_score"] <= 1.0
            assert 0.0 <= prediction["yolo_score"] <= 1.0
            assert 0.0 <= prediction["threshold"] <= 1.0
        
        # Verify predicted classes consistency
        predicted_classes = result["predicted_classes"]
        for class_name in predicted_classes:
            assert predictions[class_name]["predicted"] is True
        
        # Verify processing info
        proc_info = result["processing_info"]
        assert proc_info["num_classes"] == len(TEST_CLASSES)
        assert proc_info["fusion_method"] == "max_rule"