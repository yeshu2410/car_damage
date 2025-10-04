"""
Test inference API functionality.
"""

import pytest
import json
import base64
from io import BytesIO
from typing import Dict, Any

from fastapi.testclient import TestClient
from PIL import Image

from conftest import TEST_CLASSES, TEST_IMAGE_SIZE


@pytest.fixture
def mock_models(test_models_dir, monkeypatch):
    """Mock model loading to avoid heavy dependencies in tests."""
    
    class MockInferencePipeline:
        """Mock inference pipeline for testing."""
        
        def __init__(self, config_path=None):
            self.resnet_model = "mock_resnet"
            self.yolo_model = "mock_yolo"
            self.model_versions = {"resnet": "test_v1", "yolo": "test_v1"}
            self.inference_stats = {
                "total_inferences": 0,
                "avg_resnet_time": 0.05,
                "avg_yolo_time": 0.03,
                "avg_fusion_time": 0.01,
                "avg_total_time": 0.09
            }
        
        def load_models(self):
            """Mock model loading."""
            pass
        
        def predict(self, image):
            """Mock prediction."""
            # Simulate realistic prediction results
            predictions = {}
            for i, class_name in enumerate(TEST_CLASSES):
                score = 0.7 if i < 2 else 0.3  # First 2 classes predicted as positive
                predictions[class_name] = {
                    "fused_score": score,
                    "resnet_score": score - 0.05,
                    "yolo_score": score + 0.05,
                    "predicted": score >= 0.5,
                    "threshold": 0.5,
                    "fusion_method": "max_rule"
                }
            
            return {
                "predictions": predictions,
                "predicted_classes": [cls for cls in TEST_CLASSES[:2]],  # First 2 classes
                "model_versions": self.model_versions,
                "inference_times": {
                    "resnet_time": 0.05,
                    "yolo_time": 0.03,
                    "fusion_time": 0.01,
                    "total_time": 0.09
                },
                "image_info": {
                    "size": image.size,
                    "mode": image.mode
                },
                "pipeline_version": "1.0.0",
                "timestamp": 1699999999.0
            }
        
        def get_stats(self):
            """Mock stats."""
            return {
                **self.inference_stats,
                "models_loaded": {"resnet": True, "yolo": True},
                "device": "cpu",
                "model_versions": self.model_versions
            }
        
        def warmup(self, num_warmup=3):
            """Mock warmup."""
            pass
    
    class MockPredictionMapper:
        """Mock prediction mapper for testing."""
        
        def __init__(self, config_path=None):
            pass
        
        def map_predictions_to_codes(self, predictions):
            """Mock mapping to correction codes."""
            predicted_classes = [cls for cls, data in predictions.items() if data.get("predicted", False)]
            
            return {
                "predicted_classes": predicted_classes,
                "correction_codes": [
                    {
                        "class_name": cls,
                        "correction_code": f"{cls.upper()}_001",
                        "description": f"{cls.replace('_', ' ').title()} damage",
                        "category": "exterior",
                        "confidence": predictions[cls]["fused_score"],
                        "labor_type": "repair",
                        "estimated_cost_range": [300, 800]
                    }
                    for cls in predicted_classes
                ],
                "parts_information": [
                    {
                        "part_name": cls.split('_')[0] + "_part",
                        "part_code": f"P_{cls.upper()}_001",
                        "description": f"{cls.replace('_', ' ').title()} part",
                        "category": "exterior",
                        "estimated_labor_hours": 2.5,
                        "common_damage_types": ["scratches", "dents"],
                        "detected_damage": cls,
                        "confidence": predictions[cls]["fused_score"]
                    }
                    for cls in predicted_classes
                ],
                "severity_analysis": {
                    cls: {
                        "severity_level": "moderate",
                        "confidence": predictions[cls]["fused_score"],
                        "repair_action": "repair",
                        "urgency": "medium",
                        "estimated_cost_multiplier": 1.2
                    }
                    for cls in predicted_classes
                },
                "damage_assessment": {
                    "overall_severity": "moderate",
                    "total_estimated_cost": len(predicted_classes) * 550,
                    "repair_priority": "medium",
                    "total_labor_hours": len(predicted_classes) * 2.5,
                    "safety_impact": "low",
                    "damage_count": len(predicted_classes)
                },
                "repair_recommendations": [
                    {
                        "damage_type": cls,
                        "affected_part": cls.split('_')[0] + "_part",
                        "severity": "moderate",
                        "recommended_action": "repair",
                        "urgency": "medium",
                        "description": f"Professional repair recommended for {cls.replace('_', ' ')}",
                        "estimated_cost": {
                            "estimated_cost": 550,
                            "cost_range": [440, 715],
                            "currency": "USD"
                        }
                    }
                    for cls in predicted_classes
                ]
            }
    
    # Patch the imports
    monkeypatch.setattr("src.infer.pipeline.InferencePipeline", MockInferencePipeline)
    monkeypatch.setattr("src.infer.pipeline.create_pipeline", lambda config_path=None: MockInferencePipeline(config_path))
    monkeypatch.setattr("src.infer.mapping.PredictionMapper", MockPredictionMapper)
    monkeypatch.setattr("src.infer.mapping.create_mapper", lambda config_path=None: MockPredictionMapper(config_path))


@pytest.fixture
def api_client(mock_models):
    """Create FastAPI test client with mocked models."""
    # Import here to ensure mocking is in place
    from src.infer.service import app
    
    client = TestClient(app)
    return client


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_healthz_endpoint_returns_200(self, api_client):
        """Test that /healthz returns 200 status."""
        response = api_client.get("/healthz")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "models_loaded" in data
        assert "uptime_seconds" in data
        
        # Check models_loaded structure
        models_loaded = data["models_loaded"]
        assert "resnet" in models_loaded
        assert "yolo" in models_loaded
        assert isinstance(models_loaded["resnet"], bool)
        assert isinstance(models_loaded["yolo"], bool)
    
    def test_healthz_returns_correct_fields(self, api_client):
        """Test that /healthz returns all required fields."""
        response = api_client.get("/healthz")
        data = response.json()
        
        required_fields = ["status", "timestamp", "version", "models_loaded", "uptime_seconds"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Validate field types
        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["models_loaded"], dict)
        assert isinstance(data["uptime_seconds"], (int, float))


class TestStatsEndpoint:
    """Test statistics endpoint."""
    
    def test_stats_endpoint_returns_service_stats(self, api_client):
        """Test that /stats returns service statistics."""
        response = api_client.get("/stats")
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields
        required_fields = [
            "total_inferences", "avg_resnet_time", "avg_yolo_time",
            "avg_fusion_time", "avg_total_time", "models_loaded",
            "device", "model_versions", "uptime_seconds"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Validate field types
        assert isinstance(data["total_inferences"], int)
        assert isinstance(data["avg_resnet_time"], (int, float))
        assert isinstance(data["avg_yolo_time"], (int, float))
        assert isinstance(data["avg_fusion_time"], (int, float))
        assert isinstance(data["avg_total_time"], (int, float))
        assert isinstance(data["models_loaded"], dict)
        assert isinstance(data["device"], str)
        assert isinstance(data["model_versions"], dict)
        assert isinstance(data["uptime_seconds"], (int, float))


class TestPredictionEndpoint:
    """Test prediction endpoint."""
    
    def test_predict_endpoint_accepts_valid_image(self, api_client, sample_image_base64):
        """Test that /predict accepts valid image data."""
        request_data = {
            "image": {
                "image_data": sample_image_base64,
                "image_format": "JPEG"
            },
            "include_details": True,
            "include_timing": True
        }
        
        response = api_client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required response fields
        required_fields = [
            "predictions", "predicted_classes", "model_versions",
            "pipeline_version", "image_info", "timestamp"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
    
    def test_predict_endpoint_returns_expected_structure(self, api_client, sample_image_base64):
        """Test that /predict returns expected response structure."""
        request_data = {
            "image": {
                "image_data": sample_image_base64,
                "image_format": "JPEG"
            },
            "include_details": True,
            "include_timing": True
        }
        
        response = api_client.post("/predict", json=request_data)
        data = response.json()
        
        # Check predictions structure
        predictions = data["predictions"]
        assert isinstance(predictions, dict)
        
        for class_name, prediction in predictions.items():
            assert class_name in TEST_CLASSES
            
            required_pred_fields = [
                "fused_score", "resnet_score", "yolo_score",
                "predicted", "threshold", "fusion_method"
            ]
            
            for field in required_pred_fields:
                assert field in prediction, f"Missing prediction field: {field}"
            
            # Validate field types and ranges
            assert isinstance(prediction["fused_score"], (int, float))
            assert 0.0 <= prediction["fused_score"] <= 1.0
            
            assert isinstance(prediction["resnet_score"], (int, float))
            assert 0.0 <= prediction["resnet_score"] <= 1.0
            
            assert isinstance(prediction["yolo_score"], (int, float))
            assert 0.0 <= prediction["yolo_score"] <= 1.0
            
            assert isinstance(prediction["predicted"], bool)
            assert isinstance(prediction["threshold"], (int, float))
            assert isinstance(prediction["fusion_method"], str)
        
        # Check predicted_classes
        predicted_classes = data["predicted_classes"]
        assert isinstance(predicted_classes, list)
        
        for class_name in predicted_classes:
            assert class_name in TEST_CLASSES
            assert predictions[class_name]["predicted"] is True
        
        # Check model_versions
        model_versions = data["model_versions"]
        assert "resnet" in model_versions
        assert "yolo" in model_versions
        
        # Check image_info
        image_info = data["image_info"]
        assert "size" in image_info
        assert "mode" in image_info
        assert isinstance(image_info["size"], list)
        assert len(image_info["size"]) == 2
    
    def test_predict_with_details_includes_additional_fields(self, api_client, sample_image_base64):
        """Test that include_details=True adds additional response fields."""
        request_data = {
            "image": {
                "image_data": sample_image_base64,
                "image_format": "JPEG"
            },
            "include_details": True
        }
        
        response = api_client.post("/predict", json=request_data)
        data = response.json()
        
        # Check that detailed fields are present
        detail_fields = [
            "correction_codes", "parts_information", "severity_analysis",
            "damage_assessment", "repair_recommendations"
        ]
        
        for field in detail_fields:
            assert field in data, f"Missing detail field: {field}"
        
        # Validate correction_codes structure
        correction_codes = data["correction_codes"]
        assert isinstance(correction_codes, list)
        
        for code in correction_codes:
            required_code_fields = [
                "class_name", "correction_code", "description",
                "category", "confidence", "labor_type", "estimated_cost_range"
            ]
            
            for field in required_code_fields:
                assert field in code, f"Missing correction code field: {field}"
        
        # Validate damage_assessment structure
        damage_assessment = data["damage_assessment"]
        required_assessment_fields = [
            "overall_severity", "total_estimated_cost", "repair_priority",
            "total_labor_hours", "safety_impact", "damage_count"
        ]
        
        for field in required_assessment_fields:
            assert field in damage_assessment, f"Missing assessment field: {field}"
    
    def test_predict_with_timing_includes_timing_info(self, api_client, sample_image_base64):
        """Test that include_timing=True adds timing information."""
        request_data = {
            "image": {
                "image_data": sample_image_base64,
                "image_format": "JPEG"
            },
            "include_timing": True
        }
        
        response = api_client.post("/predict", json=request_data)
        data = response.json()
        
        # Check timing field is present
        assert "inference_times" in data
        
        timing_info = data["inference_times"]
        required_timing_fields = [
            "resnet_time", "yolo_time", "fusion_time", "total_time"
        ]
        
        for field in required_timing_fields:
            assert field in timing_info, f"Missing timing field: {field}"
            assert isinstance(timing_info[field], (int, float))
            assert timing_info[field] >= 0
    
    def test_predict_rejects_invalid_image_data(self, api_client):
        """Test that /predict rejects invalid image data."""
        # Test invalid base64
        request_data = {
            "image": {
                "image_data": "invalid_base64_data!@#",
                "image_format": "JPEG"
            }
        }
        
        response = api_client.post("/predict", json=request_data)
        
        # Should return validation error
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_predict_handles_missing_fields(self, api_client):
        """Test that /predict handles missing required fields."""
        # Test missing image data
        request_data = {
            "image": {
                "image_format": "JPEG"
                # Missing image_data
            }
        }
        
        response = api_client.post("/predict", json=request_data)
        assert response.status_code == 422
        
        # Test completely missing image
        request_data = {}
        
        response = api_client.post("/predict", json=request_data)
        assert response.status_code == 422


class TestBatchPredictionEndpoint:
    """Test batch prediction endpoint."""
    
    def test_batch_predict_accepts_multiple_images(self, api_client, sample_image_base64):
        """Test that /predict/batch accepts multiple images."""
        request_data = {
            "images": [
                {
                    "image_data": sample_image_base64,
                    "image_format": "JPEG"
                },
                {
                    "image_data": sample_image_base64,
                    "image_format": "JPEG"
                }
            ],
            "include_details": False
        }
        
        response = api_client.post("/predict/batch", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Check batch response structure
        required_fields = [
            "predictions", "batch_size", "total_processing_time",
            "average_time_per_image", "successful_predictions",
            "failed_predictions", "timestamp"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing batch field: {field}"
        
        # Check batch metrics
        assert data["batch_size"] == 2
        assert data["successful_predictions"] <= data["batch_size"]
        assert data["failed_predictions"] <= data["batch_size"]
        assert data["successful_predictions"] + data["failed_predictions"] <= data["batch_size"]
        
        # Check predictions
        predictions = data["predictions"]
        assert isinstance(predictions, list)
        assert len(predictions) == data["successful_predictions"]
    
    def test_batch_predict_enforces_size_limit(self, api_client, sample_image_base64):
        """Test that batch prediction enforces size limits."""
        # Try to send more than 10 images (max batch size)
        request_data = {
            "images": [
                {
                    "image_data": sample_image_base64,
                    "image_format": "JPEG"
                }
            ] * 15  # 15 images, exceeds limit of 10
        }
        
        response = api_client.post("/predict/batch", json=request_data)
        
        # Should return validation error
        assert response.status_code == 422


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for the complete API."""
    
    def test_api_workflow_end_to_end(self, api_client, sample_image_base64):
        """Test complete API workflow from health check to prediction."""
        # Step 1: Check health
        health_response = api_client.get("/healthz")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"
        
        # Step 2: Get stats
        stats_response = api_client.get("/stats")
        assert stats_response.status_code == 200
        
        # Step 3: Run prediction
        predict_response = api_client.post("/predict", json={
            "image": {
                "image_data": sample_image_base64,
                "image_format": "JPEG"
            },
            "include_details": True,
            "include_timing": True
        })
        
        assert predict_response.status_code == 200
        
        prediction_data = predict_response.json()
        assert len(prediction_data["predicted_classes"]) >= 0  # Could be 0 or more
        
        # Step 4: Check updated stats
        updated_stats_response = api_client.get("/stats")
        assert updated_stats_response.status_code == 200
    
    def test_api_error_handling(self, api_client):
        """Test API error handling for various edge cases."""
        # Test non-existent endpoint
        response = api_client.get("/nonexistent")
        assert response.status_code == 404
        
        # Test invalid HTTP method
        response = api_client.delete("/predict")
        assert response.status_code == 405
        
        # Test malformed JSON
        response = api_client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_api_response_consistency(self, api_client, sample_image_base64):
        """Test that API responses are consistent across multiple calls."""
        request_data = {
            "image": {
                "image_data": sample_image_base64,
                "image_format": "JPEG"
            },
            "include_details": False
        }
        
        # Make multiple requests
        responses = []
        for _ in range(3):
            response = api_client.post("/predict", json=request_data)
            assert response.status_code == 200
            responses.append(response.json())
        
        # Check that predicted classes are consistent
        # (Should be deterministic with same input and seed)
        first_prediction = responses[0]["predicted_classes"]
        for response in responses[1:]:
            assert response["predicted_classes"] == first_prediction