"""
Pytest configuration and shared fixtures for collision parts prediction tests.
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, Generator
import shutil
import os

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf, DictConfig

# Test data constants
TEST_IMAGE_SIZE = (224, 224)
TEST_CLASSES = [
    "front_bumper_damage", "rear_bumper_damage", "hood_damage",
    "door_damage", "fender_damage", "headlight_damage"
]


@pytest.fixture(scope="session")
def test_config() -> DictConfig:
    """Create test configuration."""
    config = {
        "data": {
            "classes": TEST_CLASSES,
            "image_size": 224,
            "raw_dir": "tests/data/raw",
            "processed_dir": "tests/data/processed",
            "yolo_dir": "tests/data/yolo",
            "damage_annotations_path": "tests/data/damage_annotations.json"
        },
        "paths": {
            "data_dir": "tests/data",
            "models_dir": "tests/models",
            "output_dir": "tests/outputs"
        },
        "model": {
            "name": "DamageNet",
            "backbone": "resnet50",
            "num_classes": len(TEST_CLASSES)
        },
        "train": {
            "batch_size": 4,
            "epochs": 2,
            "learning_rate": 0.001,
            "device": "cpu"  # Use CPU for tests
        }
    }
    
    return OmegaConf.create(config)


@pytest.fixture(scope="session")
def temp_test_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        yield test_dir


@pytest.fixture(scope="session")
def test_data_dir(temp_test_dir: Path, test_config: DictConfig) -> Path:
    """Create test data directory structure."""
    data_dir = temp_test_dir / "data"
    
    # Create directory structure
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    (data_dir / "yolo").mkdir(parents=True, exist_ok=True)
    
    return data_dir


@pytest.fixture(scope="session")
def test_models_dir(temp_test_dir: Path) -> Path:
    """Create test models directory."""
    models_dir = temp_test_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model subdirectories
    (models_dir / "resnet").mkdir(exist_ok=True)
    (models_dir / "yolo").mkdir(exist_ok=True)
    
    return models_dir


@pytest.fixture(scope="session")
def test_outputs_dir(temp_test_dir: Path) -> Path:
    """Create test outputs directory."""
    outputs_dir = temp_test_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output subdirectories
    (outputs_dir / "evaluation").mkdir(exist_ok=True)
    (outputs_dir / "evaluation" / "comparisons").mkdir(exist_ok=True)
    
    return outputs_dir


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample test image."""
    # Create a simple RGB image with some patterns
    image = Image.new('RGB', TEST_IMAGE_SIZE, color='red')
    
    # Add some simple patterns to make it more realistic
    pixels = image.load()
    for i in range(TEST_IMAGE_SIZE[0]):
        for j in range(TEST_IMAGE_SIZE[1]):
            # Create some simple gradient patterns
            r = min(255, i // 2)
            g = min(255, j // 2)
            b = min(255, (i + j) // 4)
            pixels[i, j] = (r, g, b)
    
    return image


@pytest.fixture
def sample_image_bytes(sample_image: Image.Image) -> bytes:
    """Convert sample image to bytes."""
    import io
    buffer = io.BytesIO()
    sample_image.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture
def sample_image_base64(sample_image_bytes: bytes) -> str:
    """Convert sample image to base64 string."""
    import base64
    return base64.b64encode(sample_image_bytes).decode('utf-8')


@pytest.fixture
def sample_via_annotations() -> Dict[str, Any]:
    """Create sample VIA annotation data."""
    return {
        "image1.jpg": {
            "filename": "image1.jpg",
            "size": 150000,
            "regions": [
                {
                    "shape_attributes": {
                        "name": "rect",
                        "x": 100,
                        "y": 100,
                        "width": 200,
                        "height": 150
                    },
                    "region_attributes": {
                        "damage_type": "front_bumper_damage"
                    }
                }
            ],
            "file_attributes": {}
        },
        "image2.jpg": {
            "filename": "image2.jpg", 
            "size": 180000,
            "regions": [
                {
                    "shape_attributes": {
                        "name": "rect",
                        "x": 50,
                        "y": 75,
                        "width": 180,
                        "height": 120
                    },
                    "region_attributes": {
                        "damage_type": "hood_damage"
                    }
                }
            ],
            "file_attributes": {}
        }
    }


@pytest.fixture
def sample_damage_annotations(test_data_dir: Path) -> Path:
    """Create sample damage annotations file."""
    damage_data = {
        "image1.jpg": {
            "filename": "image1.jpg",
            "regions": [
                {
                    "shape_attributes": {"name": "rect", "x": 100, "y": 100, "width": 200, "height": 150},
                    "region_attributes": {"damage_type": "front_bumper_damage"}
                }
            ]
        },
        "image2.jpg": {
            "filename": "image2.jpg",
            "regions": [
                {
                    "shape_attributes": {"name": "rect", "x": 50, "y": 75, "width": 180, "height": 120},
                    "region_attributes": {"damage_type": "rear_bumper_damage"}
                }
            ]
        }
    }
    
    damage_file = test_data_dir / "damage_annotations.json"
    with open(damage_file, 'w') as f:
        json.dump(damage_data, f, indent=2)
    
    return damage_file


@pytest.fixture
def sample_no_damage_images(test_data_dir: Path) -> list:
    """Create sample no-damage image files."""
    no_damage_dir = test_data_dir / "raw" / "no_damage"
    no_damage_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = []
    for i in range(3):
        # Create simple test images
        image = Image.new('RGB', TEST_IMAGE_SIZE, color=['blue', 'green', 'yellow'][i])
        image_path = no_damage_dir / f"no_damage_{i}.jpg"
        image.save(image_path, 'JPEG')
        image_files.append(image_path)
    
    return image_files


@pytest.fixture
def mock_model_weights(test_models_dir: Path) -> Dict[str, Path]:
    """Create mock model weight files."""
    # Create mock ResNet weights
    resnet_path = test_models_dir / "resnet" / "best_model.pth"
    mock_resnet_state = {
        'model_state_dict': torch.nn.Linear(10, len(TEST_CLASSES)).state_dict(),
        'epoch': 50,
        'best_f1': 0.85
    }
    torch.save(mock_resnet_state, resnet_path)
    
    # Create mock YOLO weights (just an empty file for testing)
    yolo_path = test_models_dir / "yolo" / "best.pt"
    yolo_path.touch()
    
    return {
        'resnet': resnet_path,
        'yolo': yolo_path
    }


@pytest.fixture
def sample_predictions() -> Dict[str, Any]:
    """Create sample prediction data."""
    return {
        "front_bumper_damage": {
            "fused_score": 0.85,
            "resnet_score": 0.82,
            "yolo_score": 0.85,
            "predicted": True,
            "threshold": 0.5,
            "fusion_method": "max_rule"
        },
        "hood_damage": {
            "fused_score": 0.65,
            "resnet_score": 0.60,
            "yolo_score": 0.65,
            "predicted": True,
            "threshold": 0.5,
            "fusion_method": "max_rule"
        },
        "door_damage": {
            "fused_score": 0.25,
            "resnet_score": 0.25,
            "yolo_score": 0.20,
            "predicted": False,
            "threshold": 0.5,
            "fusion_method": "max_rule"
        }
    }


@pytest.fixture
def sample_thresholds() -> Dict[str, float]:
    """Create sample threshold data."""
    return {
        class_name: 0.4 + (i * 0.1) 
        for i, class_name in enumerate(TEST_CLASSES)
    }


@pytest.fixture
def sample_evaluation_results(test_outputs_dir: Path) -> Path:
    """Create sample evaluation results file."""
    results = {
        "test": {
            "overall_metrics": {
                "f1_macro": 0.75,
                "f1_micro": 0.78,
                "precision_micro": 0.80,
                "recall_micro": 0.76,
                "hamming_loss": 0.15,
                "exact_match_ratio": 0.65
            },
            "per_class_metrics": {
                class_name: {
                    "precision": 0.7 + (i * 0.05),
                    "recall": 0.6 + (i * 0.05),
                    "f1_score": 0.65 + (i * 0.05),
                    "support": 100 + (i * 10)
                }
                for i, class_name in enumerate(TEST_CLASSES)
            }
        }
    }
    
    results_file = test_outputs_dir / "evaluation" / "resnet_evaluation_test.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results_file


@pytest.fixture
def correction_codes_config(temp_test_dir: Path) -> Path:
    """Create sample correction codes configuration."""
    codes_data = {
        "damage_types": {
            "front_bumper_damage": {
                "code": "FB001",
                "description": "Front bumper damage",
                "category": "exterior",
                "labor_type": "repair",
                "cost_range": [300, 800]
            },
            "hood_damage": {
                "code": "HD001", 
                "description": "Hood damage",
                "category": "exterior",
                "labor_type": "repair",
                "cost_range": [400, 1000]
            }
        },
        "severity_mapping": {
            "minimal": {"action": "inspect", "urgency": "low", "cost_multiplier": 0.8},
            "minor": {"action": "repair", "urgency": "medium", "cost_multiplier": 1.0},
            "moderate": {"action": "repair", "urgency": "high", "cost_multiplier": 1.3},
            "severe": {"action": "replace", "urgency": "immediate", "cost_multiplier": 1.8}
        }
    }
    
    codes_file = temp_test_dir / "correction_codes.yaml"
    with open(codes_file, 'w') as f:
        import yaml
        yaml.dump(codes_data, f, indent=2)
    
    return codes_file


@pytest.fixture(autouse=True)
def setup_test_environment(
    test_config: DictConfig, 
    test_data_dir: Path, 
    test_models_dir: Path, 
    test_outputs_dir: Path,
    monkeypatch
):
    """Set up test environment with proper paths."""
    # Update config paths to use test directories
    test_config.paths.data_dir = str(test_data_dir)
    test_config.paths.models_dir = str(test_models_dir)
    test_config.paths.output_dir = str(test_outputs_dir)
    test_config.data.raw_dir = str(test_data_dir / "raw")
    test_config.data.processed_dir = str(test_data_dir / "processed")
    test_config.data.yolo_dir = str(test_data_dir / "yolo")
    
    # Set environment variables for tests
    monkeypatch.setenv("COLLISION_PARTS_CONFIG", "test_config.yaml")
    
    # Ensure reproducible results
    torch.manual_seed(42)
    np.random.seed(42)


# Utility functions for tests
def assert_valid_json_file(file_path: Path) -> Dict[str, Any]:
    """Assert that a file exists and contains valid JSON."""
    assert file_path.exists(), f"JSON file does not exist: {file_path}"
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    assert isinstance(data, dict), "JSON data should be a dictionary"
    return data


def assert_valid_via_format(data: Dict[str, Any]) -> None:
    """Assert that data follows VIA annotation format."""
    for filename, annotation in data.items():
        assert "filename" in annotation
        assert "regions" in annotation
        assert isinstance(annotation["regions"], list)
        
        for region in annotation["regions"]:
            assert "shape_attributes" in region
            assert "region_attributes" in region


def assert_valid_thresholds(thresholds: Dict[str, float], classes: list) -> None:
    """Assert that thresholds are valid."""
    assert isinstance(thresholds, dict)
    
    for class_name in classes:
        assert class_name in thresholds, f"Missing threshold for class: {class_name}"
        threshold = thresholds[class_name]
        assert isinstance(threshold, (int, float)), f"Threshold should be numeric: {threshold}"
        assert 0.0 <= threshold <= 1.0, f"Threshold should be in [0, 1]: {threshold}"


def create_mock_resnet_model(num_classes: int):
    """Create a mock ResNet model for testing."""
    import torch.nn as nn
    
    class MockResNet(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(64, num_classes)
        
        def forward(self, x):
            x = self.backbone(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    return MockResNet(num_classes)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark tests containing 'integration' as integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        # Mark API tests as integration tests
        elif "api" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        # Mark model tests as slow
        elif any(word in item.nodeid.lower() for word in ["model", "train", "inference"]):
            item.add_marker(pytest.mark.slow)
        else:
            item.add_marker(pytest.mark.unit)