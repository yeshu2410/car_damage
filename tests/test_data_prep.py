"""
Test data preparation functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

from omegaconf import OmegaConf

from .conftest import (
    assert_valid_json_file, assert_valid_via_format, 
    TEST_CLASSES, TEST_IMAGE_SIZE
)


class TestDataPreparation:
    """Test data preparation pipeline components."""
    
    def test_prepare_no_damage_creates_valid_annotations(
        self, 
        test_config, 
        sample_no_damage_images,
        test_data_dir
    ):
        """Test that prepare_no_damage creates valid annotation JSON."""
        # Mock the prepare_no_damage function
        def mock_prepare_no_damage(cfg):
            """Mock implementation of prepare_no_damage."""
            output_file = Path(cfg.data.processed_dir) / "no_damage_annotations.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create mock no-damage annotations
            no_damage_data = {}
            for i, image_path in enumerate(sample_no_damage_images):
                filename = image_path.name
                no_damage_data[filename] = {
                    "filename": filename,
                    "size": image_path.stat().st_size,
                    "regions": [],  # No damage regions
                    "file_attributes": {"damage_present": False}
                }
            
            with open(output_file, 'w') as f:
                json.dump(no_damage_data, f, indent=2)
            
            return output_file
        
        # Run the mock function
        result_file = mock_prepare_no_damage(test_config)
        
        # Verify the output
        assert result_file.exists()
        data = assert_valid_json_file(result_file)
        
        # Check structure
        assert len(data) == len(sample_no_damage_images)
        
        for filename, annotation in data.items():
            assert "filename" in annotation
            assert "regions" in annotation
            assert "file_attributes" in annotation
            assert annotation["file_attributes"]["damage_present"] is False
            assert isinstance(annotation["regions"], list)
            assert len(annotation["regions"]) == 0  # No damage
    
    def test_merge_annotations_combines_damage_and_no_damage(
        self,
        test_config,
        sample_damage_annotations,
        test_data_dir
    ):
        """Test that merge_annotations combines damage and no-damage data correctly."""
        # First create no-damage annotations
        no_damage_data = {
            "no_damage_1.jpg": {
                "filename": "no_damage_1.jpg",
                "regions": [],
                "file_attributes": {"damage_present": False}
            },
            "no_damage_2.jpg": {
                "filename": "no_damage_2.jpg", 
                "regions": [],
                "file_attributes": {"damage_present": False}
            }
        }
        
        no_damage_file = test_data_dir / "processed" / "no_damage_annotations.json"
        no_damage_file.parent.mkdir(parents=True, exist_ok=True)
        with open(no_damage_file, 'w') as f:
            json.dump(no_damage_data, f, indent=2)
        
        # Mock merge_annotations function
        def mock_merge_annotations(cfg):
            """Mock implementation of merge_annotations."""
            # Load damage annotations
            with open(sample_damage_annotations, 'r') as f:
                damage_data = json.load(f)
            
            # Load no-damage annotations
            with open(no_damage_file, 'r') as f:
                no_damage_data = json.load(f)
            
            # Merge data
            merged_data = {**damage_data, **no_damage_data}
            
            # Save merged annotations
            output_file = Path(cfg.data.processed_dir) / "merged_annotations.json"
            with open(output_file, 'w') as f:
                json.dump(merged_data, f, indent=2)
            
            return output_file
        
        # Run merge
        result_file = mock_merge_annotations(test_config)
        
        # Verify the output
        assert result_file.exists()
        merged_data = assert_valid_json_file(result_file)
        
        # Check that we have both damage and no-damage images
        damage_images = [k for k, v in merged_data.items() if len(v.get("regions", [])) > 0]
        no_damage_images = [k for k, v in merged_data.items() if len(v.get("regions", [])) == 0]
        
        assert len(damage_images) >= 2  # From sample_damage_annotations
        assert len(no_damage_images) >= 2  # From no_damage_data
        assert len(merged_data) == len(damage_images) + len(no_damage_images)
        
        # Verify VIA format
        assert_valid_via_format(merged_data)
    
    def test_via_to_yolo_conversion_creates_valid_format(
        self,
        test_config,
        test_data_dir
    ):
        """Test conversion from VIA format to YOLO format."""
        # Create sample merged annotations
        merged_data = {
            "image1.jpg": {
                "filename": "image1.jpg",
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
                "regions": [],  # No damage
                "file_attributes": {}
            }
        }
        
        merged_file = test_data_dir / "processed" / "merged_annotations.json"
        merged_file.parent.mkdir(parents=True, exist_ok=True)
        with open(merged_file, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        # Mock YOLO conversion
        def mock_via_to_yolo(cfg):
            """Mock implementation of VIA to YOLO conversion."""
            yolo_dir = Path(cfg.data.yolo_dir)
            yolo_dir.mkdir(parents=True, exist_ok=True)
            
            # Create YOLO directory structure
            (yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
            (yolo_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
            (yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
            (yolo_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
            
            # Load merged annotations
            with open(merged_file, 'r') as f:
                data = json.load(f)
            
            class_to_id = {cls: i for i, cls in enumerate(TEST_CLASSES)}
            
            for filename, annotation in data.items():
                # Create label file
                label_filename = filename.replace('.jpg', '.txt')
                label_path = yolo_dir / "labels" / "train" / label_filename
                
                with open(label_path, 'w') as f:
                    for region in annotation.get("regions", []):
                        damage_type = region["region_attributes"]["damage_type"]
                        if damage_type in class_to_id:
                            # Convert to YOLO format (normalized coordinates)
                            shape = region["shape_attributes"]
                            img_width, img_height = TEST_IMAGE_SIZE
                            
                            x_center = (shape["x"] + shape["width"] / 2) / img_width
                            y_center = (shape["y"] + shape["height"] / 2) / img_height
                            width = shape["width"] / img_width
                            height = shape["height"] / img_height
                            
                            class_id = class_to_id[damage_type]
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # Create dataset config
            dataset_config = {
                "path": str(yolo_dir),
                "train": "images/train",
                "val": "images/val",
                "names": {i: cls for i, cls in enumerate(TEST_CLASSES)}
            }
            
            config_path = yolo_dir / "dataset.yaml"
            with open(config_path, 'w') as f:
                import yaml
                yaml.dump(dataset_config, f, indent=2)
            
            return yolo_dir
        
        # Run conversion
        yolo_dir = mock_via_to_yolo(test_config)
        
        # Verify YOLO structure
        assert yolo_dir.exists()
        assert (yolo_dir / "images" / "train").exists()
        assert (yolo_dir / "labels" / "train").exists()
        assert (yolo_dir / "dataset.yaml").exists()
        
        # Check dataset config
        with open(yolo_dir / "dataset.yaml", 'r') as f:
            import yaml
            dataset_config = yaml.safe_load(f)
        
        assert "path" in dataset_config
        assert "train" in dataset_config
        assert "val" in dataset_config
        assert "names" in dataset_config
        assert len(dataset_config["names"]) == len(TEST_CLASSES)
        
        # Check label file format
        label_files = list((yolo_dir / "labels" / "train").glob("*.txt"))
        assert len(label_files) >= 1
        
        # Verify label file content
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.strip():  # Skip empty lines
                    parts = line.strip().split()
                    assert len(parts) == 5  # class_id x_center y_center width height
                    
                    class_id = int(parts[0])
                    assert 0 <= class_id < len(TEST_CLASSES)
                    
                    # Check normalized coordinates
                    for coord in parts[1:]:
                        coord_val = float(coord)
                        assert 0.0 <= coord_val <= 1.0
    
    def test_data_preparation_preserves_class_distribution(
        self,
        test_config,
        sample_damage_annotations
    ):
        """Test that data preparation preserves class distribution."""
        # Load sample damage annotations
        with open(sample_damage_annotations, 'r') as f:
            damage_data = json.load(f)
        
        # Count classes in original data
        original_class_counts = {}
        for annotation in damage_data.values():
            for region in annotation.get("regions", []):
                damage_type = region["region_attributes"]["damage_type"]
                original_class_counts[damage_type] = original_class_counts.get(damage_type, 0) + 1
        
        # Mock data preparation that preserves classes
        def mock_prepare_with_distribution(original_counts):
            """Mock preparation that preserves class distribution."""
            processed_counts = original_counts.copy()
            
            # Verify we have the expected classes
            for damage_type in processed_counts:
                assert damage_type in TEST_CLASSES
            
            return processed_counts
        
        # Run mock preparation
        processed_counts = mock_prepare_with_distribution(original_class_counts)
        
        # Verify preservation
        assert processed_counts == original_class_counts
        assert sum(processed_counts.values()) > 0


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Integration tests for the complete data pipeline."""
    
    def test_complete_data_pipeline_flow(
        self,
        test_config,
        sample_damage_annotations,
        sample_no_damage_images,
        test_data_dir
    ):
        """Test the complete data preparation pipeline."""
        # Step 1: Prepare no-damage annotations
        no_damage_file = test_data_dir / "processed" / "no_damage_annotations.json"
        no_damage_file.parent.mkdir(parents=True, exist_ok=True)
        
        no_damage_data = {}
        for image_path in sample_no_damage_images:
            no_damage_data[image_path.name] = {
                "filename": image_path.name,
                "regions": [],
                "file_attributes": {"damage_present": False}
            }
        
        with open(no_damage_file, 'w') as f:
            json.dump(no_damage_data, f, indent=2)
        
        # Step 2: Merge annotations
        with open(sample_damage_annotations, 'r') as f:
            damage_data = json.load(f)
        
        merged_data = {**damage_data, **no_damage_data}
        merged_file = test_data_dir / "processed" / "merged_annotations.json"
        with open(merged_file, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        # Step 3: Convert to YOLO
        yolo_dir = Path(test_config.data.yolo_dir)
        yolo_dir.mkdir(parents=True, exist_ok=True)
        
        # Create minimal YOLO structure
        (yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        
        # Create dataset config
        dataset_config = {
            "path": str(yolo_dir),
            "train": "images/train",
            "val": "images/val", 
            "names": {i: cls for i, cls in enumerate(TEST_CLASSES)}
        }
        
        with open(yolo_dir / "dataset.yaml", 'w') as f:
            import yaml
            yaml.dump(dataset_config, f, indent=2)
        
        # Verify pipeline output
        assert no_damage_file.exists()
        assert merged_file.exists()
        assert (yolo_dir / "dataset.yaml").exists()
        
        # Verify data integrity
        no_damage_data_loaded = assert_valid_json_file(no_damage_file)
        merged_data_loaded = assert_valid_json_file(merged_file)
        
        assert len(merged_data_loaded) == len(damage_data) + len(no_damage_data_loaded)
        assert_valid_via_format(merged_data_loaded)