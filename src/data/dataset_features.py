"""
Dataset class for feature extraction workflow.
Handles image loading and preprocessing for feature extraction from pre-trained models.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FeatureExtractionDataset(Dataset):
    """Dataset for feature extraction from collision parts images."""
    
    def __init__(
        self,
        annotations_file: Path,
        image_base_dir: Path,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224,
        return_paths: bool = True
    ):
        """
        Initialize the feature extraction dataset.
        
        Args:
            annotations_file: Path to VIA format annotations JSON
            image_base_dir: Base directory containing images
            transform: Optional torchvision transforms
            image_size: Target image size for resizing
            return_paths: Whether to return image paths in metadata
        """
        self.annotations_file = annotations_file
        self.image_base_dir = Path(image_base_dir)
        self.image_size = image_size
        self.return_paths = return_paths
        
        # Load annotations
        with open(annotations_file, "r") as f:
            self.via_data = json.load(f)
        
        # Create list of all samples
        self.samples = list(self.via_data.items())
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default image transforms for feature extraction."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _find_image_path(self, filename: str) -> Optional[Path]:
        """Find the actual image path, checking multiple possible locations."""
        # Check main image directory
        image_path = self.image_base_dir / filename
        if image_path.exists():
            return image_path
        
        # Check no_damage directory
        no_damage_dir = self.image_base_dir.parent / "no_damage_images"
        image_path = no_damage_dir / filename
        if image_path.exists():
            return image_path
        
        # Check subdirectories
        for subdir in self.image_base_dir.iterdir():
            if subdir.is_dir():
                image_path = subdir / filename
                if image_path.exists():
                    return image_path
        
        return None
    
    def _extract_damage_info(self, annotation: Dict) -> Dict:
        """Extract damage information from annotation."""
        damage_info = {
            "has_damage": False,
            "damage_types": [],
            "num_regions": 0,
            "region_areas": [],
            "severities": []
        }
        
        # Handle no_damage case
        file_attrs = annotation.get("file_attributes", {})
        if file_attrs.get("damage_class") == "no_damage":
            return damage_info
        
        # Handle damage regions
        regions = annotation.get("regions", [])
        damage_info["num_regions"] = len(regions)
        
        if regions:
            damage_info["has_damage"] = True
            
            for region in regions:
                region_attrs = region.get("region_attributes", {})
                shape_attrs = region.get("shape_attributes", {})
                
                # Damage type
                damage_type = region_attrs.get("damage_type", "unknown")
                if damage_type not in damage_info["damage_types"]:
                    damage_info["damage_types"].append(damage_type)
                
                # Severity
                severity = region_attrs.get("severity", "medium")
                damage_info["severities"].append(severity)
                
                # Calculate region area (approximate)
                if shape_attrs.get("name") == "polygon":
                    all_x = shape_attrs.get("all_points_x", [])
                    all_y = shape_attrs.get("all_points_y", [])
                    if len(all_x) >= 3 and len(all_y) >= 3:
                        # Simple area calculation using shoelace formula
                        area = 0.5 * abs(sum(all_x[i] * all_y[i+1] - all_x[i+1] * all_y[i] 
                                           for i in range(-1, len(all_x)-1)))
                        damage_info["region_areas"].append(area)
                elif shape_attrs.get("name") == "rect":
                    width = shape_attrs.get("width", 0)
                    height = shape_attrs.get("height", 0)
                    area = width * height
                    damage_info["region_areas"].append(area)
        
        return damage_info
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a sample from the dataset.
        
        Returns:
            image: Transformed image tensor [C, H, W]
            metadata: Dictionary containing image metadata and damage information
        """
        file_key, annotation = self.samples[idx]
        filename = annotation.get("filename", "")
        
        # Load image
        image_path = self._find_image_path(filename)
        if image_path is None:
            raise FileNotFoundError(f"Image not found: {filename}")
        
        try:
            with Image.open(image_path) as img:
                original_size = img.size
                
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Apply transforms
                image = self.transform(img)
        except Exception as e:
            raise RuntimeError(f"Error loading image {filename}: {e}")
        
        # Extract damage information
        damage_info = self._extract_damage_info(annotation)
        
        # Create metadata
        metadata = {
            "filename": filename,
            "file_key": file_key,
            "original_size": original_size,
            "dataset": annotation.get("file_attributes", {}).get("dataset", "unknown"),
            **damage_info
        }
        
        if self.return_paths:
            metadata["image_path"] = str(image_path)
        
        return image, metadata
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            "total_images": len(self.samples),
            "damage_counts": {},
            "severity_counts": {},
            "dataset_sources": {},
            "has_damage_count": 0,
            "no_damage_count": 0
        }
        
        for _, annotation in self.samples:
            damage_info = self._extract_damage_info(annotation)
            
            # Count damage vs no damage
            if damage_info["has_damage"]:
                stats["has_damage_count"] += 1
            else:
                stats["no_damage_count"] += 1
            
            # Count damage types
            for damage_type in damage_info["damage_types"]:
                stats["damage_counts"][damage_type] = stats["damage_counts"].get(damage_type, 0) + 1
            
            # Count severities
            for severity in damage_info["severities"]:
                stats["severity_counts"][severity] = stats["severity_counts"].get(severity, 0) + 1
            
            # Count dataset sources
            dataset_source = annotation.get("file_attributes", {}).get("dataset", "unknown")
            stats["dataset_sources"][dataset_source] = stats["dataset_sources"].get(dataset_source, 0) + 1
        
        return stats


class FeatureBatchCollator:
    """Custom collator for feature extraction batches."""
    
    def __init__(self, return_metadata: bool = True):
        self.return_metadata = return_metadata
    
    def __call__(self, batch: List[Tuple[torch.Tensor, Dict]]) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict]]]:
        """Collate batch of samples."""
        images = torch.stack([item[0] for item in batch])
        
        if self.return_metadata:
            metadata = [item[1] for item in batch]
            return images, metadata
        else:
            return images


def create_feature_extraction_loader(
    annotations_file: Path,
    image_base_dir: Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    return_metadata: bool = True
) -> torch.utils.data.DataLoader:
    """Create data loader for feature extraction."""
    
    dataset = FeatureExtractionDataset(
        annotations_file=annotations_file,
        image_base_dir=image_base_dir,
        image_size=image_size,
        return_paths=return_metadata
    )
    
    collator = FeatureBatchCollator(return_metadata=return_metadata)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator
    )
    
    return loader


def save_features_batch(
    features: torch.Tensor,
    metadata: List[Dict],
    output_dir: Path,
    feature_name: str = "features"
) -> None:
    """Save a batch of features and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert features to numpy for storage efficiency
    features_np = features.cpu().numpy()
    
    for i, (feature_vector, meta) in enumerate(zip(features_np, metadata)):
        filename = meta["filename"]
        feature_file = output_dir / f"{Path(filename).stem}_{feature_name}.npz"
        
        np.savez_compressed(
            feature_file,
            features=feature_vector,
            metadata=meta
        )


def load_features_batch(feature_files: List[Path]) -> Tuple[torch.Tensor, List[Dict]]:
    """Load a batch of features from files."""
    features_list = []
    metadata_list = []
    
    for feature_file in feature_files:
        data = np.load(feature_file, allow_pickle=True)
        features_list.append(data["features"])
        metadata_list.append(data["metadata"].item())
    
    features = torch.from_numpy(np.stack(features_list))
    return features, metadata_list