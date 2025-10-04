"""
PyTorch Dataset class for ResNet training.
Returns image tensors and multi-hot encoded class vectors for multi-label classification.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CollisionPartsDataset(Dataset):
    """Dataset for collision parts prediction with multi-label classification."""
    
    def __init__(
        self,
        annotations_file: Path,
        image_base_dir: Path,
        class_names: List[str],
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224,
        split: str = "train"
    ):
        """
        Initialize the dataset.
        
        Args:
            annotations_file: Path to VIA format annotations JSON
            image_base_dir: Base directory containing images
            class_names: List of class names in order
            transform: Optional torchvision transforms
            image_size: Target image size for resizing
            split: Dataset split ('train', 'val', 'test')
        """
        self.annotations_file = annotations_file
        self.image_base_dir = Path(image_base_dir)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.split = split
        self.image_size = image_size
        
        # Load annotations
        with open(annotations_file, "r") as f:
            self.via_data = json.load(f)
        
        # Filter annotations for this split
        self.samples = self._filter_split()
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
            
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    def _filter_split(self) -> List[Dict]:
        """Filter annotations based on split."""
        samples = []
        
        for file_key, annotation in self.via_data.items():
            file_attrs = annotation.get("file_attributes", {})
            annotation_split = file_attrs.get("split", "train")
            
            if annotation_split == self.split:
                samples.append({
                    "file_key": file_key,
                    "annotation": annotation
                })
        
        return samples
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default image transforms based on split."""
        if self.split == "train":
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
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
    
    def _create_multi_hot_vector(self, annotation: Dict) -> torch.Tensor:
        """Create multi-hot encoded class vector from annotation."""
        multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        
        # Handle no_damage case
        file_attrs = annotation.get("file_attributes", {})
        if file_attrs.get("damage_class") == "no_damage":
            if "no_damage" in self.class_to_idx:
                multi_hot[self.class_to_idx["no_damage"]] = 1.0
            return multi_hot
        
        # Handle damage regions
        regions = annotation.get("regions", [])
        for region in regions:
            region_attrs = region.get("region_attributes", {})
            damage_type = region_attrs.get("damage_type", "")
            
            if damage_type in self.class_to_idx:
                multi_hot[self.class_to_idx[damage_type]] = 1.0
        
        # If no valid classes found, mark as no_damage
        if multi_hot.sum() == 0 and "no_damage" in self.class_to_idx:
            multi_hot[self.class_to_idx["no_damage"]] = 1.0
        
        return multi_hot
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a sample from the dataset.
        
        Returns:
            image: Transformed image tensor [C, H, W]
            target: Multi-hot encoded class vector [num_classes]
            metadata: Additional information about the sample
        """
        sample = self.samples[idx]
        annotation = sample["annotation"]
        filename = annotation.get("filename", "")
        
        # Load image
        image_path = self._find_image_path(filename)
        if image_path is None:
            raise FileNotFoundError(f"Image not found: {filename}")
        
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Apply transforms
                image = self.transform(img)
        except Exception as e:
            raise RuntimeError(f"Error loading image {filename}: {e}")
        
        # Create multi-hot target vector
        target = self._create_multi_hot_vector(annotation)
        
        # Create metadata
        metadata = {
            "filename": filename,
            "file_key": sample["file_key"],
            "image_path": str(image_path),
            "num_regions": len(annotation.get("regions", [])),
            "has_damage": target.sum() > 0 and not (len(target) > 0 and target[self.class_to_idx.get("no_damage", -1)] == 1.0 and target.sum() == 1.0)
        }
        
        return image, target, metadata
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        class_counts = torch.zeros(self.num_classes)
        
        for sample in self.samples:
            target = self._create_multi_hot_vector(sample["annotation"])
            class_counts += target
        
        # Avoid division by zero
        class_counts = torch.clamp(class_counts, min=1)
        
        # Calculate inverse frequency weights
        total_samples = len(self.samples)
        class_weights = total_samples / (self.num_classes * class_counts)
        
        return class_weights
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset."""
        class_counts = {cls: 0 for cls in self.class_names}
        
        for sample in self.samples:
            target = self._create_multi_hot_vector(sample["annotation"])
            for idx, count in enumerate(target):
                if count > 0:
                    class_counts[self.class_names[idx]] += 1
        
        return class_counts


def create_data_loaders(
    train_annotations: Path,
    val_annotations: Path,
    test_annotations: Path,
    image_base_dir: Path,
    class_names: List[str],
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create data loaders for train, validation, and test sets."""
    
    # Create datasets
    train_dataset = CollisionPartsDataset(
        annotations_file=train_annotations,
        image_base_dir=image_base_dir,
        class_names=class_names,
        image_size=image_size,
        split="train"
    )
    
    val_dataset = CollisionPartsDataset(
        annotations_file=val_annotations,
        image_base_dir=image_base_dir,
        class_names=class_names,
        image_size=image_size,
        split="val"
    )
    
    test_dataset = CollisionPartsDataset(
        annotations_file=test_annotations,
        image_base_dir=image_base_dir,
        class_names=class_names,
        image_size=image_size,
        split="test"
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader