"""
Common dataset utilities and transforms for the collision parts prediction project.
"""

from typing import Optional
import torchvision.transforms as transforms


def get_transforms(image_size: int = 224, mode: str = 'test') -> transforms.Compose:
    """
    Get image transforms for inference.
    
    Args:
        image_size: Target image size (default: 224)
        mode: Transform mode - 'train' or 'test' (default: 'test')
        
    Returns:
        Composed transforms for the specified mode
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # test/inference mode
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get transforms specifically for inference (no augmentation).
    
    Args:
        image_size: Target image size (default: 224)
        
    Returns:
        Inference transforms
    """
    return get_transforms(image_size=image_size, mode='test')