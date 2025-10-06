#!/usr/bin/env python3
"""
Prepare a subset of 2500 images for batch processing.
This creates a separate directory with exactly 2500 images so we know which ones were used.
"""

import os
import shutil
from pathlib import Path

def prepare_dataset():
    """Copy first 2500 images to a separate directory"""
    
    # Paths
    source_dir = Path("../data/processed/yolo/images")
    target_dir = Path("../data/processed/yolo/images_2500")
    
    print("="*60)
    print("PREPARING 2500 IMAGE DATASET")
    print("="*60)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print("="*60)
    
    # Check source directory
    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return False
    
    # Create target directory
    target_dir.mkdir(exist_ok=True)
    
    # Find all images
    print("\n[1/3] Scanning for images...")
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(source_dir.glob(f'*{ext}'))
        all_images.extend(source_dir.glob(f'*{ext.upper()}'))
    
    # Sort to ensure consistent order
    all_images = sorted(all_images)
    
    print(f"  Found {len(all_images)} total images")
    
    if len(all_images) < 2500:
        print(f"  WARNING: Only {len(all_images)} images available (less than 2500)")
        num_to_copy = len(all_images)
    else:
        num_to_copy = 2500
    
    # Select first 2500 images
    selected_images = all_images[:num_to_copy]
    
    print(f"\n[2/3] Copying first {num_to_copy} images...")
    
    # Copy images with progress
    for i, img_path in enumerate(selected_images, 1):
        target_path = target_dir / img_path.name
        
        # Skip if already exists
        if target_path.exists():
            continue
        
        shutil.copy2(img_path, target_path)
        
        if i % 100 == 0:
            print(f"  Copied {i}/{num_to_copy} images...")
    
    print(f"  ✓ Copied all {num_to_copy} images")
    
    # Save list of image names
    print("\n[3/3] Saving image list...")
    list_file = target_dir / "image_list.txt"
    
    with open(list_file, 'w') as f:
        for img in selected_images:
            f.write(f"{img.name}\n")
    
    print(f"  ✓ Saved image list to: {list_file}")
    
    # Summary
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"Total images copied: {num_to_copy}")
    print(f"Target directory: {target_dir}")
    print(f"Image list: {list_file}")
    print("="*60)
    print("\nNext step: Run batch processing on this dataset")
    print(f"  python batch_to_coco_2500.py")
    print("="*60)
    
    return True

if __name__ == "__main__":
    prepare_dataset()
