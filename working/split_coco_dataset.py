#!/usr/bin/env python3
"""
Split COCO dataset into train/val/test sets
Maintains class distribution across splits
"""

import json
import random
import sys
from pathlib import Path
from collections import defaultdict


def split_coco_dataset(input_json, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split COCO dataset into train/val/test sets.
    
    Args:
        input_json: Path to input COCO JSON
        output_dir: Directory to save split JSONs
        train_ratio: Proportion for training (default: 0.70)
        val_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for testing (default: 0.15)
        seed: Random seed for reproducibility
    """
    print("=" * 60)
    print("COCO DATASET SPLITTING")
    print("=" * 60)
    print(f"\nInput: {input_json}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}\n")
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        print(f"❌ Error: Ratios must sum to 1.0 (got {train_ratio + val_ratio + test_ratio})")
        return
    
    # Load COCO data
    print("Loading COCO JSON...")
    with open(input_json, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    print(f"✅ Loaded: {len(images)} images, {len(annotations)} annotations, {len(categories)} categories\n")
    
    # Group annotations by image
    image_annotations = defaultdict(list)
    for ann in annotations:
        image_annotations[ann['image_id']].append(ann)
    
    # Separate images with and without damage
    images_with_damage = [img for img in images if img['id'] in image_annotations]
    images_without_damage = [img for img in images if img['id'] not in image_annotations]
    
    print(f"Images with damage: {len(images_with_damage)}")
    print(f"Images without damage: {len(images_without_damage)}\n")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle images
    random.shuffle(images_with_damage)
    random.shuffle(images_without_damage)
    
    # Calculate split indices for damaged images
    n_damaged = len(images_with_damage)
    train_end = int(n_damaged * train_ratio)
    val_end = train_end + int(n_damaged * val_ratio)
    
    train_damaged = images_with_damage[:train_end]
    val_damaged = images_with_damage[train_end:val_end]
    test_damaged = images_with_damage[val_end:]
    
    # Calculate split indices for non-damaged images
    n_undamaged = len(images_without_damage)
    train_end_undamaged = int(n_undamaged * train_ratio)
    val_end_undamaged = train_end_undamaged + int(n_undamaged * val_ratio)
    
    train_undamaged = images_without_damage[:train_end_undamaged]
    val_undamaged = images_without_damage[train_end_undamaged:val_end_undamaged]
    test_undamaged = images_without_damage[val_end_undamaged:]
    
    # Combine damaged and undamaged images
    train_images = train_damaged + train_undamaged
    val_images = val_damaged + val_undamaged
    test_images = test_damaged + test_undamaged
    
    # Shuffle again to mix damaged and undamaged
    random.shuffle(train_images)
    random.shuffle(val_images)
    random.shuffle(test_images)
    
    print("Split sizes:")
    print(f"  Train: {len(train_images)} images ({len(train_damaged)} damaged, {len(train_undamaged)} undamaged)")
    print(f"  Val:   {len(val_images)} images ({len(val_damaged)} damaged, {len(val_undamaged)} undamaged)")
    print(f"  Test:  {len(test_images)} images ({len(test_damaged)} damaged, {len(test_undamaged)} undamaged)\n")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Helper function to create split JSON
    def create_split_json(split_images, split_name):
        split_image_ids = {img['id'] for img in split_images}
        
        # Filter annotations
        split_annotations = [ann for ann in annotations if ann['image_id'] in split_image_ids]
        
        # Create COCO structure
        split_coco = {
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', []),
            'images': split_images,
            'annotations': split_annotations,
            'categories': categories
        }
        
        # Save JSON
        output_path = output_dir / f"annotations_{split_name}.json"
        with open(output_path, 'w') as f:
            json.dump(split_coco, f, indent=2)
        
        print(f"✅ Saved {split_name} set: {output_path}")
        print(f"   - Images: {len(split_images)}")
        print(f"   - Annotations: {len(split_annotations)}")
        if split_images:
            print(f"   - Avg annotations per image: {len(split_annotations) / len(split_images):.2f}\n")
        
        return split_coco
    
    # Create split JSONs
    print("Creating split files...")
    train_coco = create_split_json(train_images, 'train')
    val_coco = create_split_json(val_images, 'val')
    test_coco = create_split_json(test_images, 'test')
    
    # Create image list files for reference
    for split_name, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
        list_path = output_dir / f"image_list_{split_name}.txt"
        with open(list_path, 'w') as f:
            for img in split_images:
                f.write(f"{img['file_name']}\n")
        print(f"✅ Saved image list: {list_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SPLIT COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"\nFiles created:")
    print(f"  - annotations_train.json ({len(train_images)} images, {len(train_coco['annotations'])} annotations)")
    print(f"  - annotations_val.json ({len(val_images)} images, {len(val_coco['annotations'])} annotations)")
    print(f"  - annotations_test.json ({len(test_images)} images, {len(test_coco['annotations'])} annotations)")
    print(f"  - image_list_train.txt")
    print(f"  - image_list_val.txt")
    print(f"  - image_list_test.txt")
    print("=" * 60)


def main():
    # Configuration
    batch_output_dir = Path("../data/processed/yolo/batch_output")
    
    # Find input JSON
    if len(sys.argv) > 1:
        input_json = Path(sys.argv[1])
    else:
        json_files = list(batch_output_dir.glob("annotations_2500_*.json"))
        if not json_files:
            print("❌ No COCO JSON file found")
            return
        input_json = max(json_files, key=lambda p: p.stat().st_mtime)
    
    if not input_json.exists():
        print(f"❌ File not found: {input_json}")
        return
    
    # Output directory
    output_dir = batch_output_dir / "splits"
    
    # Split dataset
    split_coco_dataset(
        input_json=input_json,
        output_dir=output_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )


if __name__ == "__main__":
    main()
