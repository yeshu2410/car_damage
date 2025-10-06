#!/usr/bin/env python3
"""
Validate COCO JSON annotation file
Checks structure, counts, and data quality
"""

import json
import sys
from pathlib import Path
from collections import Counter


def validate_coco_json(json_path):
    """Validate COCO format JSON file."""
    print("=" * 60)
    print("COCO JSON VALIDATION")
    print("=" * 60)
    print(f"\nFile: {json_path}")
    print(f"Size: {Path(json_path).stat().st_size / (1024*1024):.2f} MB\n")
    
    # Load JSON
    print("Loading JSON...")
    try:
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        print("✅ JSON loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return False
    
    # Check structure
    print("Checking structure...")
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in coco_data:
            print(f"❌ Missing required key: {key}")
            return False
        print(f"✅ Found key: {key}")
    print()
    
    # Basic counts
    num_images = len(coco_data['images'])
    num_annotations = len(coco_data['annotations'])
    num_categories = len(coco_data['categories'])
    
    print(f"📊 COUNTS:")
    print(f"  Images:      {num_images:,}")
    print(f"  Annotations: {num_annotations:,}")
    print(f"  Categories:  {num_categories}")
    print(f"  Avg annotations per image: {num_annotations/num_images:.2f}\n")
    
    # Check categories
    print("📋 DAMAGE CATEGORIES:")
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    for cat_id, cat_name in sorted(categories.items()):
        print(f"  {cat_id}: {cat_name}")
    print()
    
    # Annotation statistics
    print("📈 ANNOTATION STATISTICS:")
    
    # Category distribution
    category_counts = Counter()
    location_counts = Counter()
    images_with_damage = set()
    total_area = 0
    severity_scores = []
    
    for ann in coco_data['annotations']:
        category_counts[ann['category_id']] += 1
        images_with_damage.add(ann['image_id'])
        total_area += ann.get('area', 0)
        
        if 'damage_location' in ann:
            location_counts[ann['damage_location']] += 1
        
        if 'damage_severity' in ann:
            severity_scores.append(ann['damage_severity'])
    
    print(f"\n  Damage type distribution:")
    for cat_id, count in category_counts.most_common():
        cat_name = categories.get(cat_id, f"Unknown-{cat_id}")
        percentage = (count / num_annotations) * 100
        print(f"    {cat_name}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n  Top 10 damaged locations:")
    for location, count in location_counts.most_common(10):
        percentage = (count / num_annotations) * 100
        print(f"    {location}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n  Images with damage: {len(images_with_damage):,} / {num_images:,} ({len(images_with_damage)/num_images*100:.1f}%)")
    print(f"  Images without damage: {num_images - len(images_with_damage):,}")
    
    if severity_scores:
        avg_severity = sum(severity_scores) / len(severity_scores)
        min_severity = min(severity_scores)
        max_severity = max(severity_scores)
        print(f"\n  Severity scores:")
        print(f"    Average: {avg_severity:.2f}")
        print(f"    Range: {min_severity:.2f} - {max_severity:.2f}")
    
    print(f"\n  Average damage area: {total_area/num_annotations:.0f} pixels²")
    
    # Validate sample annotations
    print("\n🔍 VALIDATING SAMPLE ANNOTATIONS:")
    sample_size = min(100, num_annotations)
    errors = []
    
    for i, ann in enumerate(coco_data['annotations'][:sample_size]):
        # Check required fields
        required_fields = ['id', 'image_id', 'category_id', 'bbox', 'area']
        for field in required_fields:
            if field not in ann:
                errors.append(f"Annotation {ann.get('id', i)} missing field: {field}")
        
        # Validate bbox format
        if 'bbox' in ann:
            bbox = ann['bbox']
            if not (isinstance(bbox, list) and len(bbox) == 4):
                errors.append(f"Annotation {ann['id']} has invalid bbox format: {bbox}")
            elif any(x < 0 for x in bbox):
                errors.append(f"Annotation {ann['id']} has negative bbox values: {bbox}")
    
    if errors:
        print(f"❌ Found {len(errors)} errors in sample:")
        for error in errors[:5]:
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    else:
        print(f"✅ All {sample_size} sample annotations are valid")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    if errors:
        print("⚠️  Validation completed with warnings")
    else:
        print("✅ Validation PASSED - COCO JSON is valid!")
    print("=" * 60)
    
    return len(errors) == 0


def main():
    # Find the most recent COCO JSON file
    batch_output_dir = Path("../data/processed/yolo/batch_output")
    
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        # Find most recent annotations file
        json_files = list(batch_output_dir.glob("annotations_2500_*.json"))
        if not json_files:
            print("❌ No COCO JSON file found in batch_output/")
            print(f"   Searched in: {batch_output_dir.absolute()}")
            return
        json_path = max(json_files, key=lambda p: p.stat().st_mtime)
    
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        return
    
    validate_coco_json(json_path)


if __name__ == "__main__":
    main()
