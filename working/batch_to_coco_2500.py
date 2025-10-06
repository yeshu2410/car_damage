#!/usr/bin/env python3
"""
Batch processing script to convert images to COCO format annotations.
TEST VERSION - Only processes first 10 images
"""

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add YOLO imports
from ultralytics import YOLO

# --- Constants ---
DAMAGE_CLASSES = ['Cracked', 'Scratch', 'Flaking', 'Broken part', 'Corrosion', 'Dent', 'Paint chip', 'Missing part']
CAR_PART_CLASSES = [
    "Quarter-panel", "Front-wheel", "Back-window", "Trunk", "Front-door",
    "Rocker-panel", "Grille", "Windshield", "Front-window", "Back-door",
    "Headlight", "Back-wheel", "Back-windshield", "Hood", "Fender",
    "Tail-light", "License-plate", "Front-bumper", "Back-bumper", "Mirror", "Roof"
]

DAMAGE_MODEL_WEIGHTS_PATH = "./best (1).pt"
PART_MODEL_WEIGHTS_PATH = "./partdetection_yolobest.pt"
DEFAULT_DAMAGE_PRED_THRESHOLD = 0.4
DEFAULT_PART_PRED_THRESHOLD = 0.3

DEVICE = "cuda" if os.system("nvidia-smi > nul 2>&1") == 0 else "cpu"

print(f"Using device: {DEVICE}")

def resize_masks(masks_tensor, target_h, target_w):
    """Resize masks to target dimensions"""
    if masks_tensor is None or masks_tensor.numel() == 0 or masks_tensor.shape[0] == 0:
        return np.zeros((0, target_h, target_w), dtype=bool)
    
    try:
        masks_np_bool = masks_tensor.cpu().numpy().astype(bool)
        
        if masks_np_bool.shape[1] == target_h and masks_np_bool.shape[2] == target_w:
            return masks_np_bool
            
        resized_masks_list = []
        for i in range(masks_np_bool.shape[0]):
            mask = masks_np_bool[i]
            mask_resized = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            resized_masks_list.append(mask_resized.astype(bool))
            
        return np.array(resized_masks_list)
    except Exception as e:
        print(f"Error resizing masks: {e}")
        return np.zeros((0, target_h, target_w), dtype=bool)

def mask_to_polygon(mask):
    """Convert binary mask to polygon coordinates"""
    mask_8bit = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        polygon = contour.flatten().tolist()
        polygons.append(polygon)
    
    return polygons

def process_image_to_coco(image_np_bgr, damage_model, part_model, damage_threshold, part_threshold, image_id, image_filename):
    """
    Process image and return COCO annotations
    """
    coco_annotations = []
    annotation_id = 1
    img_h, img_w = image_np_bgr.shape[:2]
    
    try:
        # --- 1. Predict Damages ---
        print(f"  Running damage detection...")
        damage_results = damage_model.predict(image_np_bgr, verbose=False, device=DEVICE, conf=damage_threshold)
        damage_result = damage_results[0]
        
        if damage_result.masks is None:
            damage_masks_raw = None
        else:
            damage_masks_raw = damage_result.masks.data if hasattr(damage_result.masks, 'data') else damage_result.masks
        
        damage_classes_ids_cpu = damage_result.boxes.cls.cpu().numpy().astype(int) if damage_result.boxes is not None else np.array([])
        damage_boxes_xyxy_cpu = damage_result.boxes.xyxy.cpu() if damage_result.boxes is not None else None
        damage_confidences = damage_result.boxes.conf.cpu().numpy() if damage_result.boxes is not None else np.array([])

        # --- 2. Predict Parts ---
        print(f"  Running part detection...")
        part_results = part_model.predict(image_np_bgr, verbose=False, device=DEVICE, conf=part_threshold)
        part_result = part_results[0]
        
        if part_result.masks is None:
            part_masks_raw = None
        else:
            part_masks_raw = part_result.masks.data if hasattr(part_result.masks, 'data') else part_result.masks
                
        part_classes_ids_cpu = part_result.boxes.cls.cpu().numpy().astype(int) if part_result.boxes is not None else np.array([])

        # --- 3. Resize Masks ---
        if damage_masks_raw is not None:
            damage_masks_np = resize_masks(damage_masks_raw, img_h, img_w)
        else:
            damage_masks_np = np.zeros((0, img_h, img_w), dtype=bool)
            
        if part_masks_raw is not None:
            part_masks_np = resize_masks(part_masks_raw, img_h, img_w)
        else:
            part_masks_np = np.zeros((0, img_h, img_w), dtype=bool)

        # --- 4. Calculate Overlap and Create COCO Annotations ---
        print(f"  Found {len(damage_masks_np)} damage(s), {len(part_masks_np)} part(s)")
        
        overlap_threshold = 0.4
        
        for i in range(len(damage_masks_np)):
            damage_mask = damage_masks_np[i]
            damage_class_id = damage_classes_ids_cpu[i]
            damage_name = DAMAGE_CLASSES[damage_class_id] if damage_class_id < len(DAMAGE_CLASSES) else "Unknown"
            damage_area = np.sum(damage_mask)
            
            if damage_area < 10:
                continue
            
            # Find bbox
            x1, y1, x2, y2 = damage_boxes_xyxy_cpu[i].numpy()
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            
            # Convert mask to polygon
            segmentation = mask_to_polygon(damage_mask)
            
            # Find best matching part
            max_overlap = 0
            assigned_part_name = "Unknown / Outside Parts"
            assigned_part_id = -1
            
            for j in range(len(part_masks_np)):
                part_mask = part_masks_np[j]
                part_class_id = part_classes_ids_cpu[j]
                part_name = CAR_PART_CLASSES[part_class_id] if part_class_id < len(CAR_PART_CLASSES) else "Unknown"
                
                intersection = np.logical_and(damage_mask, part_mask)
                overlap_ratio = np.sum(intersection) / damage_area if damage_area > 0 else 0
                
                if overlap_ratio > max_overlap:
                    max_overlap = overlap_ratio
                    if max_overlap >= overlap_threshold:
                        assigned_part_name = part_name
                        assigned_part_id = int(part_class_id)
            
            # Create COCO annotation
            coco_ann = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(damage_class_id),
                "bbox": bbox,
                "area": float(damage_area),
                "segmentation": segmentation,
                "iscrowd": 0,
                "score": float(damage_confidences[i]) if i < len(damage_confidences) else 1.0,
                # Custom attributes
                "damage_location": assigned_part_name,
                "damage_location_id": assigned_part_id,
                "damage_severity": int(np.random.randint(3, 8)),  # Placeholder
                "damage_type": damage_name,
                "damage_type_id": int(damage_class_id),
                "overlap_ratio": float(max_overlap)
            }
            
            coco_annotations.append(coco_ann)
            annotation_id += 1
            print(f"    - {damage_name} in {assigned_part_name} (overlap: {max_overlap:.2f})")

    except Exception as e:
        print(f"  Error during processing: {e}")
        import traceback
        traceback.print_exc()

    return coco_annotations

def main():
    """Main batch processing function"""
    
    # Configuration
    dataset_dir = "../data/processed/yolo/images_2500"  # Use prepared 2500 image dataset
    output_dir = "../data/processed/yolo/batch_output"
    damage_thresh = DEFAULT_DAMAGE_PRED_THRESHOLD
    part_thresh = DEFAULT_PART_PRED_THRESHOLD
    max_images = None  # Process ALL images in images_2500 directory (all 2500)
    
    print("="*60)
    print("BATCH PROCESSING - 2500 IMAGES")
    print("="*60)
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Damage threshold: {damage_thresh}")
    print(f"Part threshold: {part_thresh}")
    print(f"Max images: {max_images}")
    print("="*60)
    
    # Check dataset directory
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("\n[1/4] Loading models...")
    try:
        print(f"  Loading damage model: {DAMAGE_MODEL_WEIGHTS_PATH}")
        damage_model = YOLO(DAMAGE_MODEL_WEIGHTS_PATH)
        damage_model.to(DEVICE)
        
        print(f"  Loading part model: {PART_MODEL_WEIGHTS_PATH}")
        part_model = YOLO(PART_MODEL_WEIGHTS_PATH)
        part_model.to(DEVICE)
        
        print("  ✓ Models loaded successfully")
    except Exception as e:
        print(f"  ERROR loading models: {e}")
        return
    
    # Find images
    print("\n[2/4] Scanning for images...")
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(dataset_dir).rglob(f'*{ext}'))
        image_files.extend(Path(dataset_dir).rglob(f'*{ext.upper()}'))
    
    # Limit to max_images if specified
    if max_images is not None:
        image_files = image_files[:max_images]
    
    if len(image_files) == 0:
        print(f"  ERROR: No images found in {dataset_dir}")
        return
    
    print(f"  Found {len(image_files)} images")
    
    # Initialize COCO structure
    print("\n[3/4] Processing images...")
    coco_output = {
        "info": {
            "description": "Car Damage Detection - Batch Processing (TEST)",
            "version": "1.0",
            "year": 2025,
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add damage categories
    for idx, damage_class in enumerate(DAMAGE_CLASSES):
        coco_output["categories"].append({
            "id": idx,
            "name": damage_class,
            "supercategory": "damage"
        })
    
    # Process images
    annotation_id = 1
    processed_count = 0
    
    for img_idx, img_path in enumerate(image_files, 1):
        print(f"\n[{img_idx}/{len(image_files)}] Processing: {img_path.name}")
        
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  WARNING: Could not load image")
                continue
            
            img_h, img_w = image.shape[:2]
            
            # Add image to COCO
            image_id = img_idx
            coco_output["images"].append({
                "id": image_id,
                "file_name": img_path.name,
                "width": img_w,
                "height": img_h,
                "date_captured": datetime.now().isoformat()
            })
            
            # Process and get annotations
            annotations = process_image_to_coco(
                image, damage_model, part_model, 
                damage_thresh, part_thresh, 
                image_id, img_path.name
            )
            
            # Update annotation IDs and add to COCO
            for ann in annotations:
                ann["id"] = annotation_id
                annotation_id += 1
                coco_output["annotations"].append(ann)
            
            processed_count += 1
        
        except Exception as e:
            print(f"  ERROR processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save COCO JSON
    print("\n[4/4] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    coco_json_path = os.path.join(output_dir, f"annotations_2500_{timestamp}.json")
    
    with open(coco_json_path, 'w') as f:
        json.dump(coco_output, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"Total images found: {len(image_files)}")
    print(f"Images processed: {processed_count}")
    print(f"Total annotations: {len(coco_output['annotations'])}")
    print(f"\nCOCO JSON saved to:")
    print(f"  {coco_json_path}")
    print("="*60)

if __name__ == "__main__":
    main()
