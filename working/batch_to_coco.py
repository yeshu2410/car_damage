#!/usr/bin/env python3
"""
Batch processing script to convert dataset images to COCO annotations
WITHOUT saving annotated images - only generates JSON file
"""

import os
import sys
import cv2
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging

# --- YOLOv8 Imports ---
from ultralytics import YOLO

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# --- Device Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# --- Load Models ---
logger.info("Loading models...")
damage_model = YOLO(DAMAGE_MODEL_WEIGHTS_PATH)
damage_model.to(DEVICE)
part_model = YOLO(PART_MODEL_WEIGHTS_PATH)
part_model.to(DEVICE)
logger.info("Models loaded successfully")


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
    except Exception as e_resize:
        logger.error(f"Error resizing masks: {e_resize}")
        return np.zeros((0, target_h, target_w), dtype=bool)


def mask_to_polygon(mask):
    """Convert binary mask to polygon coordinates"""
    mask_8bit = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if contour.shape[0] < 3:  # Skip if too few points
            continue
        # Flatten to [x1, y1, x2, y2, ...]
        polygon = contour.flatten().tolist()
        polygons.append(polygon)
    
    return polygons


def process_image_for_coco(image_np_bgr, damage_threshold, part_threshold, image_id, image_filename):
    """
    Process car image and return COCO annotations ONLY (no visualization)
    
    Returns:
        coco_annotations: List of COCO annotation dictionaries
    """
    coco_annotations = []
    annotation_id = 1
    img_h, img_w = image_np_bgr.shape[:2]
    
    try:
        # --- 1. Predict Damages ---
        damage_results = damage_model.predict(image_np_bgr, verbose=False, device=DEVICE, conf=damage_threshold)
        damage_result = damage_results[0]
        
        if damage_result.masks is None:
            damage_masks_raw = torch.empty((0,0,0), device=DEVICE)
        else:
            damage_masks_raw = damage_result.masks.data if hasattr(damage_result.masks, 'data') else damage_result.masks
        
        damage_classes_ids_cpu = damage_result.boxes.cls.cpu().numpy().astype(int) if damage_result.boxes is not None else np.array([])
        damage_boxes_xyxy_cpu = damage_result.boxes.xyxy.cpu() if damage_result.boxes is not None else torch.empty((0,4))
        damage_confidences = damage_result.boxes.conf.cpu().numpy() if damage_result.boxes is not None else np.array([])

        # --- 2. Predict Parts ---
        part_results = part_model.predict(image_np_bgr, verbose=False, device=DEVICE, conf=part_threshold)
        part_result = part_results[0]
        
        if part_result.masks is None:
            part_masks_raw = torch.empty((0,0,0), device=DEVICE)
        else:
            part_masks_raw = part_result.masks.data if hasattr(part_result.masks, 'data') else part_result.masks
                
        part_classes_ids_cpu = part_result.boxes.cls.cpu().numpy().astype(int) if part_result.boxes is not None else np.array([])

        # --- 3. Resize Masks ---
        damage_masks_np = resize_masks(damage_masks_raw, img_h, img_w)
        part_masks_np = resize_masks(part_masks_raw, img_h, img_w)

        # --- 4. Calculate Overlap and Create COCO Annotations ---
        overlap_threshold = 0.4
        
        # Process damage masks
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

    except Exception as e:
        logger.error(f"Error during processing: {e}")

    return coco_annotations


def batch_process_to_coco(dataset_dir, output_dir, damage_thresh=0.4, part_thresh=0.3):
    """
    Process all images in dataset directory and save COCO annotations JSON ONLY
    NO IMAGE SAVING - only generates annotation JSON file
    """
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory '{dataset_dir}' not found")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(dataset_dir).rglob(f'*{ext}'))
        image_files.extend(Path(dataset_dir).rglob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        logger.error(f"No images found in {dataset_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Initialize COCO structure
    coco_output = {
        "info": {
            "description": "Car Damage Detection - Batch Processing",
            "version": "1.0",
            "year": 2025,
            "date_created": datetime.now().isoformat(),
            "contributor": "YOLO Damage Detection System"
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
    global_annotation_id = 1
    processed_count = 0
    
    for img_idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Could not load {img_path}")
                continue
            
            img_h, img_w = image.shape[:2]
            
            # Add image to COCO
            image_id = img_idx + 1
            coco_output["images"].append({
                "id": image_id,
                "file_name": img_path.name,
                "width": img_w,
                "height": img_h,
                "date_captured": datetime.now().isoformat()
            })
            
            # Process and get annotations (NO IMAGE SAVING)
            annotations = process_image_for_coco(
                image, damage_thresh, part_thresh, image_id, img_path.name
            )
            
            # Update annotation IDs and add to COCO
            for ann in annotations:
                ann["id"] = global_annotation_id
                global_annotation_id += 1
                coco_output["annotations"].append(ann)
            
            processed_count += 1
            
            # Clear memory periodically
            if processed_count % 50 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue
    
    # Save COCO JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    coco_json_path = os.path.join(output_dir, f"coco_annotations_{timestamp}.json")
    
    logger.info(f"Saving COCO annotations to: {coco_json_path}")
    with open(coco_json_path, 'w') as f:
        json.dump(coco_output, f, indent=2)
    
    # Print summary
    summary = f"""
{'='*60}
Batch Processing Complete!
{'='*60}
Total images found:     {len(image_files)}
Images processed:       {processed_count}
Total annotations:      {len(coco_output['annotations'])}

COCO JSON saved to:
{coco_json_path}

NO IMAGES SAVED - Only JSON annotation file created
{'='*60}
"""
    
    logger.info(summary)
    print(summary)
    return coco_json_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process images to COCO annotations (JSON only)')
    parser.add_argument('--dataset', type=str, default='../data/processed/yolo/images',
                       help='Path to dataset directory containing images')
    parser.add_argument('--output', type=str, default='../data/processed/yolo/batch_output',
                       help='Path to output directory for COCO JSON')
    parser.add_argument('--damage-thresh', type=float, default=0.4,
                       help='Damage detection confidence threshold')
    parser.add_argument('--part-thresh', type=float, default=0.3,
                       help='Part detection confidence threshold')
    
    args = parser.parse_args()
    
    print(f"""
Starting Batch Processing
========================
Dataset: {args.dataset}
Output:  {args.output}
Damage threshold: {args.damage_thresh}
Part threshold:   {args.part_thresh}
========================
""")
    
    batch_process_to_coco(
        args.dataset,
        args.output,
        args.damage_thresh,
        args.part_thresh
    )
