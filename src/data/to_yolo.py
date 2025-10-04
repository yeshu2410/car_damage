"""
Convert VIA polygon annotations to YOLO bounding box format.
Handles both damage regions (polygons to bboxes) and no_damage (full image bbox).
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
from loguru import logger
from omegaconf import DictConfig
from PIL import Image


def polygon_to_bbox(all_x: List[float], all_y: List[float]) -> Tuple[float, float, float, float]:
    """Convert polygon coordinates to bounding box."""
    if not all_x or not all_y:
        return 0.0, 0.0, 0.0, 0.0
    
    min_x = min(all_x)
    max_x = max(all_x)
    min_y = min(all_y)
    max_y = max(all_y)
    
    return min_x, min_y, max_x, max_y


def bbox_to_yolo(bbox: Tuple[float, float, float, float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """Convert absolute bbox coordinates to YOLO format (normalized center, width, height)."""
    min_x, min_y, max_x, max_y = bbox
    
    # Calculate center coordinates and dimensions
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    width = max_x - min_x
    height = max_y - min_y
    
    # Normalize by image dimensions
    norm_center_x = center_x / img_width
    norm_center_y = center_y / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    
    # Clamp to [0, 1]
    norm_center_x = max(0.0, min(1.0, norm_center_x))
    norm_center_y = max(0.0, min(1.0, norm_center_y))
    norm_width = max(0.0, min(1.0, norm_width))
    norm_height = max(0.0, min(1.0, norm_height))
    
    return norm_center_x, norm_center_y, norm_width, norm_height


def get_class_id(class_name: str, class_list: List[str]) -> int:
    """Get class ID from class name."""
    try:
        return class_list.index(class_name)
    except ValueError:
        logger.warning(f"Unknown class: {class_name}, using class 0")
        return 0


def convert_via_to_yolo(
    via_data: Dict,
    output_dir: Path,
    class_list: List[str],
    image_base_dir: Path
) -> None:
    """Convert VIA annotations to YOLO format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    converted_count = 0
    error_count = 0
    
    for file_key, annotation in via_data.items():
        try:
            filename = annotation.get("filename", "")
            if not filename:
                continue
            
            # Find image file
            image_path = None
            for base_dir in [image_base_dir, image_base_dir.parent / "no_damage_images"]:
                potential_path = base_dir / filename
                if potential_path.exists():
                    image_path = potential_path
                    break
            
            if not image_path:
                logger.warning(f"Image not found: {filename}")
                error_count += 1
                continue
            
            # Get image dimensions
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                logger.error(f"Error reading image {filename}: {e}")
                error_count += 1
                continue
            
            # Convert annotations
            yolo_annotations = []
            regions = annotation.get("regions", [])
            
            # Handle no_damage case (full image bbox)
            if annotation.get("file_attributes", {}).get("damage_class") == "no_damage" and not regions:
                class_id = get_class_id("no_damage", class_list)
                # Full image bounding box
                yolo_bbox = (0.5, 0.5, 1.0, 1.0)  # Center at (0.5, 0.5), full width/height
                yolo_annotations.append(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}")
            
            # Handle damage regions
            for region in regions:
                shape_attrs = region.get("shape_attributes", {})
                region_attrs = region.get("region_attributes", {})
                
                damage_type = region_attrs.get("damage_type", "")
                if not damage_type:
                    continue
                
                class_id = get_class_id(damage_type, class_list)
                
                # Convert polygon to bbox
                if shape_attrs.get("name") == "polygon":
                    all_x = shape_attrs.get("all_points_x", [])
                    all_y = shape_attrs.get("all_points_y", [])
                    
                    if len(all_x) >= 3 and len(all_y) >= 3:  # Valid polygon
                        bbox = polygon_to_bbox(all_x, all_y)
                        yolo_bbox = bbox_to_yolo(bbox, img_width, img_height)
                        yolo_annotations.append(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}")
                
                elif shape_attrs.get("name") == "rect":
                    # Handle rectangle annotations
                    x = shape_attrs.get("x", 0)
                    y = shape_attrs.get("y", 0)
                    width = shape_attrs.get("width", 0)
                    height = shape_attrs.get("height", 0)
                    
                    bbox = (x, y, x + width, y + height)
                    yolo_bbox = bbox_to_yolo(bbox, img_width, img_height)
                    yolo_annotations.append(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}")
            
            if yolo_annotations:
                # Copy image to output directory
                output_image_path = images_dir / filename
                if not output_image_path.exists():
                    try:
                        # Copy and convert to RGB if necessary
                        with Image.open(image_path) as img:
                            if img.mode != "RGB":
                                img = img.convert("RGB")
                            img.save(output_image_path, "JPEG", quality=95)
                    except Exception as e:
                        logger.error(f"Error copying image {filename}: {e}")
                        error_count += 1
                        continue
                
                # Save YOLO annotation file
                label_filename = Path(filename).stem + ".txt"
                label_path = labels_dir / label_filename
                
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_annotations) + "\n")
                
                converted_count += 1
                
                if converted_count % 100 == 0:
                    logger.info(f"Converted {converted_count} annotations")
        
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            error_count += 1
            continue
    
    logger.info(f"Successfully converted {converted_count} annotations")
    logger.info(f"Errors: {error_count}")
    
    # Create dataset.yaml for YOLO
    dataset_config = {
        "path": str(output_dir.absolute()),
        "train": "images",
        "val": "images",  # Will be split later
        "test": "images",  # Will be split later
        "nc": len(class_list),
        "names": class_list
    }
    
    with open(output_dir / "dataset.yaml", "w") as f:
        import yaml
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    logger.info(f"YOLO dataset configuration saved to: {output_dir / 'dataset.yaml'}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to convert VIA to YOLO format."""
    logger.info("Starting VIA to YOLO conversion")
    
    # Paths
    via_file = Path(cfg.data.processed_data.merged_annotations)
    output_dir = Path(cfg.data.processed_data.yolo_dir)
    image_base_dir = Path(cfg.data.raw_data.vehide_images_dir)
    
    # Class list from config
    class_list = cfg.data.classes
    
    # Check input file
    if not via_file.exists():
        logger.error(f"VIA annotation file not found: {via_file}")
        logger.info("Please run merge_annotations.py first")
        return
    
    # Load VIA annotations
    with open(via_file, "r") as f:
        via_data = json.load(f)
    
    logger.info(f"Loaded {len(via_data)} VIA annotations")
    logger.info(f"Classes: {class_list}")
    
    # Convert to YOLO
    convert_via_to_yolo(
        via_data=via_data,
        output_dir=output_dir,
        class_list=class_list,
        image_base_dir=image_base_dir
    )
    
    logger.info("VIA to YOLO conversion completed")


if __name__ == "__main__":
    main()