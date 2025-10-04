"""
Merge VEHiDe annotations with no_damage annotations and rename classes.
Creates unified annotation format for the collision parts prediction dataset.
"""

import json
from pathlib import Path
from typing import Dict, List, Set

import hydra
from loguru import logger
from omegaconf import DictConfig


def load_json_annotations(file_path: Path) -> Dict:
    """Load JSON annotations from file."""
    if not file_path.exists():
        logger.warning(f"Annotation file not found: {file_path}")
        return {}
    
    with open(file_path, "r") as f:
        return json.load(f)


def normalize_class_names(class_name: str) -> str:
    """Normalize class names to standard format."""
    class_mapping = {
        # VEHiDe class mappings
        "dent": "dent",
        "scratch": "paint_scratch",
        "paint_scratch": "paint_scratch",
        "rust": "rust",
        "crack": "crack",
        "glass_shatter": "broken_glass",
        "broken_glass": "broken_glass",
        "bumper_dent": "bumper_damage",
        "bumper_damage": "bumper_damage",
        "headlight_broken": "headlight_damage",
        "headlight_damage": "headlight_damage",
        "wheel_damage": "wheel_damage",
        "mirror_damage": "mirror_damage",
        
        # Stanford Cars (no damage)
        "no_damage": "no_damage",
        "clean": "no_damage",
        "undamaged": "no_damage",
    }
    
    # Normalize to lowercase and replace spaces/underscores
    normalized = class_name.lower().replace(" ", "_").replace("-", "_")
    
    return class_mapping.get(normalized, normalized)


def convert_vehide_to_via(vehide_data: Dict, image_base_path: Path) -> Dict:
    """Convert VEHiDe annotation format to VIA format."""
    via_data = {}
    
    for annotation in vehide_data.get("annotations", []):
        image_info = annotation.get("image", {})
        filename = image_info.get("file_name", "")
        
        if not filename:
            continue
        
        # Get image dimensions
        width = image_info.get("width", 0)
        height = image_info.get("height", 0)
        
        # Convert damage annotations
        regions = []
        for damage in annotation.get("damages", []):
            # Get polygon points
            segmentation = damage.get("segmentation", [])
            if not segmentation:
                continue
            
            # Convert polygon to VIA format
            if len(segmentation[0]) >= 6:  # At least 3 points (x,y pairs)
                points = segmentation[0]
                all_x = [points[i] for i in range(0, len(points), 2)]
                all_y = [points[i] for i in range(1, len(points), 2)]
                
                region = {
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": all_x,
                        "all_points_y": all_y
                    },
                    "region_attributes": {
                        "damage_type": normalize_class_names(damage.get("category", "")),
                        "severity": damage.get("severity", "medium"),
                        "confidence": damage.get("confidence", 1.0)
                    }
                }
                regions.append(region)
        
        # Create VIA annotation
        file_size = 0
        image_path = image_base_path / filename
        if image_path.exists():
            file_size = image_path.stat().st_size
        
        via_annotation = {
            "filename": filename,
            "size": file_size,
            "regions": regions,
            "file_attributes": {
                "dataset": "vehide",
                "split": "train",
                "has_damage": len(regions) > 0
            }
        }
        
        # Add to VIA data structure
        file_key = f"{filename}{file_size}"
        via_data[file_key] = via_annotation
    
    return via_data


def merge_annotations(
    vehide_file: Path,
    no_damage_file: Path,
    output_file: Path,
    vehide_images_dir: Path
) -> None:
    """Merge VEHiDe and no_damage annotations."""
    logger.info("Loading VEHiDe annotations")
    vehide_data = load_json_annotations(vehide_file)
    
    logger.info("Loading no_damage annotations")
    no_damage_data = load_json_annotations(no_damage_file)
    
    # Convert VEHiDe to VIA format if needed
    if "annotations" in vehide_data:
        logger.info("Converting VEHiDe format to VIA format")
        vehide_via = convert_vehide_to_via(vehide_data, vehide_images_dir)
    else:
        vehide_via = vehide_data
    
    # Merge annotations
    merged_data = {}
    merged_data.update(vehide_via)
    merged_data.update(no_damage_data)
    
    # Validate and clean up class names
    valid_classes = set()
    cleaned_data = {}
    
    for key, annotation in merged_data.items():
        cleaned_regions = []
        
        for region in annotation.get("regions", []):
            damage_type = region.get("region_attributes", {}).get("damage_type", "")
            normalized_class = normalize_class_names(damage_type)
            
            if normalized_class:
                valid_classes.add(normalized_class)
                
                # Update region with normalized class
                region["region_attributes"]["damage_type"] = normalized_class
                cleaned_regions.append(region)
        
        if cleaned_regions or annotation.get("file_attributes", {}).get("damage_class") == "no_damage":
            annotation["regions"] = cleaned_regions
            cleaned_data[key] = annotation
    
    # Save merged annotations
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(cleaned_data, f, indent=2)
    
    logger.info(f"Merged {len(cleaned_data)} annotations")
    logger.info(f"Found classes: {sorted(valid_classes)}")
    logger.info(f"Merged annotations saved to: {output_file}")
    
    # Save class statistics
    stats_file = output_file.parent / "class_statistics.json"
    class_counts = {}
    
    for annotation in cleaned_data.values():
        if annotation.get("file_attributes", {}).get("damage_class") == "no_damage":
            class_counts["no_damage"] = class_counts.get("no_damage", 0) + 1
        else:
            for region in annotation.get("regions", []):
                damage_type = region.get("region_attributes", {}).get("damage_type", "")
                if damage_type:
                    class_counts[damage_type] = class_counts.get(damage_type, 0) + 1
    
    stats = {
        "total_images": len(cleaned_data),
        "total_classes": len(valid_classes),
        "class_counts": class_counts,
        "valid_classes": sorted(valid_classes)
    }
    
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Class statistics saved to: {stats_file}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to merge annotations."""
    logger.info("Starting annotation merge process")
    
    # Paths
    vehide_file = Path(cfg.data.raw_data.vehide_annotations)
    no_damage_file = Path(cfg.data.processed_data.merged_annotations).parent / "no_damage_annotations.json"
    output_file = Path(cfg.data.processed_data.merged_annotations)
    vehide_images_dir = Path(cfg.data.raw_data.vehide_images_dir)
    
    # Check input files
    if not vehide_file.exists() and not no_damage_file.exists():
        logger.error("Neither VEHiDe nor no_damage annotations found")
        logger.info("Please run prepare_no_damage.py first or provide VEHiDe annotations")
        return
    
    # Merge annotations
    merge_annotations(
        vehide_file=vehide_file,
        no_damage_file=no_damage_file,
        output_file=output_file,
        vehide_images_dir=vehide_images_dir
    )
    
    logger.info("Annotation merge completed")


if __name__ == "__main__":
    main()