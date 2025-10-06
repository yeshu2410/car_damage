"""
Convert VehiDE annotations to VIA format compatible with the training pipeline.
"""

import json
from pathlib import Path
from typing import Dict

import hydra
from loguru import logger
from omegaconf import DictConfig


def normalize_vehide_class(class_name: str) -> str:
    """Map VehiDE class names to our standardized damage types."""
    class_mapping = {
        "mat_bo_phan": "dent",  # Dent/deformation
        "khac": "paint_scratch",  # Other damage (usually scratches)
        "lop_xi": "paint_scratch",  # Paint damage
        "san_xi": "rust",  # Rust
        "nut_nho": "crack",  # Small crack
        "nut_lon": "crack",  # Large crack
        "kinh_vo": "broken_glass",  # Broken glass
        "den_pha": "headlight_damage",  # Headlight damage
        "guong": "mirror_damage",  # Mirror damage
        "banh_xe": "wheel_damage",  # Wheel damage
        "can_truoc": "bumper_damage",  # Front bumper damage
        "can_sau": "bumper_damage",  # Rear bumper damage
    }
    
    return class_mapping.get(class_name, "dent")  # Default to dent for unknown classes


def convert_vehide_to_via(vehide_file: Path, output_file: Path) -> None:
    """Convert VehiDE annotations to VIA format."""
    logger.info(f"Loading VehiDE annotations from {vehide_file}")
    
    with open(vehide_file, "r") as f:
        vehide_data = json.load(f)
    
    converted_data = {}
    class_stats = {}
    
    for image_name, annotation in vehide_data.items():
        converted_regions = []
        
        for region in annotation.get("regions", []):
            # Extract damage class
            damage_class = region.get("class", "")
            normalized_class = normalize_vehide_class(damage_class)
            
            # Count classes
            class_stats[normalized_class] = class_stats.get(normalized_class, 0) + 1
            
            # Convert polygon format to VIA format
            converted_region = {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": region.get("all_x", []),
                    "all_points_y": region.get("all_y", [])
                },
                "region_attributes": {
                    "damage_type": normalized_class
                }
            }
            converted_regions.append(converted_region)
        
        # Create VIA format annotation
        converted_annotation = {
            "filename": image_name,
            "size": -1,  # Unknown size
            "regions": converted_regions,
            "file_attributes": {
                "dataset": "vehide"
            }
        }
        
        converted_data[image_name] = converted_annotation
    
    # Save converted annotations
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(converted_data, f, indent=2)
    
    logger.info(f"Converted {len(converted_data)} VehiDE annotations")
    logger.info(f"Class distribution: {class_stats}")
    logger.info(f"Converted annotations saved to: {output_file}")
    
    # Save class statistics
    stats_file = output_file.parent / "vehide_class_statistics.json"
    with open(stats_file, "w") as f:
        json.dump({
            "total_images": len(converted_data),
            "total_regions": sum(len(ann["regions"]) for ann in converted_data.values()),
            "class_counts": class_stats
        }, f, indent=2)
    
    logger.info(f"VehiDE statistics saved to: {stats_file}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to convert VehiDE annotations."""
    logger.info("Starting VehiDE annotation conversion")
    
    # Define paths
    vehide_file = Path(cfg.data.raw_data.vehide_annotations)
    output_file = Path(cfg.data.processed_data.get("vehide_via_annotations", 
                                                   "data/processed/vehide_via_annotations.json"))
    
    # Convert annotations
    convert_vehide_to_via(vehide_file, output_file)
    
    logger.info("VehiDE annotation conversion completed")


if __name__ == "__main__":
    main()