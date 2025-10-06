"""
Split merged annotations into train, validation, and test sets.
"""

import json
from pathlib import Path
from typing import Dict, List
import random

import hydra
from loguru import logger
from omegaconf import DictConfig


def split_annotations(
    merged_file: Path,
    output_dir: Path,
    split_ratios: Dict[str, float],
    seed: int = 42
) -> None:
    """Split merged annotations into train/val/test sets."""
    
    logger.info(f"Loading merged annotations from {merged_file}")
    with open(merged_file, "r") as f:
        merged_data = json.load(f)
    
    # Get all annotation keys
    all_keys = list(merged_data.keys())
    total_count = len(all_keys)
    
    logger.info(f"Total annotations: {total_count}")
    
    # Set seed for reproducible splits
    random.seed(seed)
    random.shuffle(all_keys)
    
    # Calculate split sizes
    train_size = int(total_count * split_ratios["train"])
    val_size = int(total_count * split_ratios["val"])
    test_size = total_count - train_size - val_size
    
    logger.info(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Split the keys
    train_keys = all_keys[:train_size]
    val_keys = all_keys[train_size:train_size + val_size]
    test_keys = all_keys[train_size + val_size:]
    
    # Create split datasets
    splits = {
        "train": train_keys,
        "val": val_keys,
        "test": test_keys
    }
    
    # Save each split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, keys in splits.items():
        split_data = {}
        for key in keys:
            annotation = merged_data[key].copy()
            # Update the split attribute in file_attributes
            if "file_attributes" not in annotation:
                annotation["file_attributes"] = {}
            annotation["file_attributes"]["split"] = split_name
            split_data[key] = annotation
        
        split_file = output_dir / f"{split_name}_annotations.json"
        
        with open(split_file, "w") as f:
            json.dump(split_data, f, indent=2)
        
        logger.info(f"Saved {len(split_data)} {split_name} annotations to {split_file}")
        
        # Log class distribution for this split
        class_counts = {}
        for annotation in split_data.values():
            for region in annotation.get("regions", []):
                damage_type = region.get("region_attributes", {}).get("damage_type", "")
                if damage_type:
                    class_counts[damage_type] = class_counts.get(damage_type, 0) + 1
            
            # Check for no_damage class from file attributes
            if annotation.get("file_attributes", {}).get("damage_class") == "no_damage":
                class_counts["no_damage"] = class_counts.get("no_damage", 0) + 1
        
        logger.info(f"{split_name.title()} class distribution: {class_counts}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to split annotations."""
    logger.info("Starting annotation splitting")
    
    # Define paths
    merged_file = Path(cfg.data.processed_data.merged_annotations)
    output_dir = Path("data/processed")
    
    # Split annotations
    split_annotations(
        merged_file=merged_file,
        output_dir=output_dir,
        split_ratios=cfg.data.split_ratios,
        seed=cfg.data.split_seed
    )
    
    logger.info("Annotation splitting completed")


if __name__ == "__main__":
    main()