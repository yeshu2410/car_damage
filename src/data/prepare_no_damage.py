"""
Prepare no_damage samples from Stanford Cars dataset.
Samples images and creates VIA JSON annotations for the no_damage class.
"""

import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

import hydra
from loguru import logger
from omegaconf import DictConfig
from PIL import Image


def load_stanford_cars_metadata(cars_dir: Path) -> List[Dict]:
    """Load Stanford Cars dataset metadata."""
    cars_meta_path = cars_dir / "cars_meta.mat"
    cars_annos_path = cars_dir / "cars_train_annos.mat"
    
    if not cars_meta_path.exists() or not cars_annos_path.exists():
        logger.warning("Stanford Cars metadata files not found. Using image directory.")
        
        # Fallback: scan image directory
        image_files = []
        for img_ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_files.extend(list(cars_dir.glob(f"**/{img_ext}")))
        
        return [{"fname": str(img.relative_to(cars_dir))} for img in image_files]
    
    try:
        import scipy.io
        meta = scipy.io.loadmat(str(cars_meta_path))
        annos = scipy.io.loadmat(str(cars_annos_path))
        
        cars_data = []
        for i, fname in enumerate(annos["annotations"][0]):
            cars_data.append({
                "fname": fname[5][0],  # filename
                "class": fname[4][0][0],  # class_id
                "bbox": [fname[0][0][0], fname[1][0][0], fname[2][0][0], fname[3][0][0]]  # bbox
            })
        
        return cars_data
    except ImportError:
        logger.warning("scipy not available. Scanning directory for images.")
        image_files = []
        for img_ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_files.extend(list(cars_dir.glob(f"**/{img_ext}")))
        
        return [{"fname": str(img.relative_to(cars_dir))} for img in image_files]


def create_via_annotation(
    image_path: Path, 
    image_size: tuple, 
    class_name: str = "no_damage"
) -> Dict:
    """Create VIA format annotation for a single image."""
    filename = image_path.name
    file_size = image_path.stat().st_size
    
    # For no_damage, create a full image bounding box
    width, height = image_size
    
    return {
        "filename": filename,
        "size": file_size,
        "regions": [
            {
                "shape_attributes": {
                    "name": "rect",
                    "x": 0,
                    "y": 0,
                    "width": width,
                    "height": height
                },
                "region_attributes": {
                    "damage_type": class_name,
                    "severity": "none",
                    "confidence": 1.0
                }
            }
        ],
        "file_attributes": {
            "dataset": "stanford_cars",
            "split": "train",
            "damage_class": class_name
        }
    }


def sample_and_annotate_cars(
    cars_dir: Path,
    output_dir: Path,
    output_annotations: Path,
    num_samples: int,
    seed: int = 42
) -> None:
    """Sample cars images and create VIA annotations."""
    random.seed(seed)
    
    logger.info(f"Loading Stanford Cars dataset from {cars_dir}")
    cars_data = load_stanford_cars_metadata(cars_dir)
    
    if len(cars_data) < num_samples:
        logger.warning(f"Only {len(cars_data)} images available, requested {num_samples}")
        num_samples = len(cars_data)
    
    # Sample random images
    sampled_cars = random.sample(cars_data, num_samples)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    via_data = {}
    successful_samples = 0
    
    for i, car_data in enumerate(sampled_cars):
        try:
            # Find the image file
            img_path = cars_dir / car_data["fname"]
            if not img_path.exists():
                # Try alternative extensions
                img_name = Path(car_data["fname"])
                for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                    alt_path = cars_dir / f"{img_name.stem}{ext}"
                    if alt_path.exists():
                        img_path = alt_path
                        break
                
                if not img_path.exists():
                    logger.warning(f"Image not found: {car_data['fname']}")
                    continue
            
            # Load image to get dimensions
            with Image.open(img_path) as img:
                width, height = img.size
                
                # Copy image to output directory
                output_img_path = output_dir / f"no_damage_{i:06d}.jpg"
                
                # Convert to RGB if necessary and save as JPEG
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(output_img_path, "JPEG", quality=95)
            
            # Create VIA annotation
            via_annotation = create_via_annotation(
                output_img_path, 
                (width, height), 
                "no_damage"
            )
            
            # Add to VIA data structure
            file_key = f"{output_img_path.name}{output_img_path.stat().st_size}"
            via_data[file_key] = via_annotation
            
            successful_samples += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(sampled_cars)} images")
                
        except Exception as e:
            logger.error(f"Error processing {car_data['fname']}: {e}")
            continue
    
    # Save VIA annotations
    output_annotations.parent.mkdir(parents=True, exist_ok=True)
    with open(output_annotations, "w") as f:
        json.dump(via_data, f, indent=2)
    
    logger.info(f"Successfully processed {successful_samples} no_damage samples")
    logger.info(f"Images saved to: {output_dir}")
    logger.info(f"Annotations saved to: {output_annotations}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to prepare no_damage dataset."""
    logger.info("Starting no_damage dataset preparation")
    
    # Paths
    cars_dir = Path(cfg.data.raw_data.stanford_cars_dir)
    output_dir = Path(cfg.data.raw_data.stanford_cars_dir).parent / "no_damage_images"
    output_annotations = Path(cfg.data.processed_data.merged_annotations).parent / "no_damage_annotations.json"
    
    # Check if Stanford Cars directory exists
    if not cars_dir.exists():
        logger.error(f"Stanford Cars directory not found: {cars_dir}")
        logger.info("Please download Stanford Cars dataset and update the config path")
        return
    
    # Sample and annotate
    sample_and_annotate_cars(
        cars_dir=cars_dir,
        output_dir=output_dir,
        output_annotations=output_annotations,
        num_samples=cfg.data.no_damage_samples,
        seed=cfg.data.split_seed
    )
    
    logger.info("No damage dataset preparation completed")


if __name__ == "__main__":
    main()