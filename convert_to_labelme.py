#!/usr/bin/env python3
"""
Convert YOLO dataset to LabelMe format for additional annotation.
This allows adding 3 new features: damage_location, damage_severity, damage_type
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import click
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)


class YOLOToLabelMeConverter:
    """Convert YOLO format annotations to LabelMe JSON format."""
    
    def __init__(self, images_dir: Path, labels_dir: Path, output_dir: Path, class_names: List[str]):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        
        # Create output directories
        self.output_images_dir = self.output_dir / "images"
        self.output_annotations_dir = self.output_dir / "annotations"
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_annotations_dir.mkdir(parents=True, exist_ok=True)
    
    def yolo_to_labelme_bbox(self, yolo_bbox: List[float], img_width: int, img_height: int) -> List[List[float]]:
        """
        Convert YOLO bbox (x_center, y_center, width, height) normalized
        to LabelMe format (x_min, y_min), (x_max, y_max) pixel coordinates.
        """
        x_center, y_center, width, height = yolo_bbox
        
        # Convert from normalized to pixel coordinates
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        # Calculate corners
        x_min = x_center_px - width_px / 2
        y_min = y_center_px - height_px / 2
        x_max = x_center_px + width_px / 2
        y_max = y_center_px + height_px / 2
        
        # Return as polygon points (rectangle)
        return [
            [x_min, y_min],  # Top-left
            [x_max, y_min],  # Top-right
            [x_max, y_max],  # Bottom-right
            [x_min, y_max]   # Bottom-left
        ]
    
    def get_image_dimensions(self, image_path: Path) -> tuple:
        """Get image width and height."""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size  # Returns (width, height)
        except Exception as e:
            logger.error(f"Failed to get dimensions for {image_path}: {e}")
            return (640, 640)  # Default fallback
    
    def convert_single_image(self, image_path: Path) -> Optional[Dict]:
        """Convert a single image and its YOLO annotations to LabelMe format."""
        
        # Find corresponding label file
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        
        # Get image dimensions
        img_width, img_height = self.get_image_dimensions(image_path)
        
        # Initialize LabelMe JSON structure
        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": image_path.name,
            "imageData": None,  # We don't embed image data
            "imageHeight": img_height,
            "imageWidth": img_width
        }
        
        # Parse YOLO annotations if they exist
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    
                    # Convert bbox to polygon points
                    points = self.yolo_to_labelme_bbox(bbox, img_width, img_height)
                    
                    # Get class name
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    # Create shape entry with additional fields for annotation
                    shape = {
                        "label": class_name,
                        "points": points,
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {},
                        "attributes": {
                            "damage_location": "",  # To be annotated
                            "damage_severity": "",  # To be annotated (e.g., "minor", "moderate", "severe")
                            "damage_type": ""       # To be annotated (e.g., "scratch", "dent", "broken")
                        }
                    }
                    
                    labelme_data["shapes"].append(shape)
        
        return labelme_data
    
    def convert_all(self):
        """Convert all images and annotations."""
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(f"*{ext}"))
            image_files.extend(self.images_dir.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} images to convert")
        
        if len(image_files) == 0:
            logger.warning(f"No images found in {self.images_dir}")
            return
        
        converted_count = 0
        
        for image_path in image_files:
            try:
                # Convert to LabelMe format
                labelme_data = self.convert_single_image(image_path)
                
                if labelme_data is None:
                    continue
                
                # Copy image to output directory
                output_image_path = self.output_images_dir / image_path.name
                shutil.copy2(image_path, output_image_path)
                
                # Save LabelMe JSON
                json_path = self.output_annotations_dir / f"{image_path.stem}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(labelme_data, f, indent=2, ensure_ascii=False)
                
                converted_count += 1
                
                if converted_count % 100 == 0:
                    logger.info(f"Converted {converted_count}/{len(image_files)} images...")
                
            except Exception as e:
                logger.error(f"Failed to convert {image_path.name}: {e}")
                continue
        
        logger.info(f"✅ Conversion complete! Converted {converted_count}/{len(image_files)} images")
        logger.info(f"📁 Images saved to: {self.output_images_dir}")
        logger.info(f"📝 Annotations saved to: {self.output_annotations_dir}")
        
        # Create instructions file
        self._create_instructions_file()
    
    def _create_instructions_file(self):
        """Create an instruction file for annotators."""
        instructions = """# LabelMe Annotation Instructions

## Setup
1. Install LabelMe: `pip install labelme`
2. Open LabelMe: `labelme`
3. File > Open Dir > Select the 'images' folder

## Annotation Task
For each bounding box, you need to annotate 3 attributes:

### 1. damage_location
Specify where the damage is located on the vehicle:
- front_bumper
- rear_bumper
- hood
- trunk
- door_front_left
- door_front_right
- door_rear_left
- door_rear_right
- fender_front_left
- fender_front_right
- fender_rear_left
- fender_rear_right
- windshield
- rear_window
- headlight
- taillight
- mirror
- wheel
- roof
- other

### 2. damage_severity
Rate the severity of the damage:
- minor: Small scratches, paint chips
- moderate: Dents, significant scratches, broken lights
- severe: Major structural damage, needs replacement
- total: Complete destruction, safety hazard

### 3. damage_type
Type of damage:
- scratch: Surface level scratches
- dent: Indentations without breaking
- crack: Cracks in glass or plastic
- broken: Broken parts (lights, mirrors, etc.)
- shattered: Shattered glass
- crushed: Severely deformed/crushed metal
- paint_damage: Paint chipping or peeling
- rust: Rust damage
- missing: Missing parts
- other

## Workflow
1. Click on existing bounding box to edit
2. In the right panel, find "Attributes" section
3. Fill in the three fields:
   - damage_location: (select from list above)
   - damage_severity: (minor/moderate/severe/total)
   - damage_type: (select from list above)
4. Save the annotation (Ctrl+S)
5. Move to next image (D key)

## Tips
- Use keyboard shortcuts for faster annotation
- You can modify existing bounding boxes if needed
- If you see multiple types of damage in one box, use the most prominent one
- Save frequently!

## Output
The annotations will be saved as JSON files in the 'annotations' folder.
"""
        
        instructions_path = self.output_dir / "ANNOTATION_INSTRUCTIONS.md"
        with open(instructions_path, 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        logger.info(f"📋 Instructions saved to: {instructions_path}")


@click.command()
@click.option('--images-dir', '-i', 
              default='data/processed/yolo/images',
              help='Directory containing images')
@click.option('--labels-dir', '-l',
              default='data/processed/yolo/labels',
              help='Directory containing YOLO label files')
@click.option('--output-dir', '-o',
              default='data/labelme',
              help='Output directory for LabelMe format')
@click.option('--class-names-file', '-c',
              default='configs/data.yaml',
              help='YAML file containing class names')
def main(images_dir: str, labels_dir: str, output_dir: str, class_names_file: str):
    """
    Convert YOLO dataset to LabelMe format for additional annotation.
    
    This tool converts YOLO format annotations to LabelMe JSON format,
    allowing you to annotate additional features:
    - damage_location
    - damage_severity  
    - damage_type
    """
    
    logger.info("Starting YOLO to LabelMe conversion...")
    
    try:
        # Load class names
        class_names = []
        class_names_path = Path(class_names_file)
        
        if class_names_path.exists():
            import yaml
            with open(class_names_path, 'r') as f:
                data = yaml.safe_load(f)
                class_names = data.get('names', [])
            logger.info(f"Loaded {len(class_names)} class names from {class_names_file}")
        else:
            logger.warning(f"Class names file not found: {class_names_file}")
            logger.warning("Using default class indices...")
            class_names = [f"class_{i}" for i in range(100)]  # Fallback
        
        # Initialize converter
        converter = YOLOToLabelMeConverter(
            images_dir=Path(images_dir),
            labels_dir=Path(labels_dir),
            output_dir=Path(output_dir),
            class_names=class_names
        )
        
        # Run conversion
        converter.convert_all()
        
        logger.info("")
        logger.info("="*60)
        logger.info("Next Steps:")
        logger.info("1. Install LabelMe: pip install labelme")
        logger.info("2. Run LabelMe: labelme")
        logger.info("3. Open the 'images' directory in LabelMe")
        logger.info("4. Edit each bounding box to add the 3 attributes")
        logger.info("5. Read ANNOTATION_INSTRUCTIONS.md for details")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
