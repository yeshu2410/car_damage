#!/usr/bin/env python3
"""
Inference script for the Unified Transformer Model.

Provides functionality to run inference on images and generate damage reports.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import numpy as np
    import cv2

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: Required packages not available. This script requires PyTorch, PIL, numpy, and opencv.")


class UnifiedTransformerInference:
    """
    Inference pipeline for the Unified Transformer model.

    Handles model loading, preprocessing, inference, and postprocessing.
    """

    def __init__(self,
                 model_path: str,
                 device: str = 'auto',
                 conf_threshold: float = 0.5,
                 num_queries: int = 100,
                 img_size: Tuple[int, int] = (640, 640)):
        """
        Initialize the inference pipeline.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            conf_threshold: Confidence threshold for predictions
            num_queries: Number of object queries in the model
            img_size: Input image size
        """
        if not TORCH_AVAILABLE:
            raise ImportError("Required packages not available")

        self.conf_threshold = conf_threshold
        self.num_queries = num_queries
        self.img_size = img_size

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model
        self.model = self._load_model(model_path)

        # Preprocessing transforms
        self.transform = self._create_transform()

        # Class mappings (would be loaded from config)
        self.vehicle_parts = [
            "front_bumper", "rear_bumper", "hood", "trunk",
            "door_front_left", "door_front_right", "door_rear_left", "door_rear_right",
            "fender_front_left", "fender_front_right", "fender_rear_left", "fender_rear_right",
            "other"
        ]

        self.damage_locations = [
            "front_bumper", "rear_bumper", "hood", "trunk",
            "door_front_left", "door_front_right", "door_rear_left", "door_rear_right",
            "fender_front_left", "fender_front_right", "fender_rear_left", "fender_rear_right",
            "windshield", "rear_window", "headlight", "taillight",
            "mirror", "wheel", "roof", "side_panel", "other"
        ]

        self.damage_types = [
            "scratch", "dent", "crack", "broken", "shattered",
            "crushed", "paint_damage", "rust", "missing", "deformed", "other"
        ]

    def _load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        from models.unified_transformer import create_unified_transformer

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract config from checkpoint
        config = checkpoint.get('config', {})

        # Create model
        model = create_unified_transformer(
            model_size=config.get('model_size', 'base'),
            num_vehicle_parts=config.get('num_vehicle_parts', 11),
            num_damage_locations=config.get('num_damage_locations', 21),
            num_damage_types=config.get('num_damage_types', 11)
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        return model

    def _create_transform(self):
        """Create image preprocessing transform."""
        try:
            import torchvision.transforms as T

            return T.Compose([
                T.Resize(self.img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except ImportError:
            # Fallback transform
            return lambda x: torch.tensor(np.array(x).transpose(2, 0, 1) / 255.0, dtype=torch.float32)

    def preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for model input.

        Args:
            image: PIL Image

        Returns:
            Tuple of (processed_tensor, original_size)
        """
        original_size = image.size

        # Apply transform
        if hasattr(self, 'transform') and self.transform:
            tensor = self.transform(image)
        else:
            # Manual preprocessing
            image = image.resize(self.img_size)
            tensor = torch.tensor(np.array(image).transpose(2, 0, 1) / 255.0, dtype=torch.float32)

        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)

        return tensor, original_size

    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Run inference on a single image.

        Args:
            image: PIL Image

        Returns:
            List of detected damage instances
        """
        # Preprocess
        tensor, original_size = self.preprocess_image(image)

        # Forward pass
        with torch.no_grad():
            predictions = self.model(tensor)

        # Postprocess predictions
        results = self._postprocess_predictions(predictions, original_size)

        return results

    def _postprocess_predictions(self, predictions: Dict[str, torch.Tensor],
                               original_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Postprocess model predictions into damage instances.

        Args:
            predictions: Raw model predictions
            original_size: Original image size (width, height)

        Returns:
            List of damage detection results
        """
        results = []

        # Extract predictions
        bbox_pred = predictions['bbox_pred'][0]  # [num_queries, 4]
        class_logits = predictions['class_logits'][0]  # [num_queries, num_classes + 1]
        location_logits = predictions['location_logits'][0]  # [num_queries, num_locations]
        severity_pred = predictions['severity_pred'][0]  # [num_queries, 1]
        type_logits = predictions['type_logits'][0]  # [num_queries, num_types]

        # Convert logits to probabilities
        class_probs = F.softmax(class_logits, dim=-1)
        location_probs = F.softmax(location_logits, dim=-1)
        type_probs = F.softmax(type_logits, dim=-1)

        # Get predictions
        class_conf, class_pred = class_probs.max(dim=-1)
        location_conf, location_pred = location_probs.max(dim=-1)
        type_conf, type_pred = type_probs.max(dim=-1)
        severity_values = severity_pred.squeeze(-1)

        # Scale bounding boxes back to original size
        orig_w, orig_h = original_size
        scale_x = orig_w / self.img_size[0]
        scale_y = orig_h / self.img_size[1]

        for i in range(self.num_queries):
            # Skip low confidence predictions and "no object" class
            if class_pred[i] == len(self.vehicle_parts):  # "no object" class
                continue

            if class_conf[i] < self.conf_threshold:
                continue

            # Get bounding box (convert from normalized to pixel coordinates)
            bbox = bbox_pred[i]  # [x, y, w, h] normalized
            x_center = bbox[0] * orig_w
            y_center = bbox[1] * orig_h
            width = bbox[2] * orig_w
            height = bbox[3] * orig_h

            # Convert to corner format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # Create result
            result = {
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'vehicle_part': {
                    'class': self.vehicle_parts[class_pred[i]],
                    'confidence': float(class_conf[i])
                },
                'damage_location': {
                    'class': self.damage_locations[location_pred[i]],
                    'confidence': float(location_conf[i])
                },
                'damage_severity': float(severity_values[i]),
                'damage_type': {
                    'class': self.damage_types[type_pred[i]],
                    'confidence': float(type_conf[i])
                },
                'confidence': float(class_conf[i])  # Overall confidence
            }

            results.append(result)

        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)

        return results

    def generate_report(self, results: List[Dict[str, Any]], image_path: str) -> str:
        """
        Generate a human-readable damage report.

        Args:
            results: Detection results
            image_path: Path to the analyzed image

        Returns:
            Formatted damage report
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("COLLISION DAMAGE ASSESSMENT REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Image: {Path(image_path).name}")
        report_lines.append(f"Analysis Date: {torch.tensor(0).new_tensor(0).to('cpu').numpy()}")
        report_lines.append(f"Detected Damages: {len(results)}")
        report_lines.append("")

        if not results:
            report_lines.append("No damages detected in the image.")
            return "\n".join(report_lines)

        # Group damages by location
        location_groups = {}
        for result in results:
            loc = result['damage_location']['class']
            if loc not in location_groups:
                location_groups[loc] = []
            location_groups[loc].append(result)

        # Generate detailed report
        for location, damages in location_groups.items():
            report_lines.append(f"Location: {location.upper()}")
            report_lines.append("-" * 40)

            for i, damage in enumerate(damages, 1):
                bbox = damage['bbox']
                severity = damage['damage_severity']

                report_lines.append(f"  Damage #{i}:")
                report_lines.append(f"    Vehicle Part: {damage['vehicle_part']['class']}")
                report_lines.append(f"    Damage Type: {damage['damage_type']['class']}")
                report_lines.append(f"    Severity: {severity + 1:.0f}/4")  # Convert 0-3 to 1-4 human scale
                report_lines.append(f"    Confidence: {damage['confidence']:.2%}")
                report_lines.append(f"    Bounding Box: ({bbox[0]:.1f}, {bbox[1]:.1f}) to ({bbox[2]:.1f}, {bbox[3]:.1f})")
                report_lines.append("")

        # Summary statistics
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 20)

        severities = [r['damage_severity'] + 1 for r in results]  # Convert to 1-4 scale
        avg_severity = sum(severities) / len(severities)
        max_severity = max(severities)

        report_lines.append(f"Total Damages: {len(results)}")
        report_lines.append(f"Average Severity: {avg_severity:.2f}/4.0")
        report_lines.append(f"Maximum Severity: {max_severity:.2f}/4.0")
        report_lines.append(f"Locations Affected: {len(location_groups)}")

        # Severity assessment
        if avg_severity < 3:
            assessment = "Minor damage - cosmetic repairs only"
        elif avg_severity < 6:
            assessment = "Moderate damage - structural inspection recommended"
        else:
            assessment = "Severe damage - professional assessment required"

        report_lines.append(f"Assessment: {assessment}")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    def visualize_results(self, image: Image.Image, results: List[Dict[str, Any]],
                         save_path: Optional[str] = None) -> Image.Image:
        """
        Visualize detection results on the image.

        Args:
            image: Original PIL Image
            results: Detection results
            save_path: Optional path to save the visualization

        Returns:
            Image with detections drawn
        """
        # Convert to numpy array
        img_array = np.array(image)

        # Draw detections
        for result in results:
            bbox = result['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # Choose color based on severity
            severity = result['damage_severity']
            if severity < 3:
                color = (0, 255, 0)  # Green for minor
            elif severity < 6:
                color = (0, 255, 255)  # Yellow for moderate
            else:
                color = (0, 0, 255)  # Red for severe

            # Draw bounding box
            cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{result['damage_type']['class']} ({severity:.1f})"
            cv2.putText(img_array, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert back to PIL
        result_image = Image.fromarray(img_array)

        if save_path:
            result_image.save(save_path)

        return result_image


def main():
    parser = argparse.ArgumentParser(description='Run Unified Transformer Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization')

    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("Required packages not available.")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize pipeline
    pipeline = UnifiedTransformerInference(
        model_path=args.model_path,
        conf_threshold=args.conf_threshold
    )

    # Load image
    image = Image.open(args.image_path)

    # Run inference
    results = pipeline.predict(image)

    # Generate report
    report = pipeline.generate_report(results, args.image_path)

    # Save report
    report_path = output_dir / f"{Path(args.image_path).stem}_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    # Print report
    print(report)

    # Visualize if requested
    if args.visualize:
        vis_image = pipeline.visualize_results(image, results)
        vis_path = output_dir / f"{Path(args.image_path).stem}_visualization.jpg"
        vis_image.save(vis_path)
        print(f"Visualization saved to: {vis_path}")

    print(f"Report saved to: {report_path}")
    print(f"Detected {len(results)} damages")


if __name__ == "__main__":
    main()