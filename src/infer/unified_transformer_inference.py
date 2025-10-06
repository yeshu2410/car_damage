"""
Inference script for Unified Transformer Model.
Performs end-to-end detection and damage classification.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import json

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.unified_transformer import UnifiedTransformerModel, create_unified_transformer


class UnifiedTransformerInference:
    """Inference pipeline for unified transformer model."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        conf_threshold: float = 0.5,
        img_size: int = 640
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
            conf_threshold: Confidence threshold for detections
            img_size: Image size for inference
        """
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Image transforms
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Label mappings (should be loaded from config in production)
        self.vehicle_parts = [
            'hood', 'front_bumper', 'rear_bumper', 'headlight', 'taillight',
            'door', 'fender', 'windshield', 'wheel', 'mirror'
        ]
        
        self.damage_locations = [
            'front_bumper', 'rear_bumper', 'hood', 'trunk',
            'door_front_left', 'door_front_right', 'door_rear_left', 'door_rear_right',
            'fender_front_left', 'fender_front_right', 'fender_rear_left', 'fender_rear_right',
            'windshield', 'rear_window', 'headlight', 'taillight',
            'mirror', 'wheel', 'roof', 'side_panel', 'other'
        ]
        
        self.damage_types = [
            'scratch', 'dent', 'crack', 'broken', 'shattered',
            'crushed', 'paint_damage', 'rust', 'missing', 'deformed', 'other'
        ]
    
    def _load_model(self, model_path: str) -> UnifiedTransformerModel:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model config from checkpoint
        config = checkpoint.get('config', {})
        
        # Create model
        model = create_unified_transformer(
            num_vehicle_parts=10,
            num_damage_locations=21,
            num_damage_types=11,
            model_size='base'
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        logger.info(f"Model loaded from: {model_path}")
        
        return model
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference."""
        return self.transform(image).unsqueeze(0)
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with detections and damage predictions
        """
        # Preprocess
        image_tensor = self.preprocess_image(image).to(self.device)
        
        # Forward pass
        predictions = self.model(image_tensor)
        
        # Post-process predictions
        results = self._post_process(predictions, image.size)
        
        return results
    
    def _post_process(
        self,
        predictions: Dict[str, torch.Tensor],
        image_size: tuple
    ) -> Dict:
        """
        Post-process model predictions.
        
        Args:
            predictions: Raw model predictions
            image_size: Original image size (width, height)
            
        Returns:
            Formatted detection results
        """
        # Get predictions
        bbox_pred = predictions['bbox_pred'][0]  # [Q, 4]
        class_logits = predictions['class_logits'][0]  # [Q, num_classes+1]
        location_logits = predictions['location_logits'][0]  # [Q, num_locations]
        severity_pred = predictions['severity_pred'][0]  # [Q, 1]
        type_logits = predictions['type_logits'][0]  # [Q, num_types]
        
        # Get class probabilities
        class_probs = F.softmax(class_logits, dim=-1)
        
        # Filter by confidence (excluding "no object" class)
        max_probs, max_classes = class_probs[:, :-1].max(dim=-1)
        valid_mask = max_probs > self.conf_threshold
        
        # Filter predictions
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return {
                'num_detections': 0,
                'detections': [],
                'image_size': image_size
            }
        
        # Get valid predictions
        valid_bboxes = bbox_pred[valid_indices]
        valid_classes = max_classes[valid_indices]
        valid_confidences = max_probs[valid_indices]
        
        # Get damage predictions
        location_probs = F.softmax(location_logits[valid_indices], dim=-1)
        location_preds = location_probs.argmax(dim=-1)
        location_confs = location_probs.max(dim=-1)[0]
        
        severity_preds = severity_pred[valid_indices].squeeze(-1)
        
        type_probs = F.softmax(type_logits[valid_indices], dim=-1)
        type_preds = type_probs.argmax(dim=-1)
        type_confs = type_probs.max(dim=-1)[0]
        
        # Convert to numpy
        valid_bboxes = valid_bboxes.cpu().numpy()
        valid_classes = valid_classes.cpu().numpy()
        valid_confidences = valid_confidences.cpu().numpy()
        location_preds = location_preds.cpu().numpy()
        location_confs = location_confs.cpu().numpy()
        severity_preds = severity_preds.cpu().numpy()
        type_preds = type_preds.cpu().numpy()
        type_confs = type_confs.cpu().numpy()
        
        # Format results
        detections = []
        img_w, img_h = image_size
        
        for i in range(len(valid_indices)):
            # Convert normalized bbox to pixel coordinates
            cx, cy, w, h = valid_bboxes[i]
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            
            detection = {
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1)
                },
                'vehicle_part': {
                    'class_id': int(valid_classes[i]),
                    'class_name': self.vehicle_parts[valid_classes[i]] if valid_classes[i] < len(self.vehicle_parts) else 'unknown',
                    'confidence': float(valid_confidences[i])
                },
                'damage_location': {
                    'location_id': int(location_preds[i]),
                    'location_name': self.damage_locations[location_preds[i]] if location_preds[i] < len(self.damage_locations) else 'unknown',
                    'confidence': float(location_confs[i])
                },
                'damage_severity': {
                    'value': float(severity_preds[i]),
                    'normalized': float(severity_preds[i] / 10.0),
                    'level': self._get_severity_level(severity_preds[i])
                },
                'damage_type': {
                    'type_id': int(type_preds[i]),
                    'type_name': self.damage_types[type_preds[i]] if type_preds[i] < len(self.damage_types) else 'unknown',
                    'confidence': float(type_confs[i])
                }
            }
            
            detections.append(detection)
        
        return {
            'num_detections': len(detections),
            'detections': detections,
            'image_size': image_size
        }
    
    def _get_severity_level(self, severity: float) -> str:
        """Convert severity value to level."""
        if severity < 3:
            return 'Minor'
        elif severity < 7:
            return 'Moderate'
        else:
            return 'Severe'
    
    def predict_batch(self, images: List[Image.Image]) -> List[Dict]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of detection results
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        
        return results
    
    def generate_report(self, results: Dict, image_path: Optional[str] = None) -> str:
        """
        Generate human-readable damage assessment report.
        
        Args:
            results: Detection results from predict()
            image_path: Optional path to image
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("UNIFIED TRANSFORMER - COLLISION DAMAGE ASSESSMENT REPORT")
        report.append("=" * 80)
        
        if image_path:
            report.append(f"Image: {image_path}")
        
        report.append(f"\nTotal Detections: {results['num_detections']}")
        report.append(f"Image Size: {results['image_size'][0]}x{results['image_size'][1]}")
        report.append("\n" + "-" * 80)
        
        if results['num_detections'] == 0:
            report.append("\nNo damage detected.")
        else:
            for i, det in enumerate(results['detections'], 1):
                report.append(f"\n🔍 Detection #{i}:")
                
                # Bounding box
                bbox = det['bbox']
                report.append(f"  📦 Bounding Box:")
                report.append(f"     Position: ({bbox['x1']:.1f}, {bbox['y1']:.1f}) → ({bbox['x2']:.1f}, {bbox['y2']:.1f})")
                report.append(f"     Size: {bbox['width']:.1f} × {bbox['height']:.1f}")
                
                # Vehicle part
                part = det['vehicle_part']
                report.append(f"  🚗 Vehicle Part: {part['class_name']}")
                report.append(f"     Confidence: {part['confidence']:.1%}")
                
                # Damage location
                location = det['damage_location']
                report.append(f"  📍 Damage Location: {location['location_name']}")
                report.append(f"     Confidence: {location['confidence']:.1%}")
                
                # Severity
                severity = det['damage_severity']
                report.append(f"  ⚠️  Damage Severity: {severity['value']:.1f}/10 ({severity['level']})")
                
                # Damage type
                dtype = det['damage_type']
                report.append(f"  🔨 Damage Type: {dtype['type_name']}")
                report.append(f"     Confidence: {dtype['confidence']:.1%}")
                
                report.append("-" * 80)
            
            # Overall statistics
            avg_severity = sum(d['damage_severity']['value'] for d in results['detections']) / results['num_detections']
            severe_count = sum(1 for d in results['detections'] if d['damage_severity']['value'] >= 7)
            moderate_count = sum(1 for d in results['detections'] if 3 <= d['damage_severity']['value'] < 7)
            minor_count = sum(1 for d in results['detections'] if d['damage_severity']['value'] < 3)
            
            report.append(f"\n📊 Overall Assessment:")
            report.append(f"   Average Severity: {avg_severity:.2f}/10")
            report.append(f"   Severity Distribution:")
            report.append(f"     • Severe (7-10): {severe_count}")
            report.append(f"     • Moderate (3-7): {moderate_count}")
            report.append(f"     • Minor (0-3): {minor_count}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def main():
    """CLI for inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Transformer Inference")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--report', action='store_true', help='Generate text report')
    
    args = parser.parse_args()
    
    # Initialize inference pipeline
    pipeline = UnifiedTransformerInference(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf
    )
    
    # Load image
    image = Image.open(args.image).convert('RGB')
    logger.info(f"Loaded image: {image.size}")
    
    # Run inference
    logger.info("Running inference...")
    results = pipeline.predict(image)
    
    # Print report
    if args.report or not args.output:
        report = pipeline.generate_report(results, args.image)
        print("\n" + report)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")
    
    # Print summary
    logger.info(f"Detected {results['num_detections']} damaged areas")


if __name__ == "__main__":
    main()
