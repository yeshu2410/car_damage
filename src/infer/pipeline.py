"""
Inference pipeline for collision parts prediction.
Loads trained models, applies thresholds, and fuses predictions.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO

from omegaconf import OmegaConf, DictConfig
from loguru import logger

from ..models.resnet_endtoend import DamageNet
from ..data.dataset import get_transforms


class InferencePipeline:
    """Main inference pipeline that combines ResNet and YOLO predictions."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the inference pipeline."""
        self.cfg = OmegaConf.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model paths
        self.models_dir = Path(self.cfg.paths.models_dir)
        self.resnet_model_path = self.models_dir / "resnet" / "best_model.pth"
        self.yolo_model_path = self.models_dir / "yolo" / "best.pt"
        
        # Load thresholds and mappings
        self.thresholds = self._load_thresholds()
        self.correction_codes = self._load_correction_codes()
        
        # Initialize models
        self.resnet_model = None
        self.yolo_model = None
        self.class_names = self.cfg.data.classes  # Set class names from config
        self.transforms = None
        
        # Performance tracking
        self.model_versions = {}
        self.inference_stats = {
            "total_inferences": 0,
            "avg_resnet_time": 0.0,
            "avg_yolo_time": 0.0,
            "avg_fusion_time": 0.0,
            "avg_total_time": 0.0
        }
        
        logger.info(f"Inference pipeline initialized on device: {self.device}")
    
    def _load_thresholds(self) -> Dict[str, float]:
        """Load optimized thresholds from file."""
        threshold_file = Path(self.cfg.paths.output_dir) / "evaluation" / "optimized_thresholds.json"
        
        if threshold_file.exists():
            with open(threshold_file, "r") as f:
                thresholds_data = json.load(f)
                return thresholds_data.get("optimized_thresholds", {})
        
        # Default thresholds if optimization file not found
        logger.warning("Optimized thresholds not found, using default values")
        return {class_name: 0.5 for class_name in self.cfg.data.classes}
    
    def _load_correction_codes(self) -> Dict:
        """Load correction codes mapping."""
        codes_file = Path("configs/correction_codes.yaml")
        
        if codes_file.exists():
            return OmegaConf.load(codes_file)
        
        logger.warning("Correction codes file not found")
        return {}
    
    def load_models(self):
        """Load both ResNet and YOLO models."""
        # Check for demo mode
        demo_mode = getattr(self.cfg, 'demo_mode', False)
        
        if demo_mode:
            logger.info("Running in demo mode - using mock models")
            self.resnet_model = "mock_resnet_model"
            self.yolo_model = "mock_yolo_model"
            self.model_versions = {"resnet": "demo_v1", "yolo": "demo_v1"}
            
            # Set up transforms for demo mode
            self.transforms = get_transforms(
                image_size=self.cfg.data.image_size,
                mode='test'
            )
            return
        
        # Load ResNet model
        if self.resnet_model_path.exists():
            logger.info(f"Loading ResNet model from {self.resnet_model_path}")
            
            # Initialize model architecture
            self.resnet_model = DamageNet(
                num_classes=len(self.cfg.data.classes),
                model_name=self.cfg.model.backbone,
                pretrained=False
            ).to(self.device)
            
            # Load trained weights
            checkpoint = torch.load(self.resnet_model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.resnet_model.load_state_dict(checkpoint['model_state_dict'])
                self.model_versions['resnet'] = checkpoint.get('epoch', 'unknown')
            else:
                self.resnet_model.load_state_dict(checkpoint)
                self.model_versions['resnet'] = 'unknown'
            
            self.resnet_model.eval()
            logger.info("ResNet model loaded successfully")
        else:
            logger.error(f"ResNet model not found at {self.resnet_model_path}")
            raise FileNotFoundError(f"ResNet model not found at {self.resnet_model_path}")
        
        # Load YOLO model
        if self.yolo_model_path.exists():
            logger.info(f"Loading YOLO model from {self.yolo_model_path}")
            self.yolo_model = YOLO(str(self.yolo_model_path))
            
            # Extract version info
            self.model_versions['yolo'] = getattr(self.yolo_model.model, 'epoch', 'unknown')
            logger.info("YOLO model loaded successfully")
        else:
            logger.error(f"YOLO model not found at {self.yolo_model_path}")
            raise FileNotFoundError(f"YOLO model not found at {self.yolo_model_path}")
        
        # Set up transforms
        self.transforms = get_transforms(
            image_size=self.cfg.data.image_size,
            mode='test'
        )
        
        # Get class names
        self.class_names = self.cfg.data.classes
        
        logger.info("All models loaded and ready for inference")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for ResNet inference."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        tensor = self.transforms(image)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    def predict_resnet(self, image_tensor: torch.Tensor) -> Tuple[Dict[str, float], float]:
        """Run ResNet inference and return class probabilities."""
        start_time = time.time()
        
        with torch.no_grad():
            logits = self.resnet_model(image_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        inference_time = time.time() - start_time
        
        # Convert to class predictions with thresholds
        predictions = {}
        for i, class_name in enumerate(self.class_names):
            prob = float(probabilities[i])
            threshold = self.thresholds.get(class_name, 0.5)
            predictions[class_name] = {
                'probability': prob,
                'predicted': prob >= threshold,
                'threshold': threshold
            }
        
        return predictions, inference_time
    
    def predict_yolo(self, image: Image.Image) -> Tuple[Dict[str, float], float]:
        """Run YOLO inference and return detection results."""
        start_time = time.time()
        
        # Run YOLO inference
        results = self.yolo_model(image, verbose=False)
        result = results[0]
        
        inference_time = time.time() - start_time
        
        # Process detections
        predictions = {}
        
        if result.boxes is not None and len(result.boxes) > 0:
            # Get detection results
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Aggregate by class (max confidence rule)
            class_max_conf = {}
            for class_id, conf in zip(class_ids, confidences):
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    class_max_conf[class_name] = max(
                        class_max_conf.get(class_name, 0.0), 
                        float(conf)
                    )
            
            # Create predictions for all classes
            for class_name in self.class_names:
                confidence = class_max_conf.get(class_name, 0.0)
                threshold = 0.5  # YOLO default threshold
                
                predictions[class_name] = {
                    'confidence': confidence,
                    'predicted': confidence >= threshold,
                    'threshold': threshold
                }
        else:
            # No detections found
            for class_name in self.class_names:
                predictions[class_name] = {
                    'confidence': 0.0,
                    'predicted': False,
                    'threshold': 0.5
                }
        
        return predictions, inference_time
    
    def fuse_predictions(self, resnet_preds: Dict, yolo_preds: Dict) -> Tuple[Dict, float]:
        """Fuse ResNet and YOLO predictions using max rule."""
        start_time = time.time()
        
        fused_predictions = {}
        
        for class_name in self.class_names:
            resnet_data = resnet_preds.get(class_name, {})
            yolo_data = yolo_preds.get(class_name, {})
            
            # Extract scores
            resnet_score = resnet_data.get('probability', 0.0)
            yolo_score = yolo_data.get('confidence', 0.0)
            
            # Max rule fusion
            fused_score = max(resnet_score, yolo_score)
            
            # Determine final threshold (use ResNet threshold as primary)
            final_threshold = resnet_data.get('threshold', 0.5)
            
            fused_predictions[class_name] = {
                'fused_score': fused_score,
                'resnet_score': resnet_score,
                'yolo_score': yolo_score,
                'predicted': fused_score >= final_threshold,
                'threshold': final_threshold,
                'fusion_method': 'max_rule'
            }
        
        fusion_time = time.time() - start_time
        return fused_predictions, fusion_time
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Run complete inference pipeline on an image."""
        # Check for demo mode
        demo_mode = getattr(self.cfg, 'demo_mode', False)
        
        if demo_mode:
            return self._demo_predict(image)
        
        if self.resnet_model is None or self.yolo_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        total_start_time = time.time()
        
        # Preprocess image for ResNet
        image_tensor = self.preprocess_image(image)
        
        # Run ResNet inference
        resnet_preds, resnet_time = self.predict_resnet(image_tensor)
        
        # Run YOLO inference
        yolo_preds, yolo_time = self.predict_yolo(image)
        
        # Fuse predictions
        fused_preds, fusion_time = self.fuse_predictions(resnet_preds, yolo_preds)
        
        total_time = time.time() - total_start_time
        
        # Update statistics
        self._update_stats(resnet_time, yolo_time, fusion_time, total_time)
        
        # Prepare final results
        result = {
            'predictions': fused_preds,
            'predicted_classes': [
                class_name for class_name, data in fused_preds.items() 
                if data['predicted']
            ],
            'model_versions': self.model_versions.copy(),
            'inference_times': {
                'resnet_time': resnet_time,
                'yolo_time': yolo_time,
                'fusion_time': fusion_time,
                'total_time': total_time
            },
            'image_info': {
                'size': image.size,
                'mode': image.mode
            },
            'pipeline_version': '1.0.0',
            'timestamp': time.time()
        }
        
        return result
    
    def _update_stats(self, resnet_time: float, yolo_time: float, 
                     fusion_time: float, total_time: float):
        """Update inference statistics."""
        n = self.inference_stats['total_inferences']
        
        # Running average update
        self.inference_stats['avg_resnet_time'] = (
            (self.inference_stats['avg_resnet_time'] * n + resnet_time) / (n + 1)
        )
        self.inference_stats['avg_yolo_time'] = (
            (self.inference_stats['avg_yolo_time'] * n + yolo_time) / (n + 1)
        )
        self.inference_stats['avg_fusion_time'] = (
            (self.inference_stats['avg_fusion_time'] * n + fusion_time) / (n + 1)
        )
        self.inference_stats['avg_total_time'] = (
            (self.inference_stats['avg_total_time'] * n + total_time) / (n + 1)
        )
        
        self.inference_stats['total_inferences'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        stats = self.inference_stats.copy()
        stats['models_loaded'] = {
            'resnet': self.resnet_model is not None,
            'yolo': self.yolo_model is not None
        }
        stats['device'] = str(self.device)
        stats['model_versions'] = self.model_versions.copy()
        
        return stats
    
    def warmup(self, num_warmup: int = 3):
        """Warm up the models with dummy inferences."""
        logger.info(f"Warming up models with {num_warmup} dummy inferences...")
        
        # Create dummy image
        dummy_image = Image.new('RGB', (224, 224), color='red')
        
        for i in range(num_warmup):
            try:
                self.predict(dummy_image)
                logger.info(f"Warmup {i+1}/{num_warmup} completed")
            except Exception as e:
                logger.error(f"Warmup {i+1} failed: {e}")
        
        logger.info("Model warmup completed")
    
    def _demo_predict(self, image: Image.Image) -> Dict[str, Any]:
        """Demo prediction with mock results."""
        import random
        
        # Simulate realistic processing times
        resnet_time = random.uniform(0.02, 0.08)
        yolo_time = random.uniform(0.01, 0.05)
        fusion_time = random.uniform(0.001, 0.005)
        total_time = resnet_time + yolo_time + fusion_time
        
        # Generate mock predictions
        predictions = {}
        predicted_classes = []
        
        for i, class_name in enumerate(self.class_names):
            # Create realistic but random scores
            resnet_score = random.uniform(0.1, 0.9)
            yolo_score = random.uniform(0.1, 0.9)
            fused_score = max(resnet_score, yolo_score)  # Simple max fusion for demo
            threshold = self.thresholds.get(class_name, 0.5)
            is_predicted = fused_score >= threshold
            
            if is_predicted:
                predicted_classes.append(class_name)
            
            predictions[class_name] = {
                "resnet_score": round(resnet_score, 3),
                "yolo_score": round(yolo_score, 3),
                "fused_score": round(fused_score, 3),
                "threshold": threshold,
                "predicted": is_predicted,
                "fusion_method": "max_rule"
            }
        
        return {
            "predictions": predictions,
            "predicted_classes": predicted_classes,
            "model_versions": self.model_versions,
            "inference_times": {
                "resnet_time": round(resnet_time, 4),
                "yolo_time": round(yolo_time, 4),
                "fusion_time": round(fusion_time, 4),
                "total_time": round(total_time, 4)
            },
            "image_info": {
                "size": image.size,
                "mode": image.mode
            },
            "pipeline_version": "1.0.0-demo",
            "timestamp": time.time()
        }


def create_pipeline(config_path: str = "configs/config.yaml") -> InferencePipeline:
    """Factory function to create and initialize inference pipeline."""
    pipeline = InferencePipeline(config_path)
    pipeline.load_models()
    pipeline.warmup()
    return pipeline


if __name__ == "__main__":
    # Test the pipeline
    try:
        pipeline = create_pipeline()
        
        # Test with a dummy image
        test_image = Image.new('RGB', (224, 224), color='blue')
        result = pipeline.predict(test_image)
        
        print("Inference Pipeline Test Results:")
        print(f"Predicted classes: {result['predicted_classes']}")
        print(f"Total inference time: {result['inference_times']['total_time']:.4f}s")
        print(f"Model versions: {result['model_versions']}")
        
        # Print stats
        stats = pipeline.get_stats()
        print(f"\nPipeline Stats:")
        print(f"Total inferences: {stats['total_inferences']}")
        print(f"Average total time: {stats['avg_total_time']:.4f}s")
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        raise