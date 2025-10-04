"""
Command Line Interface for Collision Parts Prediction System.
Provides commands for data preparation, training, evaluation, and inference.
"""

import click
import sys
from pathlib import Path
import time
from typing import Optional

from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    Collision Parts Prediction CLI
    
    AI-powered vehicle collision damage detection and parts prediction system.
    """
    pass


@cli.command()
@click.option('--config', '-c', default='configs/config.yaml', help='Config file path')
@click.option('--data-dir', help='Override data directory')
@click.option('--output-dir', help='Override output directory')
def prepare_data(config: str, data_dir: Optional[str], output_dir: Optional[str]):
    """Prepare training data from VEHiDe dataset."""
    logger.info("Starting data preparation...")
    
    try:
        # Import here to avoid loading heavy dependencies at CLI startup
        from omegaconf import OmegaConf
        
        # Load config
        cfg = OmegaConf.load(config)
        
        # Override paths if provided
        if data_dir:
            cfg.paths.data_dir = data_dir
        if output_dir:
            cfg.paths.output_dir = output_dir
        
        # Step 1: Prepare no-damage annotations
        logger.info("Step 1/3: Preparing no-damage annotations...")
        from ..data.prepare_no_damage import main as prepare_no_damage
        prepare_no_damage(cfg)
        
        # Step 2: Merge annotations
        logger.info("Step 2/3: Merging damage and no-damage annotations...")
        from ..data.merge_annotations import main as merge_annotations
        merge_annotations(cfg)
        
        # Step 3: Convert to YOLO format
        logger.info("Step 3/3: Converting to YOLO format...")
        from ..data.to_yolo import main as convert_to_yolo
        convert_to_yolo(cfg)
        
        logger.info("✅ Data preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', default='configs/train_resnet.yaml', help='Config file path')
@click.option('--epochs', type=int, help='Override number of epochs')
@click.option('--batch-size', type=int, help='Override batch size')
@click.option('--lr', type=float, help='Override learning rate')
@click.option('--resume', is_flag=True, help='Resume training from checkpoint')
def train_resnet(config: str, epochs: Optional[int], batch_size: Optional[int], 
                lr: Optional[float], resume: bool):
    """Train ResNet classification model."""
    logger.info("Starting ResNet training...")
    
    try:
        # Import here to avoid loading heavy dependencies at CLI startup
        from omegaconf import OmegaConf
        import hydra
        from hydra import initialize, compose
        
        # Load config
        with initialize(version_base=None, config_path="../../configs"):
            cfg = compose(config_name="train_resnet")
        
        # Override parameters if provided
        if epochs:
            cfg.train.epochs = epochs
        if batch_size:
            cfg.train.batch_size = batch_size
        if lr:
            cfg.train.learning_rate = lr
        if resume:
            cfg.train.resume = True
        
        # Start training
        from ..training.train_resnet import ModelTrainer
        trainer = ModelTrainer(cfg)
        trainer.train()
        
        logger.info("✅ ResNet training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ ResNet training failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', default='configs/train_yolo.yaml', help='Config file path')
@click.option('--epochs', type=int, help='Override number of epochs')
@click.option('--batch-size', type=int, help='Override batch size')
@click.option('--device', help='Override device (cpu, cuda, mps)')
def train_yolo(config: str, epochs: Optional[int], batch_size: Optional[int], 
              device: Optional[str]):
    """Train YOLO detection model."""
    logger.info("Starting YOLO training...")
    
    try:
        # Import here to avoid loading heavy dependencies at CLI startup
        from omegaconf import OmegaConf
        import hydra
        from hydra import initialize, compose
        
        # Load config
        with initialize(version_base=None, config_path="../../configs"):
            cfg = compose(config_name="train_yolo")
        
        # Override parameters if provided
        if epochs:
            cfg.train.epochs = epochs
        if batch_size:
            cfg.train.batch_size = batch_size
        if device:
            cfg.train.device = device
        
        # Start training
        from ..training.train_yolo import YOLOTrainer
        trainer = YOLOTrainer(cfg)
        trainer.train()
        
        logger.info("✅ YOLO training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ YOLO training failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', default='configs/config.yaml', help='Config file path')
@click.option('--model-type', type=click.Choice(['resnet', 'yolo', 'both']), 
              default='both', help='Model type to evaluate')
@click.option('--split', type=click.Choice(['validation', 'test', 'both']), 
              default='test', help='Dataset split to evaluate')
def evaluate(config: str, model_type: str, split: str):
    """Evaluate trained models."""
    logger.info(f"Starting model evaluation - Model: {model_type}, Split: {split}")
    
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config)
        
        # Evaluate ResNet
        if model_type in ['resnet', 'both']:
            logger.info("Evaluating ResNet model...")
            from ..eval.eval_resnet import ResNetEvaluator
            resnet_evaluator = ResNetEvaluator(cfg)
            
            if split in ['validation', 'both']:
                resnet_evaluator.evaluate_split('validation')
            if split in ['test', 'both']:
                resnet_evaluator.evaluate_split('test')
        
        # Evaluate YOLO
        if model_type in ['yolo', 'both']:
            logger.info("Evaluating YOLO model...")
            from ..eval.eval_yolo import YOLOEvaluator
            yolo_evaluator = YOLOEvaluator(cfg)
            yolo_evaluator.evaluate()
        
        logger.info("✅ Model evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Model evaluation failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', default='configs/config.yaml', help='Config file path')
@click.option('--optimization-method', type=click.Choice(['grid_search', 'per_class', 'precision_recall']),
              default='per_class', help='Threshold optimization method')
@click.option('--metric', type=click.Choice(['f1', 'precision', 'recall']), 
              default='f1', help='Metric to optimize')
def tune_thresholds(config: str, optimization_method: str, metric: str):
    """Optimize classification thresholds."""
    logger.info(f"Starting threshold optimization - Method: {optimization_method}, Metric: {metric}")
    
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config)
        
        from ..utils.thresholding import ThresholdOptimizer
        optimizer = ThresholdOptimizer(cfg)
        
        # Run optimization based on method
        if optimization_method == 'grid_search':
            thresholds = optimizer.grid_search_thresholds(metric=f'{metric}_macro')
        elif optimization_method == 'per_class':
            thresholds = optimizer.optimize_thresholds_per_class(metric=metric)
        elif optimization_method == 'precision_recall':
            thresholds = optimizer.precision_recall_threshold_optimization()
        
        # Generate report
        optimizer.threshold_optimization_report(thresholds)
        
        logger.info("✅ Threshold optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Threshold optimization failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', default='configs/config.yaml', help='Config file path')
def compare_models(config: str):
    """Compare model performances and generate visualizations."""
    logger.info("Starting model comparison...")
    
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config)
        
        from ..eval.compare_models import ModelComparator
        comparator = ModelComparator(cfg)
        comparator.run_full_comparison()
        
        logger.info("✅ Model comparison completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Model comparison failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--image-path', '-i', required=True, help='Path to input image')
@click.option('--config', '-c', default='configs/config.yaml', help='Config file path')
@click.option('--output', '-o', help='Output JSON file path')
@click.option('--show-details', is_flag=True, help='Show detailed analysis')
@click.option('--show-timing', is_flag=True, help='Show timing information')
def infer(image_path: str, config: str, output: Optional[str], 
         show_details: bool, show_timing: bool):
    """Run inference on a single image."""
    logger.info(f"Running inference on: {image_path}")
    
    try:
        from PIL import Image
        import json
        
        # Load image
        if not Path(image_path).exists():
            logger.error(f"Image file not found: {image_path}")
            sys.exit(1)
        
        image = Image.open(image_path)
        logger.info(f"Loaded image: {image.size} {image.mode}")
        
        # Initialize pipeline
        from ..infer.pipeline import create_pipeline
        from ..infer.mapping import create_mapper
        
        pipeline = create_pipeline(config)
        mapper = create_mapper(config)
        
        # Run inference
        start_time = time.time()
        result = pipeline.predict(image)
        inference_time = time.time() - start_time
        
        # Map to correction codes if details requested
        if show_details:
            mapping_result = mapper.map_predictions_to_codes(result["predictions"])
            result.update(mapping_result)
        
        # Format output
        output_data = {
            "image_path": image_path,
            "predicted_classes": result["predicted_classes"],
            "inference_time": inference_time if show_timing else None,
            "model_versions": result["model_versions"],
            "timestamp": result["timestamp"]
        }
        
        if show_details:
            output_data.update({
                "correction_codes": result.get("correction_codes", []),
                "damage_assessment": result.get("damage_assessment", {}),
                "repair_recommendations": result.get("repair_recommendations", [])
            })
        
        if show_timing:
            output_data["detailed_timing"] = result["inference_times"]
        
        # Print results
        print("\n" + "="*60)
        print("COLLISION PARTS PREDICTION RESULTS")
        print("="*60)
        print(f"Image: {image_path}")
        print(f"Predicted Classes: {', '.join(result['predicted_classes']) if result['predicted_classes'] else 'No damage detected'}")
        
        if show_timing:
            print(f"Inference Time: {inference_time:.3f}s")
        
        if show_details and result.get("damage_assessment"):
            assessment = result["damage_assessment"]
            print(f"Overall Severity: {assessment.get('overall_severity', 'unknown')}")
            print(f"Estimated Cost: ${assessment.get('total_estimated_cost', 0):,}")
            print(f"Safety Impact: {assessment.get('safety_impact', 'unknown')}")
        
        print("="*60)
        
        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            logger.info(f"Results saved to: {output}")
        
        logger.info("✅ Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', default=8000, type=int, help='Port to bind to')
@click.option('--workers', default=1, type=int, help='Number of worker processes')
@click.option('--config', '-c', default='configs/config.yaml', help='Config file path')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(host: str, port: int, workers: int, config: str, reload: bool):
    """Start the FastAPI inference server."""
    logger.info(f"Starting Collision Parts Prediction Server on {host}:{port}")
    
    try:
        import uvicorn
        import os
        
        # Set config path as environment variable for the service
        os.environ['COLLISION_PARTS_CONFIG'] = config
        
        uvicorn.run(
            "src.infer.service:app",
            host=host,
            port=port,
            workers=workers if not reload else 1,  # Workers only work without reload
            reload=reload,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', default='configs/config.yaml', help='Config file path')
@click.option('--stage', help='Specific DVC stage to run')
@click.option('--force', is_flag=True, help='Force pipeline reproduction')
def pipeline(config: str, stage: Optional[str], force: bool):
    """Run the complete DVC pipeline."""
    logger.info("Running DVC pipeline...")
    
    try:
        import subprocess
        import os
        
        # Change to project root directory
        project_root = Path(__file__).parent.parent.parent
        os.chdir(project_root)
        
        # Build DVC command
        cmd = ['dvc', 'repro']
        
        if stage:
            cmd.append(stage)
        
        if force:
            cmd.append('--force')
        
        # Run DVC pipeline
        logger.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ DVC pipeline completed successfully!")
            if result.stdout:
                print(result.stdout)
        else:
            logger.error(f"❌ DVC pipeline failed: {result.stderr}")
            sys.exit(1)
        
    except FileNotFoundError:
        logger.error("❌ DVC not found. Please install DVC: pip install dvc")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)


@cli.command()
def status():
    """Show project status and configuration."""
    logger.info("Collision Parts Prediction System Status")
    
    try:
        project_root = Path(__file__).parent.parent.parent
        
        print("\n" + "="*60)
        print("COLLISION PARTS PREDICTION SYSTEM STATUS")
        print("="*60)
        
        # Check project structure
        print("\n📁 Project Structure:")
        key_paths = [
            "configs/config.yaml",
            "src/models/damage_net.py", 
            "src/training/train_resnet.py",
            "src/training/train_yolo.py",
            "src/eval/eval_resnet.py",
            "src/infer/pipeline.py",
            "dvc.yaml",
            "requirements.txt"
        ]
        
        for path in key_paths:
            full_path = project_root / path
            status = "✅" if full_path.exists() else "❌"
            print(f"  {status} {path}")
        
        # Check model files
        print("\n🤖 Model Files:")
        model_paths = [
            "models/resnet/best_model.pth",
            "models/yolo/best.pt"
        ]
        
        for path in model_paths:
            full_path = project_root / path
            if full_path.exists():
                size_mb = full_path.stat().st_size / (1024 * 1024)
                print(f"  ✅ {path} ({size_mb:.1f} MB)")
            else:
                print(f"  ❌ {path} (not found)")
        
        # Check data directories
        print("\n📊 Data Directories:")
        data_paths = [
            "data/raw",
            "data/processed", 
            "data/yolo"
        ]
        
        for path in data_paths:
            full_path = project_root / path
            if full_path.exists():
                file_count = len(list(full_path.rglob("*")))
                print(f"  ✅ {path} ({file_count} files)")
            else:
                print(f"  ❌ {path} (not found)")
        
        # Check dependencies
        print("\n📦 Python Dependencies:")
        try:
            import torch
            print(f"  ✅ PyTorch {torch.__version__}")
        except ImportError:
            print("  ❌ PyTorch (not installed)")
        
        try:
            import ultralytics
            print(f"  ✅ Ultralytics {ultralytics.__version__}")
        except ImportError:
            print("  ❌ Ultralytics (not installed)")
        
        try:
            import fastapi
            print(f"  ✅ FastAPI {fastapi.__version__}")
        except ImportError:
            print("  ❌ FastAPI (not installed)")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli()