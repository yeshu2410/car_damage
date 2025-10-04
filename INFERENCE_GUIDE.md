# Inference and Serving Guide

## Overview

The Collision Parts Prediction system provides multiple ways to use the trained models:

1. **Python API** - Direct inference pipeline for integration
2. **REST API** - FastAPI-based web service with OpenAPI docs  
3. **Command Line Interface** - CLI commands for all operations
4. **Batch Processing** - Efficient processing of multiple images

## Quick Start

### 1. Start the API Server

```bash
# Using the CLI
python collision_parts_cli.py serve --host 0.0.0.0 --port 8000

# Or using the direct script
python start_server.py --host 0.0.0.0 --port 8000
```

### 2. Check Health Status

```bash
curl http://localhost:8000/healthz
```

### 3. Run Inference via API

```python
import base64
import requests
from PIL import Image

# Load and encode image
with open("test_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Make prediction request
response = requests.post("http://localhost:8000/predict", json={
    "image": {
        "image_data": image_data,
        "image_format": "JPEG"
    },
    "include_details": True,
    "include_timing": True
})

result = response.json()
print(f"Predicted classes: {result['predicted_classes']}")
```

### 4. CLI Inference

```bash
# Single image inference
python collision_parts_cli.py infer -i path/to/image.jpg --show-details --show-timing

# Full pipeline execution
python collision_parts_cli.py pipeline

# Check system status
python collision_parts_cli.py status
```

## Inference Pipeline Architecture

### Core Components

1. **`pipeline.py`** - Main inference orchestrator
   - Loads ResNet and YOLO models
   - Manages model versions and thresholds
   - Implements score fusion (max rule)
   - Tracks performance statistics

2. **`mapping.py`** - Prediction to business logic mapper
   - Maps damage classes to correction codes
   - Identifies affected vehicle parts
   - Analyzes damage severity
   - Generates repair recommendations

3. **`service.py`** - FastAPI web service
   - RESTful API endpoints
   - Request/response validation
   - Error handling and logging
   - Health checks and monitoring

4. **`schemas.py`** - Pydantic data models
   - Type-safe request/response schemas
   - Input validation with Pydantic
   - Comprehensive API documentation

## API Endpoints

### Health Check
```
GET /healthz
```
Returns service health status and model loading state.

### Single Prediction
```
POST /predict
```
Processes a single image and returns damage predictions.

**Request Body:**
```json
{
  "image": {
    "image_data": "base64_encoded_image",
    "image_format": "JPEG"
  },
  "include_details": true,
  "include_timing": false,
  "confidence_threshold": 0.5
}
```

**Response:**
```json
{
  "predictions": {
    "front_bumper_damage": {
      "fused_score": 0.85,
      "resnet_score": 0.82,
      "yolo_score": 0.85,
      "predicted": true,
      "threshold": 0.5,
      "fusion_method": "max_rule"
    }
  },
  "predicted_classes": ["front_bumper_damage"],
  "correction_codes": [...],
  "damage_assessment": {
    "overall_severity": "moderate",
    "total_estimated_cost": 850,
    "repair_priority": "medium"
  }
}
```

### Batch Prediction
```
POST /predict/batch
```
Processes multiple images (max 10) in a single request.

### Service Statistics
```
GET /stats
```
Returns detailed service performance statistics.

## CLI Commands

### Data Preparation
```bash
# Prepare training data
python collision_parts_cli.py prepare-data --config configs/config.yaml

# Override data directory
python collision_parts_cli.py prepare-data --data-dir /path/to/data
```

### Model Training
```bash
# Train ResNet classifier
python collision_parts_cli.py train-resnet --epochs 50 --batch-size 32

# Train YOLO detector  
python collision_parts_cli.py train-yolo --epochs 100 --device cuda

# Resume training from checkpoint
python collision_parts_cli.py train-resnet --resume
```

### Model Evaluation
```bash
# Evaluate both models on test set
python collision_parts_cli.py evaluate --model-type both --split test

# Evaluate only ResNet on validation set
python collision_parts_cli.py evaluate --model-type resnet --split validation
```

### Threshold Optimization
```bash
# Optimize thresholds per class
python collision_parts_cli.py tune-thresholds --optimization-method per_class

# Grid search optimization
python collision_parts_cli.py tune-thresholds --optimization-method grid_search --metric f1
```

### Model Comparison
```bash
# Generate comparison plots and reports
python collision_parts_cli.py compare-models
```

### Pipeline Management
```bash
# Run complete DVC pipeline
python collision_parts_cli.py pipeline

# Run specific stage
python collision_parts_cli.py pipeline --stage train_resnet

# Force pipeline reproduction
python collision_parts_cli.py pipeline --force
```

### Inference
```bash
# Basic inference
python collision_parts_cli.py infer -i image.jpg

# Detailed analysis with timing
python collision_parts_cli.py infer -i image.jpg --show-details --show-timing

# Save results to file
python collision_parts_cli.py infer -i image.jpg -o results.json
```

### API Server
```bash
# Start development server
python collision_parts_cli.py serve --host 127.0.0.1 --port 8000 --reload

# Start production server with multiple workers
python collision_parts_cli.py serve --host 0.0.0.0 --port 8000 --workers 4
```

### System Status
```bash
# Check system status and configuration
python collision_parts_cli.py status
```

## Python API Usage

### Direct Pipeline Usage

```python
from src.infer import create_pipeline, create_mapper
from PIL import Image

# Initialize pipeline
pipeline = create_pipeline("configs/config.yaml")
mapper = create_mapper("configs/config.yaml")

# Load image
image = Image.open("test_image.jpg")

# Run inference
result = pipeline.predict(image)

# Map to business logic
mapped_result = mapper.map_predictions_to_codes(result["predictions"])

print(f"Predicted classes: {result['predicted_classes']}")
print(f"Estimated cost: ${mapped_result['damage_assessment']['total_estimated_cost']}")
```

### Service Integration

```python
from src.infer import CollisionPartsService
import asyncio

async def run_prediction():
    service = CollisionPartsService("configs/config.yaml")
    await service.startup()
    
    # Create request
    from src.infer.schemas import PredictionRequest, ImageInput
    
    request = PredictionRequest(
        image=ImageInput(image_data=base64_image_data),
        include_details=True
    )
    
    # Run prediction
    result = await service.predict_single(request, "test_request")
    return result

# Run async prediction
result = asyncio.run(run_prediction())
```

## Configuration

### Model Paths
Models are automatically loaded from:
- ResNet: `models/resnet/best_model.pth`
- YOLO: `models/yolo/best.pt`

### Thresholds
Optimized thresholds are loaded from:
- `outputs/evaluation/optimized_thresholds.json`

### Correction Codes
Business logic mapping from:
- `configs/correction_codes.yaml`

## Performance Features

### Model Fusion
- **Max Rule**: Takes maximum score between ResNet and YOLO
- **Adaptive Thresholds**: Uses per-class optimized thresholds
- **Confidence Weighting**: Weights predictions by model confidence

### Optimization
- **Model Warmup**: Pre-runs inference to optimize GPU memory
- **Batch Processing**: Efficient processing of multiple images
- **Response Caching**: Optional caching for repeated requests
- **Memory Management**: Automatic garbage collection and memory monitoring

### Monitoring
- **Performance Metrics**: Tracks inference times and throughput
- **Resource Usage**: Monitors CPU, memory, and GPU utilization
- **Health Checks**: Validates model loading and system status
- **Error Tracking**: Comprehensive error logging and reporting

## Deployment

### Docker Deployment
```bash
# Build container
docker build -t collision-parts-api .

# Run API server
docker run -p 8000:8000 collision-parts-api

# Run with GPU support
docker run --gpus all -p 8000:8000 collision-parts-api
```

### Production Considerations
- Use multiple workers for high throughput
- Enable HTTPS in production
- Set up load balancing for scaling
- Configure proper CORS settings
- Implement rate limiting
- Set up monitoring and alerting

## API Documentation

Once the server is running, comprehensive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

The documentation includes:
- Interactive API testing
- Request/response schemas
- Authentication details
- Example code snippets
- Error code definitions