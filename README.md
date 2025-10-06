# Collision Parts Prediction System

A comprehensive machine learning system for predicting damaged parts in vehicle collisions using computer vision and damage assessment.

## Overview

This project implements a scalable proof-of-concept for collision damage assessment using:
- **Unified Transformer Model** (NEW!) - End-to-end detection and damage classification
- **YOLO11m object detection** for part identification
- **ResNet50 classification** for damage severity assessment
- **MLflow** for experiment tracking and model management
- **FastAPI** for high-performance inference serving
- **Docker** for containerized deployment
- **Kubernetes** for orchestration

### 🆕 Unified Transformer Architecture

The new **Unified Transformer Model** combines object detection and multi-task damage classification in a single end-to-end architecture:

- **Vision Transformer backbone** for feature extraction
- **DETR-style object detection** with learned queries
- **Multi-task prediction heads**:
  - Vehicle part detection (bounding boxes + class)
  - Damage location classification (21 classes)
  - Damage severity regression (0-10 scale)
  - Damage type classification (11 classes)
- **Single forward pass** for all predictions
- **Unified loss function** balancing all tasks

## Project Structure

```
collision-parts-prediction/
├── configs/                # Hydra configuration files
├── data/                   # Data pipeline and utilities
├── docker/                 # Docker configurations
├── docs/                   # Documentation
├── k8s/                    # Kubernetes manifests
├── models/                 # Model implementations
├── notebooks/              # Jupyter notebooks for experimentation
├── scripts/                # Utility scripts
├── src/                    # Source code
├── tests/                  # Test suite (comprehensive)
├── utils/                  # Utilities and helpers
├── run_tests.py           # Test runner script
└── requirements.txt       # Project dependencies
```

## Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional)
- Git LFS for model files

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd collision-parts-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up development environment:**
   ```bash
   python run_tests.py setup
   ```

4. **Run basic tests:**
   ```bash
   python run_tests.py unit
   ```

### Testing

The project includes comprehensive testing with multiple test categories:

```bash
# Run different test suites
python run_tests.py unit            # Unit tests only
python run_tests.py integration     # Integration tests
python run_tests.py api             # API endpoint tests
python run_tests.py all             # All tests with coverage
python run_tests.py coverage        # Generate coverage report
python run_tests.py quality         # Code quality checks
python run_tests.py fix             # Auto-fix code issues
```

### Configuration

The system uses Hydra for configuration management. Edit files in `configs/` to customize:
- `configs/data/` - Data pipeline settings
- `configs/model/` - Model architecture and training
- `configs/experiment/` - Experiment tracking

### Training Models

#### Option 1: Unified Transformer (Recommended - End-to-End)

Train a single model for all tasks:

```bash
# Train unified transformer
python src/training/train_unified_transformer.py

# With custom config
python src/training/train_unified_transformer.py \
  train.model_size=base \
  train.training.batch_size=8 \
  train.training.learning_rate=0.0001
```

**Model sizes:**
- `tiny`: 384 dim, 4 layers, ~50M params
- `small`: 512 dim, 6 layers, ~90M params  
- `base`: 768 dim, 6 layers, ~150M params (default)
- `large`: 1024 dim, 12 layers, ~300M params

#### Option 2: Separate Models (Original Pipeline)

1. **Prepare data:**
   ```bash
   python src/data/prepare_data.py
   ```

2. **Train YOLO detection model:**
   ```bash
   python src/training/train_yolo.py
   ```

3. **Train ResNet classification model:**
   ```bash
   python src/training/train_resnet.py
   ```

### Inference

#### Unified Transformer Inference

Single-model inference with all predictions:

```bash
# Basic inference
python src/infer/unified_transformer_inference.py \
  --model models/unified_transformer_best.pth \
  --image test_images/damaged_car.jpg \
  --report

# Save JSON output
python src/infer/unified_transformer_inference.py \
  --model models/unified_transformer_best.pth \
  --image test_images/damaged_car.jpg \
  --output predictions.json \
  --conf 0.5

# Use GPU
python src/infer/unified_transformer_inference.py \
  --model models/unified_transformer_best.pth \
  --image test_images/damaged_car.jpg \
  --device cuda \
  --report
```

**Output includes:**
- Bounding boxes for detected damage
- Vehicle part classification
- Damage location (21 categories)
- Damage severity (0-10 scale)
- Damage type (scratch, dent, etc.)

#### Traditional Pipeline Inference

```bash
python src/evaluate.py model_path=models/best_model.pth
```

### Inference & Serving

**Start the FastAPI server:**
```bash
python start_server.py
```

**CLI inference:**
```bash
python collision_parts_cli.py predict --image path/to/image.jpg
```

### Deployment

**Local development:**
```bash
python start_server.py
```

**Docker deployment:**
```bash
docker build -f docker/Dockerfile -t collision-prediction .
docker run -p 8000:8000 collision-prediction
```

**Kubernetes deployment:**
```bash
kubectl apply -f k8s/
```

## API Usage

Once the server is running, you can make predictions:

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Single image prediction
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict/",
        files={"file": f}
    )
    
print(response.json())

# Batch prediction
files = [
    ("files", open("image1.jpg", "rb")),
    ("files", open("image2.jpg", "rb"))
]
response = requests.post(
    "http://localhost:8000/predict/batch",
    files=files
)
print(response.json())
```

## Development

This project follows best practices for ML development:

- **Code quality:** Black, isort, ruff for formatting and linting
- **Testing:** Pytest with comprehensive test coverage (70%+ required)
- **Experiment tracking:** MLflow for model versioning
- **Data versioning:** DVC for dataset management
- **CI/CD:** GitHub Actions for automated testing and deployment
- **Pre-commit hooks:** Automated quality checks on commit

### Quality Gates

The CI pipeline enforces these quality standards:
- ✅ Code formatting (Black)
- ✅ Import sorting (isort)
- ✅ Linting (Ruff)
- ✅ Security scanning (Bandit)
- ✅ Test coverage ≥ 70%
- ✅ All tests passing
- ✅ Type checking (optional)

### Running Quality Checks

```bash
# Check code quality
python run_tests.py quality

# Auto-fix issues
python run_tests.py fix

# Full test suite with coverage
python run_tests.py coverage
```

## System Features

### Data Pipeline
- VEHiDe dataset processing
- VIA annotation format support
- YOLO format conversion
- Data validation and cleaning

### Model Architecture
- YOLO11m for object detection
- ResNet for damage classification
- Model fusion for final predictions
- Threshold optimization per class

### Inference Pipeline
- Fast preprocessing pipeline
- Batch processing support
- Confidence thresholding
- Business logic mapping

### API Features
- Health monitoring endpoints
- Single and batch prediction
- Async processing support
- Comprehensive error handling
- Request validation

## Performance

The system is designed for production workloads:
- **Inference latency:** < 200ms per image
- **Throughput:** 50+ images/second (batch)
- **Memory usage:** < 2GB with model fusion
- **Scalability:** Kubernetes-ready horizontal scaling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run quality checks: `python run_tests.py quality`
5. Run tests: `python run_tests.py all`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.