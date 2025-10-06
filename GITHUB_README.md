# 🚗 Vehicle Collision Damage Detection System

**AI-powered system for automatic detection and classification of vehicle damage using computer vision and deep learning.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This system automatically detects and classifies vehicle damage from images using state-of-the-art deep learning models. It can identify:

- **8 Damage Types**: Scratch, Dent, Cracked, Broken part, Missing part, Corrosion, Paint chip, Flaking
- **21 Vehicle Parts**: Front/Back bumper, Doors, Fenders, Hood, Trunk, Wheels, Lights, Windshield, etc.
- **Damage Severity**: Scored from 1-10
- **Damage Location**: Precise bounding boxes and segmentation masks

### Use Cases
- 🚗 Insurance claim processing
- 🔧 Auto repair estimation
- 📸 Vehicle inspection automation
- 📊 Fleet damage tracking

---

## ✨ Features

- ✅ **Dual-Model Architecture**: YOLO11m for detection + ResNet50 for classification
- ✅ **Unified Transformer**: Single end-to-end model option (experimental)
- ✅ **COCO Format Annotations**: Industry-standard annotation format
- ✅ **Batch Processing**: Process thousands of images automatically
- ✅ **RESTful API**: FastAPI-based inference service
- ✅ **MLflow Integration**: Experiment tracking and model versioning
- ✅ **DVC Support**: Data version control
- ✅ **Docker Ready**: Containerized deployment

---

## 🏗️ Architecture

### Option 1: YOLO + ResNet Pipeline (Production-Ready)

```
┌─────────────┐
│   Image     │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  YOLO11m Detector   │
│  - Damage detection │
│  - Part detection   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  ResNet50 Classifier│
│  - Damage type      │
│  - Severity score   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Post-processing    │
│  - Match damage→part│
│  - Generate report  │
└──────┬──────────────┘
       │
       ▼
    Results
```

### Option 2: Unified Transformer (Experimental)

```
┌─────────────┐
│   Image     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│  Unified Transformer (DETR-like)│
│  - Patch embedding              │
│  - Encoder (6 layers)           │
│  - Decoder (6 layers)           │
│  - Multi-task heads:            │
│    • Bounding boxes             │
│    • Vehicle part class         │
│    • Damage type                │
│    • Damage location            │
│    • Severity score             │
└──────┬──────────────────────────┘
       │
       ▼
    Results
```

---

## 📊 Dataset

### Statistics
- **Total Images**: 33,214 (full dataset)
- **Annotated Subset**: 2,500 images
- **Total Annotations**: 8,118 damage instances
- **Train/Val/Test Split**: 70% / 15% / 15%

### Damage Distribution
| Type | Count | Percentage |
|------|-------|-----------|
| Scratch | 5,834 | 71.9% |
| Broken part | 632 | 7.8% |
| Dent | 560 | 6.9% |
| Cracked | 432 | 5.3% |
| Paint chip | 236 | 2.9% |
| Missing part | 216 | 2.7% |
| Corrosion | 150 | 1.8% |
| Flaking | 58 | 0.7% |

### Most Damaged Parts
1. Front-bumper (29.8%)
2. Back-bumper (18.9%)
3. Front-door (10.6%)
4. Fender (6.1%)

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- 50GB+ disk space

### Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/collision-damage-detection.git
cd collision-damage-detection

# Create virtual environment
python -m venv collision_parts_env
source collision_parts_env/bin/activate  # On Windows: collision_parts_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pretrained models (if available)
# python download_models.py
```

### Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
fastapi>=0.100.0
uvicorn>=0.23.0
mlflow>=2.5.0
hydra-core>=1.3.0
loguru>=0.7.0
```

---

## 🚀 Usage

### 1. Validate COCO Annotations

```bash
cd working
python validate_coco.py
```

### 2. Split Dataset

```bash
python split_coco_dataset.py
```

### 3. Batch Processing (Generate Annotations)

```bash
# Process first 2500 images
python prepare_2500_dataset.py
python batch_to_coco_2500.py
```

### 4. Train Model

#### Option A: Train Unified Transformer
```bash
cd working
python src/training/train_unified_transformer.py
```

#### Option B: Train ResNet Classifier
```bash
python src/training/train_resnet.py
```

### 5. Run Inference

```bash
# Single image
python src/infer/unified_transformer_inference.py --image path/to/car.jpg

# Batch inference
python batch_to_coco_2500.py --dataset-dir data/processed/yolo/images_2500
```

### 6. Start API Server

```bash
cd working
python app_batch.py
```

Then open: http://localhost:7860

---

## 📚 API Reference

### POST /predict
Predict damages from a single image.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@car_image.jpg"
```

**Response:**
```json
{
  "damages": [
    {
      "type": "Scratch",
      "location": "Front-bumper",
      "severity": 6.5,
      "bbox": [150, 200, 80, 60],
      "confidence": 0.92,
      "segmentation": [[150, 200, 230, 200, 230, 260, 150, 260]]
    }
  ],
  "processing_time_ms": 245,
  "model_version": "v1.0"
}
```

---

## 📁 Project Structure

```
collision-damage-detection/
├── configs/                      # Configuration files
│   ├── config.yaml              # Main config
│   ├── correction_codes.yaml    # Damage correction codes
│   ├── thresholds.yaml          # Detection thresholds
│   └── train/                   # Training configs
├── data/
│   ├── processed/yolo/
│   │   ├── images_2500/         # Selected 2500 images
│   │   └── batch_output/
│   │       └── splits/          # Train/Val/Test splits
│   │           ├── annotations_train.json
│   │           ├── annotations_val.json
│   │           └── annotations_test.json
├── src/
│   ├── data/                    # Data processing
│   ├── models/                  # Model architectures
│   │   └── unified_transformer.py
│   ├── training/                # Training scripts
│   │   ├── train_unified_transformer.py
│   │   ├── train_resnet.py
│   │   └── train_yolo.py
│   ├── infer/                   # Inference pipeline
│   └── utils/                   # Utilities
├── working/                     # Experimental scripts
│   ├── app_batch.py            # Gradio web interface
│   ├── batch_to_coco_2500.py  # Batch annotation generator
│   ├── validate_coco.py        # Validation script
│   ├── split_coco_dataset.py  # Dataset splitter
│   └── PROJECT_STATUS.md       # Current status
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── requirements.txt             # Python dependencies
├── dvc.yaml                     # DVC pipeline
└── README.md                    # This file
```

---

## 📈 Results

### Model Performance (Expected)

| Model | Accuracy | Inference Time | Model Size |
|-------|----------|----------------|------------|
| YOLO11m + ResNet50 | 92-95% | 30-50ms (GPU) | 250MB |
| Unified Transformer | 88-93% | 50-100ms (GPU) | 350MB |

### Training Time
- **YOLO11m**: 2-3 days (GPU) / 5-7 days (CPU)
- **ResNet50**: 1-2 days (GPU) / 3-5 days (CPU)
- **Unified Transformer**: 3-5 days (GPU) / 7-10 days (CPU)

---

## 🛠️ Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Detection thresholds
damage_threshold: 0.4    # 40% confidence
part_threshold: 0.3      # 30% confidence
overlap_threshold: 0.4   # 40% IoU for matching

# Training
epochs: 50
batch_size: 16
learning_rate: 0.001
image_size: 448

# Model
model_type: "unified_transformer"  # or "yolo_resnet"
model_size: "small"  # tiny, small, base, large
```

---

## 🧪 Testing

```bash
# Run all tests
python run_tests.py all

# Run specific test suites
python run_tests.py unit
python run_tests.py integration
python run_tests.py api

# Check code quality
python run_tests.py quality

# Generate coverage report
python run_tests.py coverage
```

---

## 📝 Training Your Own Model

### 1. Prepare Your Dataset

```bash
# Organize images
data/raw/your_images/
├── image001.jpg
├── image002.jpg
└── ...

# Generate COCO annotations (if you have labels)
python working/batch_to_coco.py --dataset-dir data/raw/your_images
```

### 2. Configure Training

Edit `configs/train/unified_transformer.yaml`:

```yaml
model:
  name: "unified_transformer"
  size: "small"  # Faster training

train:
  epochs: 50
  batch_size: 8
  learning_rate: 0.0001
```

### 3. Start Training

```bash
python src/training/train_unified_transformer.py
```

### 4. Monitor Progress

```bash
# Open MLflow UI
mlflow ui

# Navigate to: http://localhost:5000
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t collision-damage-api -f docker/Dockerfile.infer .

# Run container
docker run -p 8000:8000 collision-damage-api

# Test
curl http://localhost:8000/health
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ultralytics YOLO** - Object detection framework
- **PyTorch** - Deep learning framework
- **DETR** - Detection Transformer inspiration
- **MS COCO** - Annotation format standard

---

## 📞 Contact

- **Project Lead**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

## 🗺️ Roadmap

- [x] COCO annotation generation (2,500 images)
- [x] Dataset splitting (Train/Val/Test)
- [ ] Train Unified Transformer model
- [ ] Model evaluation and benchmarking
- [ ] FastAPI production deployment
- [ ] Real-time inference optimization
- [ ] Mobile app integration
- [ ] Multi-language support

---

## ⭐ Star History

If you find this project useful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=YOUR_USERNAME/collision-damage-detection&type=Date)](https://star-history.com/#YOUR_USERNAME/collision-damage-detection&Date)

---

**Made with ❤️ for safer roads and smarter insurance**
