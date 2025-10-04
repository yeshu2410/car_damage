# Collision Parts Prediction

A large-scale machine learning project for predicting collision parts using computer vision and deep learning techniques.

## Overview

This project implements state-of-the-art object detection and classification models to identify and predict collision parts from vehicle images. The system supports both ResNet-based classification and YOLO-based object detection approaches.

## Features

- Multiple model architectures (ResNet, YOLO)
- Scalable training pipeline with hyperparameter tuning
- RESTful API for inference
- MLflow experiment tracking
- DVC data versioning
- Docker containerization
- Kubernetes deployment ready

## Installation

```bash
make setup
```

## Usage

### Data Preparation
```bash
make data
```

### Training
```bash
# Train ResNet model
make train-resnet

# Train YOLO model
make train-yolo
```

### Hyperparameter Tuning
```bash
make tune
```

### Inference
```bash
# Local inference
make infer

# Serve API
make serve
```

## Project Structure

```
├── configs/          # Configuration files
├── src/             # Source code
├── tests/           # Test files
├── data/            # Data directory
│   ├── raw/         # Raw data
│   ├── processed/   # Processed data
│   └── artifacts/   # Model artifacts
├── docker/          # Docker configurations
└── k8s/            # Kubernetes manifests
```

## License

MIT License - see LICENSE file for details.