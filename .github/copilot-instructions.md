# Collision Parts Prediction - AI Agent Instructions

## Project Overview
This is an ML system for vehicle collision damage assessment using dual-model architecture:
- **YOLO11m** for object detection (identifies vehicle parts)
- **ResNet50** for damage classification (assesses damage severity)
- **FastAPI** service for production inference
- **Hydra** configuration management
- **MLflow** experiment tracking
- **DVC** data versioning

## Architecture Patterns

### Configuration Management
- Use Hydra configs from `configs/` directory
- Override parameters via command line: `--config-name=train_resnet train.lr=0.001`
- Access nested config values: `cfg.train.learning_rate`
- Example: `with initialize(config_path="../../configs"): cfg = compose(config_name="train_resnet")`

### Model Training Structure
- Training scripts in `src/training/` use MLflow autologging
- Models saved to `models/` directory with versioning
- Follow pattern in `train_resnet.py`: initialize MLflow, create model, train loop with validation
- Use `loguru` logger instead of standard logging

### Inference Pipeline
- FastAPI service in `src/infer/service.py` with Pydantic schemas
- Pipeline pattern: `create_pipeline()` → `predict()` → `create_mapper()` → business logic mapping
- Response includes timing, model versions, and correction codes
- Example endpoint: `POST /predict/` accepts multipart file upload

### Data Processing
- DVC pipeline defined in `dvc.yaml` with stages: prepare_data → train_models → evaluate → compare
- Data flows: raw → processed → yolo format → model features
- Use `src/data/` modules for dataset classes and preprocessing

## Developer Workflows

### Testing
Use `run_tests.py` script instead of direct pytest:
```bash
python run_tests.py unit          # Unit tests only
python run_tests.py integration   # Integration tests  
python run_tests.py api           # API endpoint tests
python run_tests.py coverage      # Coverage report (70% minimum)
python run_tests.py quality       # Black + isort + ruff + bandit
python run_tests.py fix           # Auto-fix formatting/linting
```

### CLI Operations
Use `collision_parts_cli.py` for common tasks:
```bash
python collision_parts_cli.py prepare-data --config configs/config.yaml
python collision_parts_cli.py train-resnet --epochs 50 --lr 0.001
python collision_parts_cli.py infer --image path/to/image.jpg --show-details
python collision_parts_cli.py serve --host 0.0.0.0 --port 8000
```

### DVC Pipeline
```bash
dvc repro                    # Run full pipeline
dvc repro train_resnet       # Run specific stage
dvc repro --force            # Force re-run all stages
```

### Environment Setup
- Virtual environment in `collision_parts_env/`
- Install: `pip install -r requirements.txt`
- Pre-commit: `pre-commit install && pre-commit run --all-files`

### Terminal and Shell (Windows)
- Default terminal: Command Prompt (cmd). This repo includes `.vscode/settings.json` that sets `terminal.integrated.defaultProfile.windows` to "Command Prompt".
- Always prefer cmd syntax when providing commands for Windows.
- Detect the active shell:
	- In cmd: `echo %ComSpec%` (should output `C:\Windows\System32\cmd.exe`)
	- In PowerShell: `$PSVersionTable.PSVersion` and `$env:ComSpec`
- If a PowerShell terminal is active, do one of the following:
	- always run `cmd /c "c:\Users\myesh\Desktop\sample\collision_parts_env\Scripts\activate.bat && cmd"`
- Python interpreter path for this project (cmd): `c:/Users/myesh/Desktop/sample/collision_parts_env/Scripts/python.exe`

## Code Patterns

### Logging
```python
from loguru import logger
logger.info("Message with {variable}", variable=value)
# No need for string formatting - loguru handles it
```

### Error Handling
- Use specific exceptions from `src/infer/schemas.py`
- FastAPI endpoints return `ErrorResponse` with error codes
- CLI commands catch exceptions and exit with status codes

### Model Loading
```python
import torch
model = torch.load("models/resnet/best_model.pth", map_location=device)
model.eval()  # Always set to eval mode for inference
```

### Configuration Access
```python
from omegaconf import OmegaConf
cfg = OmegaConf.load("configs/config.yaml")
data_dir = cfg.paths.data_dir  # Access nested values
```

## Key Files to Reference

- `configs/config.yaml` - Main configuration with paths and defaults
- `src/infer/service.py` - FastAPI service implementation
- `src/training/train_resnet.py` - Training loop pattern
- `run_tests.py` - Test orchestration script
- `dvc.yaml` - Pipeline definition
- `src/data/dataset.py` - Data loading patterns

## Integration Points

- **MLflow**: All training logs to `mlflow.db`, UI at `mlflow ui`
- **DVC**: Data versioning with `dvc add data/processed/`
- **FastAPI**: OpenAPI docs at `/docs`, health check at `/health`
- **CLI**: Entry point `collision_parts_cli.py` with subcommands

## Common Gotchas

- Always add `src/` to Python path when running scripts directly
- Use absolute paths for model/data directories in configs
- Check `demo_mode: true` in config for development without model files
- Inference expects PIL Images, not file paths
- Threshold optimization requires validation data first</content>
<parameter name="filePath">c:\Users\myesh\Desktop\sample\.github\copilot-instructions.md