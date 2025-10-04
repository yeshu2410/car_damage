# Development Guide

This document provides detailed information for developers working on the Collision Parts Prediction system.

## Setup for Development

### Quick Setup
```bash
# 1. Clone and enter repository
git clone <repository-url>
cd collision-parts-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup development environment (installs pre-commit hooks)
python run_tests.py setup

# 4. Verify installation
python run_tests.py unit
```

### Manual Setup (Alternative)
```bash
# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run initial checks
pre-commit run --all-files
```

## Testing Framework

### Test Categories

The project uses pytest with custom markers for different test types:

- **Unit tests** (`-m "not slow and not integration"`): Fast tests of individual components
- **Integration tests** (`-m integration`): Tests of component interactions
- **API tests** (`tests/test_infer_api.py`): FastAPI endpoint testing
- **Slow tests** (`-m slow`): Performance tests, model loading tests

### Running Tests

Using the test runner (recommended):
```bash
python run_tests.py unit            # Fast unit tests
python run_tests.py integration     # Integration tests
python run_tests.py api             # API endpoint tests
python run_tests.py all             # All tests with coverage
python run_tests.py coverage        # Detailed coverage report
python run_tests.py slow            # Performance tests
```

Using pytest directly:
```bash
pytest tests/ -v                    # All tests, verbose
pytest tests/ -m "not slow"         # Skip slow tests
pytest tests/test_data_prep.py      # Specific test file
pytest -k "test_via_parsing"        # Tests matching pattern
```

### Test Configuration

Tests are configured via `pytest.ini`:
- Coverage reporting enabled
- Custom markers defined
- Test discovery patterns
- Output formatting

Key test markers:
- `@pytest.mark.slow`: For tests that take >1 second
- `@pytest.mark.integration`: For tests requiring multiple components
- `@pytest.mark.api`: For API endpoint tests

### Writing Tests

Follow these patterns when adding tests:

```python
import pytest
from unittest.mock import Mock, patch

class TestMyComponent:
    """Test cases for MyComponent."""
    
    def test_basic_functionality(self, sample_config):
        """Test basic functionality with sample config."""
        # Arrange
        component = MyComponent(sample_config)
        
        # Act
        result = component.process()
        
        # Assert
        assert result is not None
        assert len(result) > 0
    
    @pytest.mark.slow
    def test_performance(self, large_dataset):
        """Test performance with large dataset."""
        # Performance test code
        pass
    
    @pytest.mark.integration
    def test_integration_with_pipeline(self, mock_pipeline):
        """Test integration with full pipeline."""
        # Integration test code
        pass
```

## Code Quality

### Automated Checks

The project enforces code quality through:
- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Fast Python linting
- **Bandit**: Security scanning

### Running Quality Checks

```bash
# Check all quality standards
python run_tests.py quality

# Auto-fix formatting issues
python run_tests.py fix

# Individual tools
black --check src/ tests/
isort --check-only src/ tests/
ruff check src/ tests/
bandit -r src/
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`:
- Black formatting
- isort import sorting
- Ruff linting
- Bandit security checks
- Pytest (fast tests only)

To run manually:
```bash
pre-commit run --all-files
```

## Project Architecture

### Directory Structure

```
src/
├── data/               # Data processing pipeline
│   ├── __init__.py
│   ├── prepare_data.py # Main data preparation script
│   └── utils.py        # Data utilities
├── models/             # Model implementations
│   ├── __init__.py
│   ├── resnet.py       # ResNet classification model
│   └── yolo.py         # YOLO detection model
├── train.py            # Training orchestration
├── evaluate.py         # Model evaluation
├── thresholding.py     # Threshold optimization
└── infer/              # Inference pipeline
    ├── __init__.py
    ├── api.py          # FastAPI application
    ├── inference.py    # Core inference logic
    └── utils.py        # Inference utilities
```

### Key Components

1. **Data Pipeline** (`src/data/`):
   - VEHiDe dataset processing
   - VIA annotation parsing
   - YOLO format conversion
   - Data validation

2. **Model Training** (`src/train.py`):
   - Hydra configuration management
   - MLflow experiment tracking
   - Multi-architecture support

3. **Evaluation** (`src/evaluate.py`):
   - Model performance metrics
   - Threshold optimization
   - Confusion matrix generation

4. **Inference** (`src/infer/`):
   - FastAPI web service
   - Batch processing
   - Model fusion pipeline

### Configuration Management

Uses Hydra for hierarchical configuration:

```yaml
# configs/config.yaml
defaults:
  - data: vehicular_parts
  - model: yolo_detection
  - experiment: baseline

# Override from command line
python src/train.py model=resnet_classification experiment=tuning
```

## CI/CD Pipeline

### GitHub Actions Workflow

The CI pipeline (`.github/workflows/ci.yml`) includes:
1. **Multi-OS testing** (Ubuntu, Windows, macOS)
2. **Python version matrix** (3.9, 3.10, 3.11)
3. **Quality gates** (formatting, linting, security)
4. **Test execution** (unit, integration, API)
5. **Coverage reporting** (minimum 70%)

### Quality Gates

All PRs must pass:
- ✅ Code formatting (Black)
- ✅ Import sorting (isort)
- ✅ Linting (Ruff)
- ✅ Security scanning (Bandit)
- ✅ Test coverage ≥ 70%
- ✅ All tests passing

### Release Process

1. Create feature branch from `main`
2. Implement changes with tests
3. Run quality checks: `python run_tests.py quality`
4. Run full test suite: `python run_tests.py all`
5. Submit PR with comprehensive description
6. Automated CI checks must pass
7. Code review and approval required
8. Merge to `main` triggers deployment

## Performance Optimization

### Inference Performance

- **Model fusion**: Combine YOLO + ResNet predictions
- **Batch processing**: Process multiple images efficiently
- **Preprocessing optimization**: Fast image transformations
- **Memory management**: Efficient tensor operations

### Testing Performance

- **Mocking**: Avoid loading actual models in tests
- **Fixtures**: Reuse expensive setup across tests
- **Parallel execution**: Run independent tests concurrently
- **Selective testing**: Use markers to skip slow tests

## Troubleshooting

### Common Issues

1. **Import errors in tests**:
   ```bash
   # Ensure src/ is in Python path
   export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
   ```

2. **Pre-commit hook failures**:
   ```bash
   # Fix formatting issues
   python run_tests.py fix
   
   # Update hooks
   pre-commit autoupdate
   ```

3. **Test failures with mock models**:
   ```bash
   # Check fixture setup in conftest.py
   pytest tests/conftest.py -v
   ```

4. **Coverage too low**:
   ```bash
   # Generate detailed coverage report
   python run_tests.py coverage
   # Check htmlcov/index.html for detailed info
   ```

### Debug Tips

- Use `pytest -s` to see print statements
- Use `pytest --pdb` to drop into debugger on failure
- Use `pytest -v` for verbose test output
- Use `pytest -k pattern` to run specific tests

## Contributing Guidelines

### Code Style
- Follow PEP 8 (enforced by Black)
- Use type hints where beneficial
- Write descriptive docstrings
- Keep functions focused and testable

### Test Requirements
- Write tests for new functionality
- Maintain test coverage above 70%
- Use appropriate test markers
- Mock external dependencies

### Documentation
- Update README for user-facing changes
- Add docstrings for public APIs
- Include examples in docstrings
- Update this development guide for process changes

### Commit Messages
Follow conventional commits:
```
feat: add batch prediction endpoint
fix: resolve memory leak in model loading
docs: update API documentation
test: add integration tests for pipeline
```

This ensures clear commit history and enables automated changelog generation.