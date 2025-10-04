# Tests package
"""
Test suite for the Collision Parts Prediction system.

This package contains comprehensive tests for:
- Data preparation and preprocessing
- Model training and evaluation  
- Threshold optimization
- Inference pipeline and API
- Integration testing

Test categories:
- Unit tests: Fast, isolated component tests
- Integration tests: Multi-component workflow tests
- API tests: FastAPI endpoint testing
- Slow tests: Heavy model/training tests (marked as @pytest.mark.slow)

Usage:
    # Run all tests
    pytest tests/

    # Run fast tests only
    pytest tests/ -m "not slow"
    
    # Run with coverage
    pytest tests/ --cov=src --cov-report=html
    
    # Run specific test file
    pytest tests/test_data_prep.py -v
    
    # Run API tests
    pytest tests/test_infer_api.py -v
"""