#!/usr/bin/env python3
"""
Test script for the Unified Transformer model.

Verifies that the model can be created and run forward passes.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_model_creation():
    """Test model creation and parameter counting."""
    try:
        from models.unified_transformer import create_unified_transformer

        print("Testing model creation...")

        # Test different sizes
        for size in ["tiny", "small", "base", "large"]:
            model = create_unified_transformer(size)
            params = model.get_num_parameters()
            print(f"  {size}: {params:,} parameters")

        print("✓ Model creation test passed")
        return True

    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False

def test_forward_pass():
    """Test forward pass with dummy data."""
    try:
        import torch
        from models.unified_transformer import create_unified_transformer

        print("Testing forward pass...")

        model = create_unified_transformer("base")
        model.eval()

        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 3, 640, 640)

        with torch.no_grad():
            outputs = model(x)

        # Check output shapes
        expected_shapes = {
            'bbox_pred': (batch_size, 100, 4),
            'class_logits': (batch_size, 100, 12),  # 11 parts + 1 no-object
            'location_logits': (batch_size, 100, 21),
            'severity_pred': (batch_size, 100, 1),
            'type_logits': (batch_size, 100, 11)
        }

        for key, expected_shape in expected_shapes.items():
            actual_shape = outputs[key].shape
            if actual_shape != expected_shape:
                print(f"✗ Wrong shape for {key}: expected {expected_shape}, got {actual_shape}")
                return False

        print("✓ Forward pass test passed")
        return True

    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        return False

    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        return False

def test_loss_function():
    """Test loss function computation."""
    try:
        import torch
        from models.losses import create_unified_loss

        print("Testing loss function...")

        criterion = create_unified_loss()

        # Create dummy predictions and targets
        predictions = {
            'bbox_pred': torch.randn(2, 100, 4),
            'class_logits': torch.randn(2, 100, 12),
            'location_logits': torch.randn(2, 100, 21),
            'severity_pred': torch.randn(2, 100, 1),
            'type_logits': torch.randn(2, 100, 11)
        }

        targets = {
            'bbox': torch.rand(2, 100, 4),
            'class': torch.randint(0, 11, (2, 100)),
            'location': torch.randint(0, 21, (2, 100)),
            'severity': torch.rand(2, 100, 1) * 3,       # 0-3 scale
            'type': torch.randint(0, 11, (2, 100)),
            'object_mask': torch.rand(2, 100) > 0.5
        }

        losses = criterion(predictions, targets)

        # Check that all losses are computed
        expected_keys = ['total_loss', 'bbox_loss', 'class_loss', 'location_loss', 'severity_loss', 'type_loss']
        for key in expected_keys:
            if key not in losses:
                print(f"✗ Missing loss: {key}")
                return False

        print(f"✓ Loss function test passed")
        return True

    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running Unified Transformer tests...")
    print("=" * 50)

    tests = [
        test_model_creation,
        test_forward_pass,
        test_loss_function
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())