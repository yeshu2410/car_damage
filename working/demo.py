#!/usr/bin/env python3
"""
Demo script showing how to use the Unified Transformer model.

This script demonstrates model creation, forward pass, and basic usage.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Demonstrate Unified Transformer usage."""

    print("Unified Transformer Demo")
    print("=" * 50)

    try:
        import torch
        from models.unified_transformer import create_unified_transformer

        # Create model
        print("Creating Unified Transformer (base)...")
        model = create_unified_transformer("base")
        model.eval()

        print(f"Model parameters: {model.get_num_parameters():,}")

        # Create sample input
        batch_size = 1
        image = torch.randn(batch_size, 3, 640, 640)

        print(f"Input shape: {image.shape}")

        # Forward pass
        with torch.no_grad():
            predictions = model(image)

        print("\nModel outputs:")
        for key, value in predictions.items():
            print(f"  {key}: {value.shape}")

        # Show sample predictions
        print("\nSample predictions (first query):")
        print(f"  Bounding box: {predictions['bbox_pred'][0, 0].tolist()}")
        print(f"  Vehicle part class: {predictions['class_logits'][0, 0].argmax().item()}")
        print(f"  Damage location: {predictions['location_logits'][0, 0].argmax().item()}")
        print(".3f")
        print(f"  Damage type: {predictions['type_logits'][0, 0].argmax().item()}")

        print("\n✅ Demo completed successfully!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install PyTorch and other dependencies:")
        print("pip install torch torchvision torchaudio")
        return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())