#!/usr/bin/env python3
"""
Entry point script for Collision Parts Prediction CLI.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.cli.main import cli

if __name__ == "__main__":
    cli()