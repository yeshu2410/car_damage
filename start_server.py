#!/usr/bin/env python3
"""
Entry point script for Collision Parts Prediction API Server.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.infer.service import run_server

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collision Parts Prediction API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, workers=args.workers)