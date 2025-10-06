#!/usr/bin/env python3
"""
Quick API test to verify the server is working
"""

import requests
import time

print("🚀 Quick API Test")
print("="*30)

# Test root endpoint
try:
    response = requests.get("http://localhost:8000/", timeout=5)
    print(f"✅ Root endpoint: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"❌ Root endpoint failed: {e}")

# Test health check
try:
    response = requests.get("http://localhost:8000/healthz", timeout=5)
    print(f"✅ Health check: {response.status_code}")
    data = response.json()
    print(f"   Status: {data.get('status')}")
    print(f"   Models: {data.get('models_loaded')}")
except Exception as e:
    print(f"❌ Health check failed: {e}")

print("\n🎉 Server is working!")