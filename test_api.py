#!/usr/bin/env python3
"""
Simple API test script to verify the Collision Parts Prediction service is working.
"""

import requests
import json
import base64
from PIL import Image
import io
import time

def test_root_endpoint():
    """Test the root endpoint."""
    print("Testing root endpoint...")
    try:
        response = requests.get("http://localhost:8000/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_health_check():
    """Test the health check endpoint."""
    print("\nTesting health check...")
    try:
        response = requests.get("http://localhost:8000/healthz")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_stats():
    """Test the stats endpoint."""
    print("\nTesting stats endpoint...")
    try:
        response = requests.get("http://localhost:8000/stats")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def create_test_image():
    """Create a simple test image."""
    # Create a simple colored image
    img = Image.new('RGB', (224, 224), color='red')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64

def test_prediction():
    """Test the prediction endpoint."""
    print("\nTesting prediction endpoint...")
    try:
        # Create test image
        img_base64 = create_test_image()
        
        # Prepare request
        payload = {
            "image": {
                "data": img_base64,
                "format": "base64"
            },
            "include_details": True,
            "include_timing": True,
            "confidence_threshold": 0.5
        }
        
        headers = {"Content-Type": "application/json"}
        
        print("Sending prediction request...")
        response = requests.post(
            "http://localhost:8000/predict", 
            data=json.dumps(payload),
            headers=headers,
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Prediction successful!")
            print(f"Predicted classes: {result.get('predicted_classes', [])}")
            print(f"Model versions: {result.get('model_versions', {})}")
            print(f"Processing time: {result.get('inference_times', {}).get('total_time', 'N/A')}")
            return True
        else:
            print(f"Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Collision Parts Prediction API Test")
    print("=" * 50)
    
    # Wait a moment for server to start
    print("Waiting for server to start...")
    time.sleep(2)
    
    tests = [
        ("Root Endpoint", test_root_endpoint),
        ("Health Check", test_health_check),
        ("Stats", test_stats),
        ("Prediction", test_prediction)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-' * 20}")
        success = test_func()
        results.append((test_name, success))
        print(f"{test_name}: {'✅ PASS' if success else '❌ FAIL'}")
    
    print(f"\n{'=' * 50}")
    print("Test Summary:")
    print(f"{'=' * 50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the server logs.")

if __name__ == "__main__":
    main()