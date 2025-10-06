#!/usr/bin/env python3
"""
Integration test that starts server and tests it
"""

import subprocess
import time
import requests
import threading
import sys
import os

def start_server():
    """Start the server in a subprocess"""
    print("🚀 Starting server...")
    
    # Start server process
    process = subprocess.Popen([
        sys.executable, "start_server.py", 
        "--host", "127.0.0.1", "--port", "8001"
    ], cwd=os.getcwd())
    
    return process

def test_server():
    """Test the server endpoints"""
    print("⏳ Waiting for server to start...")
    time.sleep(5)  # Give server time to start
    
    base_url = "http://127.0.0.1:8001"
    
    print("\n🧪 Testing API endpoints...")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        print(f"✅ Root endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Service: {data.get('service')}")
            print(f"   Status: {data.get('status')}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

    # Test health check  
    try:
        response = requests.get(f"{base_url}/healthz", timeout=10)
        print(f"✅ Health check: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Models loaded: {data.get('models_loaded')}")
            print(f"   Uptime: {data.get('uptime_seconds'):.2f}s")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

    # Test stats endpoint
    try:
        response = requests.get(f"{base_url}/stats", timeout=10)
        print(f"✅ Stats endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Total inferences: {data.get('total_inferences')}")
            print(f"   Device: {data.get('device')}")
            print(f"   Memory usage: {data.get('memory_usage_mb'):.1f} MB")
    except Exception as e:
        print(f"❌ Stats endpoint failed: {e}")
        return False

    return True

def main():
    """Main test function"""
    print("🔧 Collision Parts API Integration Test")
    print("=" * 50)
    
    # Start server
    server_process = start_server()
    
    try:
        # Test the server
        success = test_server()
        
        if success:
            print("\n🎉 All tests passed! API is working correctly.")
            print("\nThe server is running at: http://127.0.0.1:8001")
            print("API docs available at: http://127.0.0.1:8001/docs")
            print("\nPress Ctrl+C to stop the server.")
            
            # Keep server running
            server_process.wait()
        else:
            print("\n❌ Some tests failed.")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
    finally:
        # Clean up
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()

if __name__ == "__main__":
    main()