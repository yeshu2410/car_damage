import requests
import json

# Test with minimal payload first
payload = {
    "image": {
        "image_data": "test", 
        "image_format": "JPEG"
    },
    "include_details": False
}

try:
    r = requests.post("http://127.0.0.1:8000/predict", json=payload)
    print("Status:", r.status_code)
    print("Response:", r.text)
except Exception as e:
    print("Error:", e)