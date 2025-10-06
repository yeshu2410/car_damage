import requests
import base64
import json
from PIL import Image
import io

# Create a proper test image
img = Image.new('RGB', (224, 224), color='red')
buffer = io.BytesIO()
img.save(buffer, format='JPEG')
img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Test with proper payload
payload = {
    "image": {
        "image_data": img_base64,
        "image_format": "JPEG"
    },
    "include_details": True,
    "include_timing": True
}

try:
    print("Sending prediction request...")
    r = requests.post("http://127.0.0.1:8000/predict", json=payload)
    print("Status:", r.status_code)
    
    if r.status_code == 200:
        result = r.json()
        print("SUCCESS! Prediction completed.")
        print("Predicted Classes:", result.get('predicted_classes', []))
        print("Model Versions:", result.get('model_versions', {}))
        print("Request ID:", result.get('request_id'))
        if 'inference_times' in result:
            print("Total Time:", result['inference_times'].get('total_time', 'N/A'))
    else:
        print("Error Response:", r.text)
        
except Exception as e:
    print("Exception occurred:", e)