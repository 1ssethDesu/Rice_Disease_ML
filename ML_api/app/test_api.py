#!/usr/bin/env python3
"""
Test script for the Rice Disease Classification API
"""

import requests
import json
import time
from pathlib import Path

# API Configuration
BASE_URL = "http://localhost:8001"
API_ENDPOINTS = {
    "root": "/",
    "health": "/health",
    "diseases": "/diseases",
    "predict": "/predict"
}

def test_root_endpoint():
    """Test the root endpoint."""
    print("🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}{API_ENDPOINTS['root']}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint working")
            print(f"   API Version: {data.get('version', 'N/A')}")
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Available Diseases: {len(data.get('available_diseases', []))}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing root endpoint: {e}")

def test_health_endpoint():
    """Test the health endpoint."""
    print("\n🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}{API_ENDPOINTS['health']}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health endpoint working")
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Model Status: {data.get('model_status', 'N/A')}")
            print(f"   Version: {data.get('version', 'N/A')}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing health endpoint: {e}")

def test_diseases_endpoint():
    """Test the diseases endpoint."""
    print("\n🔍 Testing diseases endpoint...")
    try:
        response = requests.get(f"{BASE_URL}{API_ENDPOINTS['diseases']}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Diseases endpoint working")
            print(f"   Total Diseases: {data.get('total_count', 'N/A')}")
            print(f"   Diseases: {', '.join(data.get('diseases', []))}")
        else:
            print(f"❌ Diseases endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing diseases endpoint: {e}")

def test_disease_info_endpoint():
    """Test the disease info endpoint."""
    print("\n🔍 Testing disease info endpoint...")
    try:
        # Test with a known disease
        disease_name = "Bacterial Leaf Blight"
        response = requests.get(f"{BASE_URL}/diseases/{disease_name}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Disease info endpoint working")
            print(f"   Disease: {data.get('disease_name', 'N/A')}")
            print(f"   Risk Level: {data.get('details', {}).get('risk_level', 'N/A')}")
        else:
            print(f"❌ Disease info endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing disease info endpoint: {e}")

def test_predict_endpoint():
    """Test the predict endpoint with a sample image."""
    print("\n🔍 Testing predict endpoint...")
    
    # Check if test image exists
    test_image_path = Path("test_image.jpg")
    if not test_image_path.exists():
        print("⚠️  No test image found. Create a 'test_image.jpg' file to test prediction.")
        print("   You can use any rice leaf image for testing.")
        return
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            start_time = time.time()
            response = requests.post(f"{BASE_URL}{API_ENDPOINTS['predict']}", files=files)
            end_time = time.time()
            
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Predict endpoint working")
            print(f"   Filename: {data.get('filename', 'N/A')}")
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Processing Time: {data.get('processing_time', 'N/A')}s")
            print(f"   Actual Request Time: {end_time - start_time:.2f}s")
            
            if data.get('predictions'):
                prediction = data['predictions'][0]
                print(f"   Predicted Disease: {prediction.get('class_name', 'N/A')}")
                print(f"   Confidence: {prediction.get('confidence', 'N/A'):.2%}")
            
            if data.get('disease_info'):
                disease = data['disease_info'][0]
                print(f"   Risk Level: {disease.get('risk_level', 'N/A')}")
                print(f"   Affected Area: {disease.get('affected_area', 'N/A')}")
        else:
            print(f"❌ Predict endpoint failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Error testing predict endpoint: {e}")

def test_invalid_requests():
    """Test invalid requests for error handling."""
    print("\n🔍 Testing error handling...")
    
    # Test invalid disease name
    try:
        response = requests.get(f"{BASE_URL}/diseases/InvalidDisease")
        if response.status_code == 404:
            print("✅ Invalid disease name handling working")
        else:
            print(f"❌ Invalid disease name handling failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing invalid disease name: {e}")
    
    # Test predict without file
    try:
        response = requests.post(f"{BASE_URL}{API_ENDPOINTS['predict']}")
        if response.status_code == 422:  # Validation error
            print("✅ Missing file validation working")
        else:
            print(f"❌ Missing file validation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing missing file: {e}")

def main():
    """Run all tests."""
    print("🚀 Starting Rice Disease Classification API Tests")
    print("=" * 50)
    
    # Test all endpoints
    test_root_endpoint()
    test_health_endpoint()
    test_diseases_endpoint()
    test_disease_info_endpoint()
    test_predict_endpoint()
    test_invalid_requests()
    
    print("\n" + "=" * 50)
    print("✨ Testing completed!")
    print("\n💡 Tips:")
    print("   - Make sure the API is running on localhost:8001")
    print("   - Place a test image named 'test_image.jpg' in this directory")
    print("   - Check the API documentation at http://localhost:8001/docs")

if __name__ == "__main__":
    main() 