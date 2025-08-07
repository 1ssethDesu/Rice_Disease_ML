import requests
import base64
import json
from PIL import Image
import io

# API endpoint
API_URL = "http://localhost:8000"

def encode_image_to_base64(image_path):
    """Convert image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_file_upload_prediction():
    """Test the file upload prediction endpoint with a sample image."""
    
    # Test with a sample image from the test dataset
    test_image_path = "tests/Rice_Diseases/Bacterial Blight Disease/BB (1).jpg"
    
    try:
        # Prepare the file upload
        with open(test_image_path, "rb") as image_file:
            files = {"file": ("test_image.jpg", image_file, "image/jpeg")}
            
            # Make the prediction request
            print(f"Making file upload prediction request to {API_URL}/predict...")
            response = requests.post(f"{API_URL}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ File upload prediction successful!")
            print(f"Disease: {result['results'][0]['diseaseName']}")
            print(f"Confidence: {result['results'][0]['confidence']:.2%}")
            print(f"Risk Level: {result['results'][0]['riskLevel']}")
            print(f"Description: {result['results'][0]['description']}")
            print(f"Affected Area: {result['results'][0]['affectedArea']}")
            print(f"Key Symptoms: {', '.join(result['results'][0]['keySymptoms'])}")
            print(f"Recommendations: {', '.join(result['results'][0]['recommendations'])}")
            
            # Save the Grad-CAM image
            img_data = result['results'][0]['imgUrl'].split(',')[1]
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes))
            img.save("grad_cam_result_file_upload.png")
            print("📸 Grad-CAM visualization saved as 'grad_cam_result_file_upload.png'")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error during file upload testing: {e}")

def test_base64_prediction():
    """Test the base64 prediction endpoint with a sample image."""
    
    # Test with a sample image from the test dataset
    test_image_path = "tests/Rice_Diseases/Bacterial Blight Disease/BB (1).jpg"
    
    try:
        # Encode the image
        image_base64 = encode_image_to_base64(test_image_path)
        
        # Prepare the request
        payload = {
            "image_base64": image_base64
        }
        
        # Make the prediction request
        print(f"Making base64 prediction request to {API_URL}/predict-base64...")
        response = requests.post(f"{API_URL}/predict-base64", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Base64 prediction successful!")
            print(f"Disease: {result['results'][0]['diseaseName']}")
            print(f"Confidence: {result['results'][0]['confidence']:.2%}")
            print(f"Risk Level: {result['results'][0]['riskLevel']}")
            print(f"Description: {result['results'][0]['description']}")
            print(f"Affected Area: {result['results'][0]['affectedArea']}")
            print(f"Key Symptoms: {', '.join(result['results'][0]['keySymptoms'])}")
            print(f"Recommendations: {', '.join(result['results'][0]['recommendations'])}")
            
            # Save the Grad-CAM image
            img_data = result['results'][0]['imgUrl'].split(',')[1]
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes))
            img.save("grad_cam_result_base64.png")
            print("📸 Grad-CAM visualization saved as 'grad_cam_result_base64.png'")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error during base64 testing: {e}")

def test_health_check():
    """Test the health check endpoint."""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check successful!")
            print(f"Status: {result['status']}")
            print(f"Model loaded: {result['model_loaded']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error during health check: {e}")

def test_diseases_endpoint():
    """Test the diseases endpoint."""
    try:
        response = requests.get(f"{API_URL}/diseases")
        if response.status_code == 200:
            result = response.json()
            print("✅ Diseases endpoint successful!")
            print(f"Available diseases: {result['diseases']}")
            print(f"Total count: {result['total_count']}")
        else:
            print(f"❌ Diseases endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error during diseases test: {e}")

def test_specific_disease():
    """Test getting information about a specific disease."""
    try:
        disease_name = "Bacterial Leaf Blight"
        response = requests.get(f"{API_URL}/diseases/{disease_name}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Disease info for '{disease_name}' successful!")
            print(f"Description: {result['details']['description']}")
            print(f"Risk Level: {result['details']['riskLevel']}")
        else:
            print(f"❌ Disease info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error during specific disease test: {e}")

if __name__ == "__main__":
    print("🚀 Testing Rice Disease Classification FastAPI")
    print("=" * 50)
    
    # Test health check first
    print("\n1. Testing health check...")
    test_health_check()
    
    # Test diseases endpoint
    print("\n2. Testing diseases endpoint...")
    test_diseases_endpoint()
    
    # Test specific disease info
    print("\n3. Testing specific disease info...")
    test_specific_disease()
    
    # Test file upload prediction
    print("\n4. Testing file upload prediction...")
    test_file_upload_prediction()
    
    # Test base64 prediction
    print("\n5. Testing base64 prediction...")
    test_base64_prediction()
    
    print("\n" + "=" * 50)
    print("🏁 Testing completed!") 