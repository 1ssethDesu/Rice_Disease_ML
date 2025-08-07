# Rice Disease Classification FastAPI

A FastAPI implementation for classifying rice diseases using an InceptionV3 model with Grad-CAM visualization.

## Features

- **FastAPI Framework**: Modern, fast web framework with automatic API documentation
- **InceptionV3 Model**: Pre-trained deep learning model for rice disease classification
- **Grad-CAM Visualization**: Explainable AI with heatmap overlays showing model attention
- **Comprehensive Disease Information**: Detailed descriptions, symptoms, and treatment recommendations
- **Type Safety**: Pydantic models for request/response validation
- **Health Monitoring**: Built-in health check endpoints
- **Interactive Documentation**: Automatic Swagger UI at `/docs`

## Supported Diseases

1. **Bacterial Leaf Blight** - High risk bacterial disease
2. **Brown Spot** - Moderate risk fungal disease  
3. **Healthy** - No disease detected
4. **Leaf Blast** - High risk fungal disease
5. **Leaf Scald** - Moderate risk fungal disease
6. **Narrow Brown Spot** - Low to moderate risk fungal disease

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Model File**:
   Ensure the model file exists at:
   ```
   ML_api/model/InceptionV3 Best Model.h5
   ```

## Usage

### Starting the Server

```bash
cd ML_api
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### API Endpoints

#### 1. Health Check
```http
GET /health
```
Returns server status and model loading status.

#### 2. Home
```http
GET /
```
Welcome message and basic information.

#### 3. Get All Diseases
```http
GET /diseases
```
Returns list of all supported diseases.

#### 4. Get Disease Information
```http
GET /diseases/{disease_name}
```
Returns detailed information about a specific disease.

#### 5. Predict Disease (File Upload - Recommended)
```http
POST /predict
```

**Request**: Multipart form data with image file
- **file**: Image file (JPEG, PNG, JPG)

#### 6. Predict Disease (Base64 - Legacy)
```http
POST /predict-base64
```

**Request Body**:
```json
{
  "image_base64": "base64_encoded_image_string"
}
```

**Response**:
```json
{
  "status": "success",
  "timestamp": "2024-01-01T12:00:00Z",
  "results": [
    {
      "diseaseName": "Bacterial Leaf Blight",
      "description": "Water-soaked lesions on leaves...",
      "confidence": 0.95,
      "riskLevel": "High",
      "affectedArea": "Leaf tips and margins...",
      "imgUrl": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
      "keySymptoms": [
        "Yellowing and wilting of leaves",
        "Lesions with water-soaked appearance"
      ],
      "diseaseProfile": {
        "class": "Bacterial",
        "type": "Foliar disease",
        "transmission": "Rain splash, irrigation water...",
        "conditions": "High humidity, warm temperatures..."
      },
      "recommendations": [
        "Use resistant rice varieties",
        "Avoid overhead irrigation..."
      ]
    }
  ]
}
```

## Testing

Run the test script to verify the API functionality:

```bash
python test_api.py
```

This will:
1. Test the health check endpoint
2. Test the diseases endpoint
3. Test getting specific disease information
4. Test the file upload prediction endpoint with a sample image
5. Test the base64 prediction endpoint with a sample image
6. Save the Grad-CAM visualizations as `grad_cam_result_file_upload.png` and `grad_cam_result_base64.png`

## Example Usage with Python

### File Upload (Recommended)
```python
import requests

# Upload image file
with open("path/to/your/image.jpg", "rb") as image_file:
    files = {"file": ("image.jpg", image_file, "image/jpeg")}
    response = requests.post("http://localhost:8000/predict", files=files)

if response.status_code == 200:
    result = response.json()
    disease = result['results'][0]['diseaseName']
    confidence = result['results'][0]['confidence']
    print(f"Predicted: {disease} with {confidence:.2%} confidence")
else:
    print(f"Error: {response.status_code}")
```

### Base64 (Legacy)
```python
import requests
import base64

# Encode image to base64
with open("path/to/your/image.jpg", "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

# Make prediction request
response = requests.post(
    "http://localhost:8000/predict-base64",
    json={"image_base64": image_base64}
)

if response.status_code == 200:
    result = response.json()
    disease = result['results'][0]['diseaseName']
    confidence = result['results'][0]['confidence']
    print(f"Predicted: {disease} with {confidence:.2%} confidence")
else:
    print(f"Error: {response.status_code}")
```

## Example Usage with cURL

```bash
# Health check
curl http://localhost:8000/health

# Get all diseases
curl http://localhost:8000/diseases

# Get specific disease info
curl http://localhost:8000/diseases/Bacterial%20Leaf%20Blight

# File upload prediction (recommended)
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/your/image.jpg"

# Base64 prediction (legacy)
curl -X POST http://localhost:8000/predict-base64 \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "your_base64_image_string_here"}'
```

## Model Information

- **Architecture**: InceptionV3
- **Input Size**: 224x224 pixels
- **Preprocessing**: Normalized to [-1, 1] range
- **Output**: 6 disease classes
- **Grad-CAM Layer**: `conv2d_281` (last convolutional layer)

## Error Handling

The API includes comprehensive error handling:
- **400 Bad Request**: Invalid input data
- **404 Not Found**: Disease not found
- **500 Internal Server Error**: Model loading or prediction errors

## Performance

- **Model Loading**: ~2-5 seconds on startup
- **Prediction Time**: ~1-3 seconds per image
- **Grad-CAM Generation**: ~0.5-1 second additional

## Development

### Adding New Diseases

1. Update `CLASS_NAMES` list in `main.py`
2. Add disease details to `DISEASE_DETAILS` dictionary
3. Retrain the model with the new classes
4. Update the model file

### Customizing Grad-CAM

The Grad-CAM layer name can be customized by:
1. Inspecting your model architecture: `model.summary()`
2. Finding the last convolutional layer name
3. Updating `LAST_CONV_LAYER_NAME` in the predict function

## Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure the model file exists at the specified path
2. **Memory Issues**: Reduce batch size or use smaller images
3. **CUDA Errors**: Ensure TensorFlow GPU compatibility
4. **Port Already in Use**: Change the port in `main.py` or kill existing processes

### Logs

Check the console output for detailed error messages and model loading status.

## License

This project is for educational and research purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request 