# Rice Disease Classification API

A FastAPI-based REST API for classifying rice diseases using deep learning with the InceptionV3 model. This API provides real-time disease detection, comprehensive disease information, and detailed treatment recommendations.

## 🌾 Features

- **AI-Powered Classification**: Uses InceptionV3 deep learning model for accurate rice disease detection
- **Multiple Disease Support**: Classifies 6 different rice conditions including healthy plants
- **Comprehensive Information**: Provides detailed disease descriptions, symptoms, causes, treatment, and prevention
- **Image Annotation**: Returns annotated images with prediction results
- **Real-time Processing**: Fast inference with processing time tracking
- **Production Ready**: Includes comprehensive error handling, validation, and logging
- **API Documentation**: Auto-generated interactive documentation

## 🚀 Supported Diseases

1. **Bacterial Leaf Blight** - High risk bacterial disease
2. **Brown Spot** - Moderate risk fungal disease  
3. **Healthy** - No disease detected
4. **Leaf Blast** - High risk fungal disease
5. **Leaf Scald** - Moderate risk bacterial disease
6. **Narrow Brown Spot** - Low to moderate risk fungal disease

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python)
- **ML Framework**: TensorFlow/Keras
- **Model**: InceptionV3 (pre-trained and fine-tuned)
- **Image Processing**: OpenCV, Pillow
- **Validation**: Pydantic
- **Server**: Uvicorn

## 📋 Prerequisites

- Python 3.8+
- TensorFlow 2.15.0+
- OpenCV 4.8+
- FastAPI 0.104+

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Rice_disease_ML_Model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Model
Place your trained InceptionV3 model file at:
```
ML_api/model/InceptionV3_Model.h5
```

### 4. Run the API
```bash
cd ML_api
python main.py
```

The API will be available at `http://localhost:8001`

## 🔧 Configuration

The API can be configured using environment variables:

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8001
DEBUG=false

# Logging
LOG_LEVEL=INFO

# Model Configuration
MODEL_PATH=./model/InceptionV3_Model.h5
```

## 📚 API Endpoints

### Root Endpoint
- **GET** `/` - API information and documentation

### Health Check
- **GET** `/health` - Health status and model loading status

### Disease Classification
- **POST** `/predict` - Upload image for disease classification
  - **Input**: Image file (JPEG, PNG, JPG, BMP, TIFF)
  - **Max Size**: 10MB
  - **Response**: Classification results, confidence scores, disease information

### Disease Information
- **GET** `/diseases` - List all supported diseases
- **GET** `/diseases/{disease_name}` - Get detailed information about a specific disease

### Documentation
- **GET** `/docs` - Interactive API documentation (Swagger UI)
- **GET** `/redoc` - Alternative API documentation

## 📖 Usage Examples

### Python Client Example
```python
import requests

# Upload image for classification
with open('rice_leaf.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8001/predict', files=files)
    
result = response.json()
print(f"Disease: {result['predictions'][0]['class_name']}")
print(f"Confidence: {result['predictions'][0]['confidence']:.2%}")
```

### cURL Example
```bash
curl -X POST "http://localhost:8001/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@rice_leaf.jpg"
```

## 📊 Response Format

### Prediction Response
```json
{
  "filename": "rice_leaf.jpg",
  "predictions": [
    {
      "class_name": "Bacterial Leaf Blight",
      "confidence": 0.95
    }
  ],
  "image": "data:image/jpeg;base64,...",
  "disease_info": [
    {
      "description": "Water-soaked lesions on leaves...",
      "causes": ["Xanthomonas oryzae pv. oryzae bacteria", ...],
      "symptoms": ["Yellowing and wilting of leaves", ...],
      "treatment": ["Remove and destroy infected plants", ...],
      "prevention": ["Use resistant rice varieties", ...],
      "risk_level": "High",
      "affected_area": "Leaf tips and margins",
      "disease_profile": {...}
    }
  ],
  "processing_time": 0.85,
  "status": "success",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🧪 Testing

Run the test suite:
```bash
cd ML_api
python -m pytest tests/
```

## 📈 Performance

- **Model Loading**: ~2-5 seconds (first startup)
- **Inference Time**: ~0.5-2 seconds per image
- **Throughput**: 10-50 images per minute (depending on hardware)
- **Memory Usage**: ~2-4GB RAM (including model)

## 🔒 Security Features

- File type validation
- File size limits
- Input sanitization
- Comprehensive error handling
- CORS configuration

## 🚀 Deployment

### Production Deployment
```bash
# Using Gunicorn with Uvicorn workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001

# Using Docker
docker build -t rice-disease-api .
docker run -p 8001:8001 rice-disease-api
```

### Environment Variables for Production
```bash
DEBUG=false
LOG_LEVEL=WARNING
HOST=0.0.0.0
PORT=8001
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow/Keras for the deep learning framework
- FastAPI for the web framework
- The rice disease research community for datasets and knowledge

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the health endpoint at `/health`

## 🔄 Version History

- **v1.0.0** - Initial release with InceptionV3 model
- Support for 6 rice disease classifications
- Comprehensive API endpoints
- Production-ready error handling and validation 