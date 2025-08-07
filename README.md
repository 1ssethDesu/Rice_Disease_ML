# Rice Disease Classification ML Model

A comprehensive machine learning solution for classifying rice diseases using deep learning with FastAPI web service and Grad-CAM visualization.

## 🌾 Project Overview

This project provides an end-to-end solution for rice disease classification using:
- **InceptionV3** deep learning model
- **FastAPI** web service for real-time predictions
- **Grad-CAM** visualization for explainable AI
- **Comprehensive disease information** with treatment recommendations

## 🚀 Features

### Core Functionality
- **6 Disease Classes**: Bacterial Leaf Blight, Brown Spot, Healthy, Leaf Blast, Leaf Scald, Narrow Brown Spot
- **Real-time Prediction**: Fast API endpoints for instant disease classification
- **Explainable AI**: Grad-CAM heatmaps showing model attention areas
- **File Upload Support**: Direct image upload (JPEG, PNG, JPG)
- **Base64 Support**: Legacy support for base64 encoded images

### Technical Features
- **FastAPI Framework**: Modern, fast web framework with automatic documentation
- **Type Safety**: Pydantic models for request/response validation
- **Health Monitoring**: Built-in health check endpoints
- **Interactive Documentation**: Swagger UI at `/docs`
- **Comprehensive Testing**: Test scripts for all endpoints

## 📁 Project Structure

```
Rice_disease_ML_Model/
├── ML_api/                          # FastAPI application
│   ├── main.py                      # Main FastAPI application
│   ├── test_api.py                  # Test scripts
│   ├── README.md                    # API documentation
│   └── model/                       # Model directory
│       └── InceptionV3 Best Model.h5 # Trained model file
├── tests/                           # Test dataset
│   └── Rice_Diseases/              # Disease image samples
│       ├── Bacterial Blight Disease/
│       ├── Blast Disease/
│       ├── Brown Spot Disease/
│       └── False Smut Disease/
├── requirements.txt                 # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Rice_disease_ML_Model
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model file**:
   Ensure the model file exists at:
   ```
   ML_api/model/InceptionV3 Best Model.h5
   ```

## 🚀 Quick Start

### Start the API Server
```bash
cd ML_api
python main.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Test the API
```bash
cd ML_api
python test_api.py
```

## 📚 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | Health check |
| `/diseases` | GET | List all diseases |
| `/diseases/{name}` | GET | Get disease details |
| `/predict` | POST | File upload prediction |
| `/predict-base64` | POST | Base64 prediction |

### Example Usage

#### File Upload (Recommended)
```python
import requests

with open("rice_leaf.jpg", "rb") as image_file:
    files = {"file": ("rice_leaf.jpg", image_file, "image/jpeg")}
    response = requests.post("http://localhost:8000/predict", files=files)
    
result = response.json()
print(f"Disease: {result['results'][0]['diseaseName']}")
print(f"Confidence: {result['results'][0]['confidence']:.2%}")
```

#### cURL Example
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@rice_leaf.jpg"
```

## 🧪 Testing

The project includes comprehensive testing:

1. **Health Check**: Verify server and model status
2. **Disease Information**: Test disease lookup endpoints
3. **File Upload Prediction**: Test with sample images
4. **Base64 Prediction**: Test legacy endpoint
5. **Grad-CAM Visualization**: Save heatmap results

Run tests:
```bash
cd ML_api
python test_api.py
```

## 📊 Model Information

- **Architecture**: InceptionV3
- **Input Size**: 224x224 pixels
- **Preprocessing**: Normalized to [-1, 1] range
- **Output Classes**: 6 disease categories
- **Grad-CAM Layer**: `conv2d_281`

## 🏥 Supported Diseases

### 1. Bacterial Leaf Blight
- **Risk Level**: High
- **Type**: Bacterial disease
- **Symptoms**: Water-soaked lesions, yellowing leaves
- **Treatment**: Resistant varieties, copper-based bactericides

### 2. Brown Spot
- **Risk Level**: Moderate
- **Type**: Fungal disease
- **Symptoms**: Oval spots with brown margins
- **Treatment**: Fungicides, balanced fertilization

### 3. Healthy
- **Risk Level**: Low
- **Type**: No disease
- **Symptoms**: Vibrant green, smooth texture
- **Treatment**: Continue good practices

### 4. Leaf Blast
- **Risk Level**: High
- **Type**: Fungal disease
- **Symptoms**: Spindle-shaped lesions, neck rot
- **Treatment**: Resistant varieties, proper drainage

### 5. Leaf Scald
- **Risk Level**: Moderate
- **Type**: Fungal disease
- **Symptoms**: Irregular lesions with concentric rings
- **Treatment**: Fungicides, debris removal

### 6. Narrow Brown Spot
- **Risk Level**: Low to Moderate
- **Type**: Fungal disease
- **Symptoms**: Small linear brown spots
- **Treatment**: Balanced fertilization, resistant varieties

## 🔧 Development

### Adding New Diseases
1. Update `CLASS_NAMES` in `main.py`
2. Add disease details to `DISEASE_DETAILS`
3. Retrain model with new classes
4. Update model file

### Customizing Grad-CAM
1. Inspect model: `model.summary()`
2. Find last convolutional layer
3. Update `LAST_CONV_LAYER_NAME`

## 🐛 Troubleshooting

### Common Issues
- **Model Not Found**: Check model file path
- **Memory Issues**: Reduce image size or batch size
- **Port Conflicts**: Change port in `main.py`
- **CUDA Errors**: Check TensorFlow GPU compatibility

### Performance
- **Model Loading**: 2-5 seconds
- **Prediction Time**: 1-3 seconds
- **Grad-CAM Generation**: 0.5-1 second

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Open an issue on GitHub

---

**Note**: This project requires the trained model file (`InceptionV3 Best Model.h5`) to function. Ensure the model file is present in the `ML_api/model/` directory. 