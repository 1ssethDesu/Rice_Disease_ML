#!/usr/bin/env python3
"""
Configuration settings for Rice Disease Classification API
Centralized configuration management for better organization
"""

import os
from typing import Tuple

class Config:
    """Application configuration."""
    
    # API Configuration
    API_TITLE = "Rice Disease Classification API"
    API_DESCRIPTION = "API for classifying rice diseases using InceptionV3 model"
    API_VERSION = "1.0.0"
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8001))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Model Configuration
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'InceptionV3_Model.h5')
    ONNX_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'rice_disease_model.onnx')
    USE_ONNX = os.getenv("USE_ONNX", "true").lower() == "true"  # Default to ONNX now that it's available
    TARGET_IMAGE_SIZE: Tuple[int, int] = (299, 299)
    
    # File Upload Configuration
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Batch Processing Configuration
    MAX_BATCH_SIZE = 10  # Maximum number of files for batch prediction
    
    # Disease Classification
    CLASS_NAMES = [
        "Bacterial Leaf Blight",
        "Brown Spot",
        "Healthy",
        "Leaf Blast",
        "Leaf Scald",
        "Narrow Brown Spot"
    ]
    
    # Image Processing Configuration
    MAX_DISPLAY_DIMENSION = 800  # Maximum width/height for annotated images
    JPEG_QUALITY = 85  # Quality for JPEG compression in annotated images 