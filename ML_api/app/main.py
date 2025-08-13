import os
import base64
import io
import numpy as np
import csv
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import uvicorn
from contextlib import asynccontextmanager
import logging
import onnxruntime as ort

# Import helper functions
from util.helper import validate_image_file

# Import configuration and prediction service
from config import Config
from service import prediction_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)



# Enhanced Pydantic models for API responses
class Detection(BaseModel):
    """Single detection result"""
    class_name: str
    confidence: float
class DiseaseInfo(BaseModel):
    """Comprehensive disease information with prediction results"""
    # Core disease information
    disease_name: str
    description: str
    confidence: float = None  
    risk_level: str
    affected_area: str
    causes: list[str]
    symptoms: list[str]
    treatment: list[str]
    prevention: list[str]
    recommendations: list[str]
    
    # Disease classification
    disease_profile: dict

class PredictionResponse(BaseModel):
    """Complete API response"""
    filename: str
    predictions: list[Detection]
    image: str = None  # Base64 encoded annotated image
    disease_info: list[DiseaseInfo]
    processing_time: float
    status: str
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_status: str
    version: str
    timestamp: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - prediction service is already initialized
    logger.info("Starting Rice Disease Classification API")
    yield
    # Shutdown
    logger.info("Shutting down Rice Disease Classification API")

app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CSV-based disease information loading
def load_disease_info_from_csv(disease_name: str) -> dict:
    """
    Load disease information from CSV file.
    Returns a dictionary with disease details or None if not found.
    """
    try:
        csv_path = os.path.join(os.path.dirname(__file__), 'util', 'disease_details.csv')
        
        if not os.path.exists(csv_path):
            logger.warning(f"Disease details CSV not found: {csv_path}")
            return None
        
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                if row['Disease Name'].lower() == disease_name.lower():
                    return {
                        "description": row['Description'],
                        "riskLevel": row['Risk Level'],
                        "affectedArea": row['Affected Area'],
                        "keySymptoms": [s.strip() for s in row['Key Symptoms'].split(';')],
                        "causes": [c.strip() for c in row['Causes'].split(';')],
                        "treatment": [t.strip() for t in row['Treatment'].split(';')],
                        "prevention": [p.strip() for p in row['Prevention'].split(';')],
                        "diseaseProfile": {
                            "class": row['Disease Class'],
                            "type": row['Disease Type'],
                            "transmission": row['Transmission'],
                            "conditions": row['Conditions']
                        },
                        "recommendations": [r.strip() for r in row['Recommendations'].split(';')]
                    }
        
        logger.warning(f"Disease not found in CSV: {disease_name}")
        return None
        
    except Exception as e:
        logger.error(f"Error loading disease info for {disease_name}: {e}")
        return None

# Cache for disease information to avoid repeated CSV reads
_disease_cache = {}

def get_disease_info(disease_name: str) -> dict:
    """
    Get disease information with caching for performance.
    """
    if disease_name not in _disease_cache:
        _disease_cache[disease_name] = load_disease_info_from_csv(disease_name)
    return _disease_cache[disease_name]







# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "An internal server error occurred",
            "detail": str(exc) if app.debug else "Internal server error",
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }
    )

@app.get("/")
async def home():
    """
    Root endpoint with comprehensive API information.
    """
    return {
        "message": "Welcome to the Rice Disease Classification API!",
        "description": "AI-powered API for detecting and classifying rice diseases using InceptionV3 deep learning model",
        "version": Config.API_VERSION,
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "model_status": "loaded" if prediction_service.model_loaded else "not loaded",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + 'Z'
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    API endpoint to receive an image file, preprocess it,
    make a prediction using the InceptionV3 model, and return the result.
    
    Supports: JPEG, PNG, JPG, BMP, TIFF image formats
    """
    # Check if prediction service is ready
    if not prediction_service.model_loaded:
        raise HTTPException(
            status_code=500, 
            detail="Prediction service not ready. Please check server logs for errors during startup."
        )

    # Validate file using utility function
    validate_image_file(file)
    
    try:
        # Read the uploaded file
        image_bytes = await file.read()
        original_img_pil = Image.open(io.BytesIO(image_bytes))

        # Use prediction service for the complete prediction pipeline
        prediction_result = prediction_service.predict_single_image(
            image=original_img_pil,
            filename=file.filename or "unknown"
        )
        
        # Check if prediction failed
        if "error" in prediction_result:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {prediction_result['error']}"
            )
        
        # Get disease information from CSV
        disease_info = get_disease_info(prediction_result["top_prediction"]["class_name"])
        if not disease_info:
            disease_info = {
                "description": "No detailed description available.",
                "riskLevel": "Unknown",
                "affectedArea": "Unknown",
                "keySymptoms": ["No symptoms listed."],
                "causes": ["Unknown causes"],
                "treatment": ["No specific treatment available"],
                "prevention": ["No specific prevention available"],
                "diseaseProfile": {"class": "Unknown", "type": "Unknown", "transmission": "Unknown", "conditions": "Unknown"},
                "recommendations": ["No specific recommendations available."]
            }

        # Create response using the prediction service result
        response = PredictionResponse(
            filename=file.filename,
            predictions=[
                Detection(
                    class_name=pred["class_name"],
                    confidence=pred["confidence"]
                )
                for pred in prediction_result["predictions"]
            ],
            disease_info=[
                DiseaseInfo(
                    disease_name=prediction_result["top_prediction"]["class_name"],
                    description=disease_info["description"],
                    confidence=prediction_result["top_prediction"]["confidence"],
                    causes=disease_info["causes"],
                    symptoms=disease_info["keySymptoms"],
                    treatment=disease_info["treatment"],
                    prevention=disease_info["prevention"],
                    risk_level=disease_info["riskLevel"],
                    affected_area=disease_info["affectedArea"],
                    recommendations=disease_info["recommendations"],
                    disease_profile=disease_info["diseaseProfile"]
                )
            ],
            image=prediction_result["annotated_image"],
            processing_time=prediction_result["processing_time"],
            status="success",
            timestamp=prediction_result["timestamp"]
        )

        return response

    except Exception as e:
        logger.error(f"Error during prediction for {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred: {str(e)}"
        )

@app.get("/diseases")
async def get_diseases():
    """
    Get information about all supported diseases.
    """
    return {
        "diseases": Config.CLASS_NAMES,
        "total_count": len(Config.CLASS_NAMES)
    }



 

if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT) 