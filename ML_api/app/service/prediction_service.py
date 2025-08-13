#!/usr/bin/env python3
"""
Prediction Service for Rice Disease Classification API
Handles model management, image processing, detection processing, and performance monitoring
"""

import os
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
import onnxruntime as ort
from dataclasses import dataclass
from datetime import datetime

# Import configuration and helper functions
from config import Config
from util.helper import image_process, create_annotated_image

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Structured prediction result"""
    class_name: str
    confidence: float
    class_index: int
    processing_time: float
    timestamp: datetime

@dataclass
class ServiceMetrics:
    """Service performance metrics"""
    total_predictions: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    last_prediction_time: Optional[datetime] = None
    model_load_time: Optional[datetime] = None

class PredictionService:
    """
    Service class for handling ML model predictions and image processing
    """
    
    def __init__(self):
        """Initialize the prediction service"""
        self.keras_model: Optional[Model] = None
        self.onnx_session: Optional[ort.InferenceSession] = None
        self.model_type: str = "unknown"
        self.model_loaded: bool = False
        self.metrics = ServiceMetrics()
        
        # Load model on initialization
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the trained model (Keras or ONNX) and cache it for reuse.
        """
        try:
            logger.info(f"Loading model... USE_ONNX: {Config.USE_ONNX}")
            
            if Config.USE_ONNX and self._check_onnx_availability():
                logger.info(f"Loading ONNX model from: {Config.ONNX_MODEL_PATH}")
                self.onnx_session = ort.InferenceSession(Config.ONNX_MODEL_PATH)
                self.model_type = "onnx"
                self.model_loaded = True
                logger.info("ONNX model loaded successfully")
                
            else:
                if Config.USE_ONNX and not self._check_onnx_availability():
                    logger.warning("ONNX requested but not available, falling back to Keras model")
                
                logger.info(f"Loading Keras model from: {Config.MODEL_PATH}")
                self.keras_model = tf.keras.models.load_model(Config.MODEL_PATH)
                
                # Ensure model is in evaluation mode
                self.keras_model.trainable = False
                self.model_type = "keras"
                self.model_loaded = True
                logger.info("Keras model loaded successfully")
            
            self.metrics.model_load_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _check_onnx_availability(self) -> bool:
        """Check if ONNX runtime is available"""
        try:
            import onnxruntime as ort
            return True
        except ImportError:
            return False
    
    def _get_model(self) -> Tuple[Any, str]:
        """
        Get the cached model instance and type.
        
        Returns:
            Tuple of (model_instance, model_type)
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Please check service initialization.")
        
        if self.model_type == "onnx":
            return self.onnx_session, "onnx"
        else:
            return self.keras_model, "keras"
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for model input using the image processing pipeline.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image array ready for model input
        """
        try:
            # Use helper function for consistent image processing
            processed_image = image_process(
                image, 
                Config.TARGET_IMAGE_SIZE, 
                normalize=True
            )
            return processed_image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise RuntimeError(f"Image preprocessing failed: {e}")
    
    def _run_inference(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """
        Run model inference on preprocessed image.
        
        Args:
            preprocessed_image: Preprocessed image array
            
        Returns:
            Model predictions array
        """
        try:
            model, model_type = self._get_model()
            
            if model_type == "keras":
                # Ensure model is in evaluation mode
                model.trainable = False
                predictions = model.predict(preprocessed_image, verbose=0)
            else:  # ONNX
                input_name = model.get_inputs()[0].name
                predictions = model.run(None, {input_name: preprocessed_image})[0]
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            raise RuntimeError(f"Model inference failed: {e}")
    
    def _process_detections(self, predictions: np.ndarray) -> List[PredictionResult]:
        """
        Process model predictions and filter results.
        
        Args:
            predictions: Raw model predictions array
            
        Returns:
            List of processed prediction results
        """
        try:
            results = []
            
            # Get top predictions
            top_indices = np.argsort(predictions[0])[::-1]  # Sort by confidence descending
            
            for idx in top_indices:
                confidence = float(predictions[0][idx])
                class_name = Config.CLASS_NAMES[idx] if idx < len(Config.CLASS_NAMES) else "Unknown"
                
                result = PredictionResult(
                    class_name=class_name,
                    confidence=confidence,
                    class_index=idx,
                    processing_time=0.0,  # Will be set by caller
                    timestamp=datetime.utcnow()
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing detections: {e}")
            raise RuntimeError(f"Detection processing failed: {e}")
    
    def _encode_image_output(self, image: Image.Image, top_result: PredictionResult) -> Optional[str]:
        """
        Encode image with prediction annotations for output.
        
        Args:
            image: Original PIL Image
            top_result: Top prediction result for annotation
            
        Returns:
            Base64 encoded annotated image or None if failed
        """
        try:
            annotated_image = create_annotated_image(
                image=image,
                class_name=top_result.class_name,
                confidence=top_result.confidence,
                max_dim=Config.MAX_DISPLAY_DIMENSION,
                jpeg_quality=Config.JPEG_QUALITY
            )
            return annotated_image
            
        except Exception as e:
            logger.warning(f"Failed to create annotated image: {e}")
            return None
    
    def _update_metrics(self, processing_time: float) -> None:
        """
        Update service performance metrics.
        
        Args:
            processing_time: Time taken for prediction in seconds
        """
        self.metrics.total_predictions += 1
        self.metrics.total_processing_time += processing_time
        self.metrics.average_processing_time = (
            self.metrics.total_processing_time / self.metrics.total_predictions
        )
        self.metrics.last_prediction_time = datetime.utcnow()
    
    def predict_single_image(self, image: Image.Image, filename: str = "unknown") -> Dict[str, Any]:
        """
        Perform single image prediction with full processing pipeline.
        
        Args:
            image: PIL Image object to predict
            filename: Original filename for logging
            
        Returns:
            Dictionary containing prediction results and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting prediction for {filename}")
            
            # Image preprocessing
            preprocessed_image = self._preprocess_image(image)
            
            # Model inference
            predictions = self._run_inference(preprocessed_image)
            
            # Process detections
            detection_results = self._process_detections(predictions)
            
            # Get top result
            top_result = detection_results[0]
            
            # Calculate processing time
            processing_time = time.time() - start_time
            top_result.processing_time = processing_time
            
            # Update metrics
            self._update_metrics(processing_time)
            
            # Create annotated image
            annotated_image = self._encode_image_output(image, top_result)
            
            # Prepare response
            response = {
                "filename": filename,
                "predictions": [
                    {
                        "class_name": result.class_name,
                        "confidence": result.confidence,
                        "class_index": result.class_index
                    }
                    for result in detection_results[:3]  # Top 3 results
                ],
                "top_prediction": {
                    "class_name": top_result.class_name,
                    "confidence": top_result.confidence,
                    "class_index": top_result.class_index
                },
                "annotated_image": annotated_image,
                "processing_time": processing_time,
                "model_type": self.model_type,
                "timestamp": top_result.timestamp.isoformat() + 'Z'
            }
            
            logger.info(
                f"Prediction completed for {filename}: "
                f"{top_result.class_name} ({top_result.confidence:.2%}) "
                f"in {processing_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Prediction failed for {filename}: {e}")
            
            # Return error response
            return {
                "filename": filename,
                "error": str(e),
                "processing_time": processing_time,
                "timestamp": datetime.utcnow().isoformat() + 'Z'
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get current service status and metrics.
        
        Returns:
            Dictionary containing service status information
        """
        return {
            "model_loaded": self.model_loaded,
            "model_type": self.model_type,
            "model_load_time": self.metrics.model_load_time.isoformat() + 'Z' if self.metrics.model_load_time else None,
            "metrics": {
                "total_predictions": self.metrics.total_predictions,
                "total_processing_time": self.metrics.total_processing_time,
                "average_processing_time": self.metrics.average_processing_time,
                "last_prediction_time": self.metrics.last_prediction_time.isoformat() + 'Z' if self.metrics.last_prediction_time else None
            },
            "config": {
                "target_image_size": Config.TARGET_IMAGE_SIZE,
                "max_file_size": Config.MAX_FILE_SIZE,
                "allowed_extensions": list(Config.ALLOWED_EXTENSIONS),
                "class_names": Config.CLASS_NAMES
            }
        }
    
    def reload_model(self) -> bool:
        """
        Reload the model (useful for model updates).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Reloading model...")
            self._load_model()
            return self.model_loaded
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
            return False

# Global service instance
prediction_service = PredictionService() 