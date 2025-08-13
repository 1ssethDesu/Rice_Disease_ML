#!/usr/bin/env python3
"""
Simplified utility functions for Rice Disease Classification API
Contains core image processing, rescaling, and disease information helpers
"""

import os
import csv
import numpy as np
import cv2
import base64
import io
from PIL import Image
from typing import Dict, Optional, List
from fastapi import HTTPException, UploadFile

# =============================================================================
# CORE UTILITY FUNCTIONS
# =============================================================================

def image_process(
    image, 
    target_size: tuple = (299, 299),
    normalize: bool = True
) -> np.ndarray:
    """
    Resizes, normalizes, and reshapes images for ML model input.
    
    Args:
        image: Input image as file path, PIL Image, or numpy array
        target_size: Target dimensions (width, height)
        normalize: Whether to normalize pixel values to [-1, 1] (InceptionV3 style)
    
    Returns:
        Preprocessed image array ready for model input
    """
    try:
        # Load image if path is provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Handle grayscale images
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[-1] == 4:  # RGBA
            img_array = img_array[..., :3]
        
        # Normalize if requested (InceptionV3 style: [-1, 1])
        if normalize:
            img_array = (img_array / 127.5) - 1.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

def rescale_back(
    coordinates, 
    original_size: tuple,
    normalized_size: tuple = (299, 299)
) -> np.ndarray:
    """
    Converts normalized coordinates back to original image dimensions.
    
    Args:
        coordinates: Normalized coordinates (can be single point or array of points)
        original_size: Original image dimensions (width, height)
        normalized_size: Normalized image dimensions (width, height)
    
    Returns:
        Rescaled coordinates in original image dimensions
    """
    try:
        coords = np.array(coordinates, dtype=np.float32)
        
        # Calculate scaling factors
        scale_x = original_size[0] / normalized_size[0]
        scale_y = original_size[1] / normalized_size[1]
        
        # Rescale coordinates
        if coords.ndim == 1:  # Single point
            rescaled = np.array([
                coords[0] * scale_x,
                coords[1] * scale_y
            ])
        else:  # Multiple points
            rescaled = coords.copy()
            rescaled[:, 0] *= scale_x  # X coordinates
            rescaled[:, 1] *= scale_y  # Y coordinates
        
        return rescaled
        
    except Exception as e:
        print(f"Error rescaling coordinates: {e}")
        raise

def get_disease_info(
    disease_name: str,
    csv_path: str = "disease_details.csv"
) -> Optional[Dict]:
    """
    Retrieves detailed disease information from CSV data.
    
    Args:
        disease_name: Name of the disease to look up
        csv_path: Path to the disease details CSV file
    
    Returns:
        Dictionary containing disease information or None if not found
    """
    try:
        # Construct full path
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), csv_path)
        
        if not os.path.exists(csv_path):
            print(f"Disease details CSV not found: {csv_path}")
            return None
        
        # Read CSV and find disease
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                if row['Disease Name'].lower() == disease_name.lower():
                    # Convert CSV row to structured dictionary
                    disease_info = {
                        'name': row['Disease Name'],
                        'description': row['Description'],
                        'risk_level': row['Risk Level'],
                        'affected_area': row['Affected Area'],
                        'key_symptoms': [s.strip() for s in row['Key Symptoms'].split(';')],
                        'causes': [c.strip() for c in row['Causes'].split(';')],
                        'treatment': [t.strip() for t in row['Treatment'].split(';')],
                        'prevention': [p.strip() for p in row['Prevention'].split(';')],
                        'disease_profile': {
                            'class': row['Disease Class'],
                            'type': row['Disease Type'],
                            'transmission': row['Transmission'],
                            'conditions': row['Conditions']
                        },
                        'recommendations': [r.strip() for r in row['Recommendations'].split(';')]
                    }
                    
                    return disease_info
        
        print(f"Disease not found in CSV: {disease_name}")
        return None
        
    except Exception as e:
        print(f"Error retrieving disease info for {disease_name}: {e}")
        return None

# =============================================================================
# IMAGE PROCESSING AND VALIDATION FUNCTIONS
# =============================================================================

def create_annotated_image(image: Image.Image, class_name: str, confidence: float, max_dim: int = 800, jpeg_quality: int = 85) -> str:
    """
    Create an annotated image with prediction results and return as base64 string.
    
    Args:
        image: PIL Image object
        class_name: Name of the predicted disease class
        confidence: Confidence score of the prediction
        max_dim: Maximum dimension for resizing (default: 800)
        jpeg_quality: JPEG quality for compression (default: 85)
    
    Returns:
        Base64 encoded annotated image string or None if failed
    """
    try:
        # Convert PIL image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize image for display (max dimension)
        height, width = img_cv.shape[:2]
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_cv = cv2.resize(img_cv, (new_width, new_height))
        
        # Add text overlay
        text = f"{class_name} ({confidence:.2%})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate text position (top-left corner with padding)
        padding = 10
        text_x = padding
        text_y = text_height + padding
        
        # Add background rectangle for text
        cv2.rectangle(img_cv, 
                     (text_x - 5, text_y - text_height - 5),
                     (text_x + text_width + 5, text_y + baseline + 5),
                     (0, 0, 0), -1)
        
        # Add text
        cv2.putText(img_cv, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        # Convert back to PIL and then to base64
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Convert to base64
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=jpeg_quality)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_base64}"
        
    except Exception as e:
        print(f"Failed to create annotated image: {e}")
        return None

def process_multiple_images(images: List[Image.Image], target_size: tuple = None, normalize: bool = True) -> List[np.ndarray]:
    """
    Process multiple images using the helper function for batch processing.
    This demonstrates the utility of the helper functions for handling multiple images.
    
    Args:
        images: List of PIL Image objects
        target_size: Target dimensions for resizing (default: None uses config)
        normalize: Whether to normalize pixel values
    
    Returns:
        List of preprocessed image arrays ready for model input
    """
    try:
        if target_size is None:
            # Import config here to avoid circular imports
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from config import Config
            target_size = Config.TARGET_IMAGE_SIZE
            
        processed_images = []
        for img in images:
            # Use helper function for consistent image processing
            processed_img = image_process(img, target_size, normalize=normalize)
            processed_images.append(processed_img)
        
        return processed_images
    except Exception as e:
        print(f"Error processing multiple images: {e}")
        raise

def validate_image_file(file: UploadFile, max_size: int = 10 * 1024 * 1024, allowed_extensions: set = None) -> None:
    """Validate uploaded image file.
    
    Args:
        file: UploadFile object to validate
        max_size: Maximum file size in bytes (default: 10MB)
        allowed_extensions: Set of allowed file extensions (default: None uses config)
    
    Raises:
        HTTPException: If validation fails
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image. Supported formats: JPEG, PNG, JPG, BMP, TIFF"
        )
    
    # Check file extension
    file_extension = os.path.splitext(file.filename.lower())[1] if file.filename else ''
    
    if allowed_extensions is None:
        # Import config here to avoid circular imports
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from config import Config
        allowed_extensions = Config.ALLOWED_EXTENSIONS
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    # Check file size
    if file.size and file.size > max_size:
        max_size_mb = max_size // (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"File size too large. Maximum size is {max_size_mb}MB."
        )
