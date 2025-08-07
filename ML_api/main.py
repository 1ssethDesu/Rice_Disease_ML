import os
import base64
import io
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import uvicorn

app = FastAPI(
    title="Rice Disease Classification API",
    description="API for classifying rice diseases using InceptionV3 model with Grad-CAM visualization",
    version="1.0.0"
)

# Define the path to your model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'InceptionV3 Best Model.h5')

# Global variable to store the loaded model
model = None

# Define the target size for InceptionV3 images (224x224 pixels)
TARGET_IMAGE_SIZE = (224, 224)

# Define your class names (disease names) here
CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy",
    "Leaf Blast",
    "Leaf Scald",
    "Narrow Brown Spot"
]

# Detailed information for each disease
DISEASE_DETAILS = {
    "Bacterial Leaf Blight": {
        "description": "Water-soaked lesions on leaves turning brown and dry. Can cause significant yield loss.",
        "riskLevel": "High",
        "affectedArea": "Leaf tips and margins, eventually whole leaves",
        "keySymptoms": [
            "Yellowing and wilting of leaves",
            "Lesions with water-soaked appearance",
            "Dry brown streaks on leaves"
        ],
        "diseaseProfile": {
            "class": "Bacterial",
            "type": "Foliar disease",
            "transmission": "Rain splash, irrigation water, infected seeds",
            "conditions": "High humidity, warm temperatures (25-30°C), strong winds"
        },
        "recommendations": [
            "Use resistant rice varieties",
            "Avoid overhead irrigation and excessive nitrogen fertilizer",
            "Apply copper-based bactericides if necessary (consult local guidelines)"
        ]
    },
    "Brown Spot": {
        "description": "Oval-shaped spots with a brown margin and a light brown or gray center, often with a yellow halo.",
        "riskLevel": "Moderate",
        "affectedArea": "Leaves, leaf sheaths, glumes, and grains",
        "keySymptoms": [
            "Small, circular to oval spots on leaves",
            "Dark brown margins with light brown centers",
            "Reduced grain quality and yield"
        ],
        "diseaseProfile": {
            "class": "Fungal",
            "type": "Foliar and grain disease",
            "transmission": "Airborne spores, infected seeds, crop residues",
            "conditions": "High humidity, warm temperatures (20-30°C), nutrient deficiencies (especially potassium)"
        },
        "recommendations": [
            "Use resistant varieties",
            "Improve soil fertility and balanced fertilization",
            "Apply fungicides (e.g., strobilurins) as preventive measures"
        ]
    },
    "Healthy": {
        "description": "The leaf shows no visible signs of disease or stress. Indicates optimal plant health.",
        "riskLevel": "Low",
        "affectedArea": "N/A",
        "keySymptoms": [
            "Vibrant green color",
            "Smooth texture",
            "No lesions, spots, or discoloration"
        ],
        "diseaseProfile": {
            "class": "N/A",
            "type": "N/A",
            "transmission": "N/A",
            "conditions": "Optimal growing conditions, balanced nutrition"
        },
        "recommendations": [
            "Continue good agricultural practices",
            "Regular monitoring for early detection of potential issues"
        ]
    },
    "Leaf Blast": {
        "description": "Elliptical lesions with gray centers and dark borders. Can affect all aerial parts of the rice plant.",
        "riskLevel": "High",
        "affectedArea": "Leaf blades, leaf collars, nodes, panicles",
        "keySymptoms": [
            "Spindle-shaped lesions with grey centers and reddish-brown margins",
            "Premature leaf drying (blasting)",
            "Neck rot (panicle blast) causing white heads"
        ],
        "diseaseProfile": {
            "class": "Fungal",
            "type": "Foliar, neck, and panicle disease",
            "transmission": "Wind and rain-borne spores",
            "conditions": "High humidity, cool temperatures (20-25°C), prolonged leaf wetness, excessive nitrogen"
        },
        "recommendations": [
            "Use resistant varieties",
            "Maintain proper field drainage and water management",
            "Reduce excessive nitrogen fertilizer",
            "Apply fungicides like Tricyclazole or Azoxystrobin"
        ]
    },
    "Leaf Scald": {
        "description": "Irregularly shaped lesions with concentric zones, often starting from the leaf tip or margin.",
        "riskLevel": "Moderate",
        "affectedArea": "Leaf blades, leaf sheaths",
        "keySymptoms": [
            "Irregularly shaped lesions with distinct concentric rings",
            "Lesions often merge to cover large areas of the leaf",
            "Leaf tips and margins may dry out and shred"
        ],
        "diseaseProfile": {
            "class": "Fungal",
            "type": "Foliar disease",
            "transmission": "Airborne spores, infected seeds, crop residues",
            "conditions": "High humidity, warm temperatures (25-30°C), prolonged leaf wetness"
        },
        "recommendations": [
            "Use resistant varieties",
            "Remove and destroy infected plant debris",
            "Apply appropriate fungicides (e.g., those containing propiconazole)"
        ]
    },
    "Narrow Brown Spot": {
        "description": "Small, short, linear brown spots on leaves, often appearing between leaf veins.",
        "riskLevel": "Low to Moderate",
        "affectedArea": "Leaves, leaf sheaths, glumes",
        "keySymptoms": [
            "Small, linear to oval brown spots",
            "Spots usually confined between leaf veins",
            "Can cause premature leaf senescence in severe cases"
        ],
        "diseaseProfile": {
            "class": "Fungal",
            "type": "Foliar disease",
            "transmission": "Airborne spores, infected seeds, crop residues",
            "conditions": "High humidity, warm temperatures (25-30°C), nutrient imbalances"
        },
        "recommendations": [
            "Use resistant varieties",
            "Maintain balanced fertilization",
            "Fungicide application usually not necessary unless severe"
        ]
    }
}

# Pydantic model for response validation (no request model needed for file upload)

# Pydantic model for response
class DiseaseResult(BaseModel):
    diseaseName: str
    description: str
    confidence: float
    riskLevel: str
    affectedArea: str
    imgUrl: str
    keySymptoms: list[str]
    diseaseProfile: dict
    recommendations: list[str]

class PredictionResponse(BaseModel):
    status: str
    timestamp: str
    results: list[DiseaseResult]

def load_trained_model():
    """
    Loads the pre-trained Keras model.
    This function is called once when the FastAPI app starts.
    """
    global model
    try:
        model = load_model(MODEL_PATH)
        print(f"Model '{MODEL_PATH}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        print("Please ensure the model file exists at the specified path and is not corrupted.")
        raise e

def generate_grad_cam(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image and model.
    """
    # Create a model that maps the input image to the activations of the last conv layer
    # and the final output predictions.
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Use tf.GradientTape to compute gradients
    with tf.GradientTape() as tape:
        # Get the outputs of the last conv layer and the model predictions
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            # If no specific prediction index is provided, use the highest probability class
            pred_index = tf.argmax(preds[0])
        # Get the loss for the predicted class
        class_channel = preds[:, pred_index]

    # Compute the gradients of the predicted class with respect to the last conv layer outputs
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Compute the mean intensity of the gradient over all feature map channels
    # This is the "weights" for each feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by its corresponding weight
    # Then sum all channels to get the heatmap
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap) # Remove single-dimensional entries from the shape of an array

    # Apply ReLU to the heatmap (only positive gradients contribute to activation)
    heatmap = tf.maximum(heatmap, 0)

    # Normalize the heatmap to a 0-1 range
    max_heatmap = tf.reduce_max(heatmap)
    if max_heatmap == 0: # Avoid division by zero if heatmap is all zeros
        heatmap = heatmap * 0
    else:
        heatmap = heatmap / max_heatmap

    # Convert to NumPy array and scale to 0-255 for image processing
    heatmap = heatmap.numpy() * 255
    return heatmap.astype(np.uint8)

def superimpose_heatmap(original_img_pil, heatmap):
    """
    Superimposes the Grad-CAM heatmap onto the original image.
    """
    # Resize heatmap to original image size (which is TARGET_IMAGE_SIZE here)
    heatmap = cv2.resize(heatmap, TARGET_IMAGE_SIZE)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # Apply a colormap (e.g., JET)

    # Convert PIL image to OpenCV format (BGR)
    original_img_cv = cv2.cvtColor(np.array(original_img_pil), cv2.COLOR_RGB2BGR)

    # Superimpose the heatmap on the original image
    # alpha controls the transparency of the heatmap (0.4 is common)
    superimposed_img_cv = cv2.addWeighted(original_img_cv, 0.6, heatmap, 0.4, 0)

    # Convert back to RGB for PIL and then to base64
    superimposed_img_pil = Image.fromarray(cv2.cvtColor(superimposed_img_cv, cv2.COLOR_BGR2RGB))

    buffered = io.BytesIO()
    superimposed_img_pil.save(buffered, format="PNG") # Save as PNG for transparency support
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.on_event("startup")
async def startup_event():
    """
    Load the model when the application starts up.
    """
    load_trained_model()

@app.get("/")
async def home():
    """
    A simple home route to confirm the API is running.
    """
    return {
        "message": "Welcome to the Rice Disease Classification API!",
        "description": "Send a POST request to /predict for image classification.",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat() + 'Z'
    }

# Pydantic model for base64 request (for backward compatibility)
class PredictionRequest(BaseModel):
    image_base64: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    API endpoint to receive an image file, preprocess it,
    make a prediction using the InceptionV3 model, and return the result,
    along with a Grad-CAM visualization.
    
    Supports: JPEG, PNG, JPG image formats
    """
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please check server logs for errors during startup."
        )

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image. Supported formats: JPEG, PNG, JPG"
        )

    try:
        # Read the uploaded file
        image_bytes = await file.read()
        original_img_pil = Image.open(io.BytesIO(image_bytes))

        # Store the original PIL image (resized) for Grad-CAM superimposition
        img_for_gradcam = original_img_pil.resize(TARGET_IMAGE_SIZE)

        # Preprocess the image for model prediction
        img_array = np.array(img_for_gradcam).astype('float32')
        if img_array.ndim == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        img_array = (img_array / 127.5) - 1.0 # Normalize to [-1, 1]
        input_data = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(input_data)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_confidence = float(np.max(predictions)) # Get the highest probability

        predicted_class_name = "Unknown Disease"
        if 0 <= predicted_class_index < len(CLASS_NAMES):
            predicted_class_name = CLASS_NAMES[predicted_class_index]

        # Grad-CAM Generation
        LAST_CONV_LAYER_NAME = 'conv2d_281' # This should be the last Conv2D layer in your InceptionV3 model

        heatmap = generate_grad_cam(input_data, model, LAST_CONV_LAYER_NAME, predicted_class_index)
        grad_cam_image_base64 = superimpose_heatmap(img_for_gradcam, heatmap)

        # Construct the detailed response
        current_time = datetime.utcnow().isoformat() + 'Z' # Get current UTC time in ISO format

        # Retrieve detailed info for the predicted disease
        disease_info = DISEASE_DETAILS.get(predicted_class_name, {
            "description": "No detailed description available.",
            "riskLevel": "Unknown",
            "affectedArea": "Unknown",
            "keySymptoms": ["No symptoms listed."],
            "diseaseProfile": {"class": "Unknown", "type": "Unknown", "transmission": "Unknown", "conditions": "Unknown"},
            "recommendations": ["No specific recommendations available."]
        })

        response = PredictionResponse(
            status="success",
            timestamp=current_time,
            results=[
                DiseaseResult(
                    diseaseName=predicted_class_name,
                    description=disease_info["description"],
                    confidence=predicted_confidence,
                    riskLevel=disease_info["riskLevel"],
                    affectedArea=disease_info["affectedArea"],
                    imgUrl=f"data:image/png;base64,{grad_cam_image_base64}",
                    keySymptoms=disease_info["keySymptoms"],
                    diseaseProfile=disease_info["diseaseProfile"],
                    recommendations=disease_info["recommendations"]
                )
            ]
        )

        return response

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred: {str(e)}"
        )

@app.post("/predict-base64", response_model=PredictionResponse)
async def predict_base64(request: PredictionRequest):
    """
    API endpoint to receive an image (base64 encoded), preprocess it,
    make a prediction using the InceptionV3 model, and return the result,
    along with a Grad-CAM visualization.
    
    This endpoint is for backward compatibility with base64 encoded images.
    For new implementations, use /predict with file upload.
    """
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please check server logs for errors during startup."
        )

    try:
        image_base64 = request.image_base64

        image_bytes = base64.b64decode(image_base64)
        original_img_pil = Image.open(io.BytesIO(image_bytes))

        # Store the original PIL image (resized) for Grad-CAM superimposition
        img_for_gradcam = original_img_pil.resize(TARGET_IMAGE_SIZE)

        # Preprocess the image for model prediction
        img_array = np.array(img_for_gradcam).astype('float32')
        if img_array.ndim == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        img_array = (img_array / 127.5) - 1.0 # Normalize to [-1, 1]
        input_data = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(input_data)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_confidence = float(np.max(predictions)) # Get the highest probability

        predicted_class_name = "Unknown Disease"
        if 0 <= predicted_class_index < len(CLASS_NAMES):
            predicted_class_name = CLASS_NAMES[predicted_class_index]

        # Grad-CAM Generation
        LAST_CONV_LAYER_NAME = 'conv2d_281' # This should be the last Conv2D layer in your InceptionV3 model

        heatmap = generate_grad_cam(input_data, model, LAST_CONV_LAYER_NAME, predicted_class_index)
        grad_cam_image_base64 = superimpose_heatmap(img_for_gradcam, heatmap)

        # Construct the detailed response
        current_time = datetime.utcnow().isoformat() + 'Z' # Get current UTC time in ISO format

        # Retrieve detailed info for the predicted disease
        disease_info = DISEASE_DETAILS.get(predicted_class_name, {
            "description": "No detailed description available.",
            "riskLevel": "Unknown",
            "affectedArea": "Unknown",
            "keySymptoms": ["No symptoms listed."],
            "diseaseProfile": {"class": "Unknown", "type": "Unknown", "transmission": "Unknown", "conditions": "Unknown"},
            "recommendations": ["No specific recommendations available."]
        })

        response = PredictionResponse(
            status="success",
            timestamp=current_time,
            results=[
                DiseaseResult(
                    diseaseName=predicted_class_name,
                    description=disease_info["description"],
                    confidence=predicted_confidence,
                    riskLevel=disease_info["riskLevel"],
                    affectedArea=disease_info["affectedArea"],
                    imgUrl=f"data:image/png;base64,{grad_cam_image_base64}",
                    keySymptoms=disease_info["keySymptoms"],
                    diseaseProfile=disease_info["diseaseProfile"],
                    recommendations=disease_info["recommendations"]
                )
            ]
        )

        return response

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
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
        "diseases": list(DISEASE_DETAILS.keys()),
        "total_count": len(DISEASE_DETAILS)
    }

@app.get("/diseases/{disease_name}")
async def get_disease_info(disease_name: str):
    """
    Get detailed information about a specific disease.
    """
    if disease_name not in DISEASE_DETAILS:
        raise HTTPException(
            status_code=404, 
            detail=f"Disease '{disease_name}' not found. Available diseases: {list(DISEASE_DETAILS.keys())}"
        )
    
    return {
        "disease_name": disease_name,
        "details": DISEASE_DETAILS[disease_name]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 