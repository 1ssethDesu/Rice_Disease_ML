#!/usr/bin/env python3
"""
Script to convert TensorFlow/Keras model to ONNX format
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tf2onnx
import onnx
import onnxruntime as ort
from PIL import Image
import time
import argparse

def load_keras_model(model_path):
    """Load the trained Keras model."""
    print(f"Loading Keras model from: {model_path}")
    try:
        model = load_model(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Model output shape: {model.output_shape}")
        print(f"   Model parameters: {model.count_params():,}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

def convert_to_onnx(keras_model, output_path, input_shape=(299, 299, 3)):
    """Convert Keras model to ONNX format."""
    print(f"\n🔄 Converting model to ONNX format...")
    
    try:
        # Add batch dimension to input shape if not present
        if len(input_shape) == 3:
            input_shape_with_batch = (None,) + input_shape
        else:
            input_shape_with_batch = input_shape
            
        # Convert to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(
            keras_model, 
            input_signature=[tf.TensorSpec(input_shape_with_batch, tf.float32, name="input_1")],
            opset=13,
            output_path=output_path
        )
        
        print(f"✅ ONNX conversion successful!")
        print(f"   Output file: {output_path}")
        print(f"   ONNX model size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        return onnx_model
        
    except Exception as e:
        print(f"❌ Error during ONNX conversion: {e}")
        raise e

def validate_onnx_model(onnx_path):
    """Validate the ONNX model."""
    print(f"\n🔍 Validating ONNX model...")
    
    try:
        # Load and validate ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model validation successful!")
        
        # Print model info
        print(f"   ONNX IR version: {onnx_model.ir_version}")
        print(f"   Producer: {onnx_model.producer_name}")
        print(f"   Opset version: {onnx_model.opset_import[0].version}")
        
        return onnx_model
        
    except Exception as e:
        print(f"❌ ONNX model validation failed: {e}")
        raise e

def test_onnx_inference(onnx_path, test_input_shape=(1, 299, 299, 3)):
    """Test ONNX model inference."""
    print(f"\n🧪 Testing ONNX model inference...")
    
    try:
        # Create test input
        test_input = np.random.random(test_input_shape).astype(np.float32)
        
        # Run inference with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get input/output names
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        print(f"   Input name: {input_name}")
        print(f"   Output name: {output_name}")
        
        # Warm-up run
        _ = ort_session.run([output_name], {input_name: test_input})
        
        # Benchmark inference
        num_runs = 10
        start_time = time.time()
        
        for _ in range(num_runs):
            _ = ort_session.run([output_name], {input_name: test_input})
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        print(f"✅ ONNX inference test successful!")
        print(f"   Average inference time: {avg_time*1000:.2f} ms")
        print(f"   Throughput: {1/avg_time:.1f} inferences/second")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX inference test failed: {e}")
        raise e

def compare_predictions(keras_model, onnx_path, test_input_shape=(1, 299, 299, 3)):
    """Compare predictions between Keras and ONNX models."""
    print(f"\n🔍 Comparing predictions between Keras and ONNX models...")
    
    try:
        # Create test input
        test_input = np.random.random(test_input_shape).astype(np.float32)
        
        # Keras prediction
        keras_start = time.time()
        keras_output = keras_model.predict(test_input, verbose=0)
        keras_time = time.time() - keras_start
        
        # ONNX prediction
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        onnx_start = time.time()
        onnx_output = ort_session.run([output_name], {input_name: test_input})
        onnx_time = time.time() - onnx_start
        
        # Compare outputs
        keras_pred = np.argmax(keras_output[0])
        onnx_pred = np.argmax(onnx_output[0])
        
        keras_conf = np.max(keras_output[0])
        onnx_conf = np.max(onnx_output[0])
        
        print(f"✅ Prediction comparison successful!")
        print(f"   Keras - Class: {keras_pred}, Confidence: {keras_conf:.4f}, Time: {keras_time*1000:.2f}ms")
        print(f"   ONNX  - Class: {onnx_pred}, Confidence: {onnx_conf:.4f}, Time: {onnx_time*1000:.2f}ms")
        
        # Check if predictions match
        if keras_pred == onnx_pred:
            print(f"   ✅ Predictions match!")
        else:
            print(f"   ⚠️  Predictions differ!")
        
        # Check confidence difference
        conf_diff = abs(keras_conf - onnx_conf)
        if conf_diff < 0.01:
            print(f"   ✅ Confidence scores are very close (diff: {conf_diff:.4f})")
        else:
            print(f"   ⚠️  Confidence scores differ significantly (diff: {conf_diff:.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction comparison failed: {e}")
        raise e

def create_onnx_inference_script(onnx_path, output_script_path):
    """Create a simple inference script for the ONNX model."""
    print(f"\n📝 Creating ONNX inference script...")
    
    script_content = f'''#!/usr/bin/env python3
"""
Simple inference script for ONNX rice disease classification model
"""

import numpy as np
import onnxruntime as ort
from PIL import Image
import time

class ONNXRiceDiseaseClassifier:
    def __init__(self, model_path="{onnx_path}"):
        """Initialize the ONNX classifier."""
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Class names for rice diseases
        self.class_names = [
            "Bacterial Leaf Blight",
            "Brown Spot", 
            "Healthy",
            "Leaf Blast",
            "Leaf Scald",
            "Narrow Brown Spot"
        ]
        
        print(f"✅ ONNX model loaded: {{model_path}}")
    
    def preprocess_image(self, image_path, target_size=(299, 299)):
        """Preprocess image for inference."""
        # Load and resize image
        img = Image.open(image_path).resize(target_size)
        img_array = np.array(img).astype(np.float32)
        
        # Handle grayscale images
        if img_array.ndim == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        
        # Normalize to [-1, 1] (InceptionV3 preprocessing)
        img_array = (img_array / 127.5) - 1.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path):
        """Make prediction on an image."""
        try:
            # Preprocess image
            input_data = self.preprocess_image(image_path)
            
            # Run inference
            start_time = time.time()
            outputs = self.session.run([self.output_name], {{self.input_name: input_data}})
            inference_time = time.time() - start_time
            
            # Get predictions
            predictions = outputs[0][0]
            predicted_class_index = np.argmax(predictions)
            confidence = float(np.max(predictions))
            
            # Get class name
            predicted_class = self.class_names[predicted_class_index]
            
            return {{
                'class_name': predicted_class,
                'confidence': confidence,
                'inference_time': inference_time,
                'all_predictions': predictions.tolist()
            }}
            
        except Exception as e:
            print(f"Error during prediction: {{e}}")
            return None

def main():
    """Example usage."""
    # Initialize classifier
    classifier = ONNXRiceDiseaseClassifier()
    
    # Test with an image (replace with your image path)
    test_image = "test_image.jpg"
    
    if os.path.exists(test_image):
        result = classifier.predict(test_image)
        if result:
            print(f"\\n🌾 Rice Disease Classification Result:")
            print(f"   Disease: {{result['class_name']}}")
            print(f"   Confidence: {{result['confidence']:.2%}}")
            print(f"   Inference Time: {{result['inference_time']*1000:.2f}}ms")
    else:
        print(f"Test image {{test_image}} not found. Please provide a valid image path.")

if __name__ == "__main__":
    main()
'''
    
    try:
        with open(output_script_path, 'w') as f:
            f.write(script_content)
        
        print(f"✅ Inference script created: {output_script_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating inference script: {e}")
        return False

def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description='Convert Keras model to ONNX format')
    parser.add_argument('--model_path', type=str, default='./model/InceptionV3_Model.h5',
                       help='Path to the Keras model file')
    parser.add_argument('--output_path', type=str, default='./model/rice_disease_model.onnx',
                       help='Output path for the ONNX model')
    parser.add_argument('--input_shape', type=str, default='299,299,3',
                       help='Input shape as comma-separated values (e.g., 299,299,3)')
    
    args = parser.parse_args()
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    print("🚀 Starting Keras to ONNX conversion...")
    print("=" * 60)
    
    try:
        # Step 1: Load Keras model
        keras_model = load_keras_model(args.model_path)
        
        # Step 2: Convert to ONNX
        onnx_model = convert_to_onnx(keras_model, args.output_path, input_shape)
        
        # Step 3: Validate ONNX model
        validate_onnx_model(args.output_path)
        
        # Step 4: Test ONNX inference
        test_onnx_inference(args.output_path, (1,) + input_shape)
        
        # Step 5: Compare predictions
        compare_predictions(keras_model, args.output_path, (1,) + input_shape)
        
        # Step 6: Create inference script
        script_path = args.output_path.replace('.onnx', '_inference.py')
        create_onnx_inference_script(args.output_path, script_path)
        
        print("\n" + "=" * 60)
        print("🎉 ONNX conversion completed successfully!")
        print(f"📁 ONNX model saved to: {args.output_path}")
        print(f"📝 Inference script saved to: {script_path}")
        print("\n💡 Next steps:")
        print("   1. Test the ONNX model with: python {script_path}")
        print("   2. Integrate ONNX model into your FastAPI")
        print("   3. Benchmark performance improvements")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 