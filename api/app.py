# api/app.py - Complete Fixed Version with Comprehensive Debugging

import os
import io
import sys
import base64
import tempfile
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse
from PIL import Image
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append('src')

app = FastAPI(
    title="Gender Classifier API",
    description="Predict gender from facial images using a trained CNN model.",
    version="1.0.0"
)

# Global variables
model = None
class_names = ['female', 'male']

def load_model_direct():
    """Comprehensive model loading with detailed debugging"""
    global model
    
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        
        print("🔍 DEBUGGING MODEL LOADING ISSUE")
        print("=" * 50)
        
        # 1. Environment Information
        current_dir = os.getcwd()
        print(f"📁 Current directory: {current_dir}")
        print(f"🐍 Python executable: {sys.executable}")
        print(f"🧠 TensorFlow version: {tf.__version__}")
        
        # 2. Check if we're on Railway
        is_railway = os.environ.get('RAILWAY_ENVIRONMENT') is not None
        print(f"🚂 Running on Railway: {is_railway}")
        
        # 3. List ALL files recursively to find any .keras files
        def find_all_files(directory, max_depth=3):
            """Find all files recursively with depth limit"""
            all_files = []
            try:
                for root, dirs, files in os.walk(directory):
                    # Limit depth to avoid infinite recursion
                    level = root.replace(directory, '').count(os.sep)
                    if level < max_depth:
                        for file in files:
                            full_path = os.path.join(root, file)
                            all_files.append(full_path)
            except Exception as e:
                print(f"❌ Error walking {directory}: {e}")
            return all_files
        
        print("\n📂 COMPLETE FILE SYSTEM SCAN:")
        all_files = find_all_files('.', max_depth=4)
        keras_files = [f for f in all_files if f.endswith(('.keras', '.h5', '.pb'))]
        
        print(f"📊 Total files found: {len(all_files)}")
        print(f"🧠 Model files found: {len(keras_files)}")
        
        if keras_files:
            print("\n🎯 FOUND MODEL FILES:")
            for i, keras_file in enumerate(keras_files, 1):
                size = os.path.getsize(keras_file) if os.path.exists(keras_file) else 0
                print(f"   {i}. {keras_file} ({size:,} bytes)")
        else:
            print("\n❌ NO MODEL FILES FOUND!")
            print("   This is likely why your model isn't loading.")
            print("   Expected files: .keras, .h5, or .pb files")
        
        # 4. Check specific directories
        print("\n📁 DIRECTORY CONTENTS:")
        check_dirs = ['.', 'models', 'src', 'api', 'ui', '../models', './models']
        for check_dir in check_dirs:
            if os.path.exists(check_dir):
                try:
                    contents = os.listdir(check_dir)
                    print(f"   📂 {check_dir}/: {contents}")
                except Exception as e:
                    print(f"   ❌ Error listing {check_dir}: {e}")
            else:
                print(f"   ❌ {check_dir}: Does not exist")
        
        # 5. Try to load model from found files
        if keras_files:
            print(f"\n🔄 ATTEMPTING TO LOAD {len(keras_files)} MODEL FILES:")
            
            for i, model_path in enumerate(keras_files, 1):
                print(f"\n📥 Attempt {i}/{len(keras_files)}: {model_path}")
                
                if not os.path.exists(model_path):
                    print(f"   ❌ File doesn't exist")
                    continue
                
                try:
                    file_size = os.path.getsize(model_path)
                    print(f"   📊 File size: {file_size:,} bytes")
                    
                    if file_size == 0:
                        print(f"   ❌ File is empty")
                        continue
                    
                    # Try different loading methods
                    loading_methods = [
                        ("compile=False", lambda: tf.keras.models.load_model(model_path, compile=False)),
                        ("safe_mode=False", lambda: tf.keras.models.load_model(model_path, compile=False, safe_mode=False)),
                        ("standard", lambda: tf.keras.models.load_model(model_path))
                    ]
                    
                    for method_name, load_func in loading_methods:
                        try:
                            print(f"   🔧 Trying {method_name}...")
                            model = load_func()
                            print(f"   ✅ SUCCESS! Model loaded with {method_name}")
                            
                            # Test the model
                            print(f"   🧪 Testing model...")
                            test_input = tf.random.normal((1, 160, 160, 3))
                            test_pred = model.predict(test_input, verbose=0)
                            print(f"   ✅ Model test successful! Output shape: {test_pred.shape}")
                            
                            return True
                            
                        except Exception as e:
                            print(f"   ❌ {method_name} failed: {str(e)[:100]}...")
                            continue
                
                except Exception as e:
                    print(f"   ❌ Error with {model_path}: {str(e)[:100]}...")
                    continue
        
        # 6. If no model files found, provide guidance
        print("\n" + "=" * 50)
        print("🚨 MODEL LOADING FAILED!")
        print("=" * 50)
        
        if not keras_files:
            print("❌ ROOT CAUSE: No model files found in the deployment")
            print("\n🔧 SOLUTIONS:")
            print("1. Check if your model file is in your GitHub repository")
            print("2. Make sure 'models/gender_classifier.keras' is not in .gitignore")
            print("3. Verify the model file was committed and pushed to GitHub")
            print("4. Consider using Git LFS for large model files")
            print("5. Check if Railway has file size limits for your plan")
        else:
            print("❌ ROOT CAUSE: Model files found but couldn't load any of them")
            print("\n🔧 SOLUTIONS:")
            print("1. The model files might be corrupted")
            print("2. TensorFlow version mismatch")
            print("3. Model was saved with different TensorFlow version")
            print("4. Try retraining and saving the model with current TensorFlow version")
        
        return False
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in load_model_direct: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def predict_gender_direct(img_path):
    """Direct prediction function that definitely works"""
    global model
    
    if model is None:
        print("❌ Model is None, attempting to load...")
        if not load_model_direct():
            return {'error': 'Model not loaded properly'}
    
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing import image
        
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(160, 160))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction with error handling
        with tf.device('/CPU:0'):
            prediction = model.predict(img_array, verbose=0)[0]
        
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[predicted_class])
        predicted_gender = class_names[predicted_class]
        
        result = {
            'predicted_gender': predicted_gender,
            'confidence': confidence,
            'probabilities': {
                'female': float(prediction[0]),
                'male': float(prediction[1])
            }
        }
        
        print(f"✅ Prediction successful: {predicted_gender} ({confidence:.3f})")
        return result
        
    except Exception as e:
        error_msg = f'Prediction failed: {str(e)}'
        print(f"❌ {error_msg}")
        return {'error': error_msg}

# Load model on startup
print("🚀 Starting Gender Classifier API...")
print("🔄 Loading model...")
model_loaded = load_model_direct()

if model_loaded:
    print("✅ API ready for predictions")
else:
    print("❌ API started but model not loaded")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "Gender Classifier API is up and running!",
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": str(type(model)) if model else None
    }

@app.get("/debug")
def debug_info():
    """Debug endpoint to check system status"""
    return {
        "current_directory": os.getcwd(),
        "model_loaded": model is not None,
        "model_type": str(type(model)) if model else None,
        "models_folder_exists": os.path.exists('models'),
        "model_file_exists": os.path.exists('models/gender_classifier.keras'),
        "working_directory_files": os.listdir('.'),
        "models_directory_files": os.listdir('models') if os.path.exists('models') else "No models directory"
    }

@app.get("/debug-comprehensive")
def debug_comprehensive():
    """Most comprehensive debug information"""
    import sys
    import platform
    
    def safe_operation(operation, default="Error"):
        try:
            return operation()
        except Exception as e:
            return f"{default}: {str(e)}"
    
    # Force a model load attempt
    print("\n🔄 FORCING MODEL LOAD ATTEMPT:")
    load_success = load_model_direct()
    
    return {
        "timestamp": str(datetime.now()),
        "model_load_attempt_result": load_success,
        "model_loaded": model is not None,
        "model_type": str(type(model)) if model else None,
        
        "system_info": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "tensorflow_version": safe_operation(lambda: __import__('tensorflow').__version__, "Not available"),
            "current_directory": os.getcwd(),
            "executable": sys.executable,
        },
        
        "environment": {
            "RAILWAY_ENVIRONMENT": os.environ.get("RAILWAY_ENVIRONMENT", "Not set"),
            "PORT": os.environ.get("PORT", "Not set"),
            "PWD": os.environ.get("PWD", "Not set"),
            "HOME": os.environ.get("HOME", "Not set"),
        },
        
        "file_system": {
            "root_files": safe_operation(lambda: os.listdir('.'), []),
            "models_exists": os.path.exists('models'),
            "models_files": safe_operation(lambda: os.listdir('models') if os.path.exists('models') else [], []),
            "src_exists": os.path.exists('src'),
            "api_exists": os.path.exists('api'),
            "ui_exists": os.path.exists('ui'),
        },
        
        "keras_files_scan": safe_operation(lambda: [
            f for f in [
                os.path.join(root, file) 
                for root, dirs, files in os.walk('.') 
                for file in files
            ] if f.endswith(('.keras', '.h5', '.pb'))
        ][:10], []),  # Limit to first 10 results
        
        "disk_usage": safe_operation(lambda: {
            "total_mb": sum(os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk('.')
                        for filename in filenames) / (1024*1024)  # MB
        }, {}),
    }

@app.post("/create-dummy-model")
def create_dummy_model():
    """Create a simple dummy model to test if TensorFlow works"""
    try:
        import tensorflow as tf
        
        # Create a simple model
        dummy_model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(160, 160, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Test prediction
        test_input = tf.random.normal((1, 160, 160, 3))
        prediction = dummy_model.predict(test_input, verbose=0)
        
        # Try to save model (this will fail on Railway due to ephemeral storage)
        save_success = False
        save_error = None
        try:
            if not os.path.exists('models'):
                os.makedirs('models')
            dummy_model.save('models/dummy_model.keras')
            save_success = True
        except Exception as e:
            save_error = str(e)
        
        return {
            "tensorflow_works": True,
            "model_creation_works": True,
            "prediction_works": True,
            "prediction_shape": prediction.shape,
            "model_save_works": save_success,
            "save_error": save_error if not save_success else None,
            "message": "TensorFlow is working properly"
        }
        
    except Exception as e:
        return {
            "tensorflow_works": False,
            "error": str(e),
            "message": "TensorFlow has issues"
        }

@app.post("/test-prediction")
def test_prediction():
    """Test prediction with a dummy image"""
    try:
        # Create a dummy image for testing
        from PIL import Image
        import tempfile
        
        # Create a simple test image
        test_img = Image.new('RGB', (160, 160), color='red')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            test_img.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            result = predict_gender_direct(temp_path)
            return {"test_result": result, "status": "success"}
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return {"error": f"Test failed: {str(e)}", "status": "failed"}

# Base64 Prediction Endpoint
class ImagePayload(BaseModel):
    image: str

@app.post("/predict")
def predict_base64(payload: ImagePayload):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        image_data = base64.b64decode(payload.image)
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image_pil.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            result = predict_gender_direct(temp_path)
            return JSONResponse(content=result)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-file")
async def predict_upload(file: UploadFile = File(...)):
    print(f"📥 Received file: {file.filename} ({file.content_type})")
    
    if model is None:
        print("❌ Model is None")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        print(f"📄 File size: {len(contents)} bytes")
        
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        print(f"🖼️ Image size: {image_pil.size}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image_pil.save(tmp_file.name)
            temp_path = tmp_file.name
            print(f"💾 Saved to temp file: {temp_path}")
        
        try:
            print("🔮 Making prediction...")
            result = predict_gender_direct(temp_path)
            print(f"📊 Prediction result: {result}")
            
            # Add metadata if successful
            if isinstance(result, dict) and 'error' not in result:
                result["filename"] = file.filename
                result["file_size"] = len(contents)
            
            return JSONResponse(content=result)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"🗑️ Cleaned up temp file")
                
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/reload-model")
def reload_model():
    """Manually reload the model"""
    global model
    model = None
    success = load_model_direct()
    return {
        "success": success,
        "model_loaded": model is not None,
        "message": "Model reloaded successfully" if success else "Failed to reload model"
    }

@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    def retrain_task():
        try:
            from retrain import retrain
            model_path = retrain()
            # Reload the model after retraining
            load_model_direct()
            print(f"✅ Model retrained and reloaded: {model_path}")
        except Exception as e:
            print(f"❌ Retraining failed: {e}")
    
    background_tasks.add_task(retrain_task)
    return {"message": "Model retraining started in background"}

if __name__ == "__main__":
    import uvicorn
    # Use Railway's PORT environment variable, fallback to 8000 for local development
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)