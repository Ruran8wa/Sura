# api/app.py - Final Fixed Version

import os
import io
import sys
import base64
import tempfile
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
    """Load model directly with comprehensive path checking"""
    global model
    
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        
        # Get current working directory
        current_dir = os.getcwd()
        print(f"🔍 Current working directory: {current_dir}")
        
        # List all files and directories to debug
        print(f"📁 Files in current directory: {os.listdir('.')}")
        
        # Try multiple model paths (including Railway-specific paths)
        model_paths = [
            'models/gender_classifier.keras',           # Standard path
            'models/test_model_small.keras',            # Smaller fallback model
            './models/gender_classifier.keras',         # Explicit relative path
            './models/test_model_small.keras',          # Smaller fallback
            '../models/gender_classifier.keras',        # Parent directory
            'src/models/gender_classifier.keras',       # In src folder
            './src/models/gender_classifier.keras',     # Explicit src path
            '/app/models/gender_classifier.keras',      # Railway absolute path
            '/app/models/test_model_small.keras',       # Railway small model
            '/opt/railway/models/gender_classifier.keras', # Alternative Railway path
        ]
        
        # Also check if models directory exists
        models_dirs = ['models', './models', '../models', 'src/models', './src/models']
        for models_dir in models_dirs:
            if os.path.exists(models_dir):
                print(f"📁 Found models directory: {models_dir}")
                print(f"   Contents: {os.listdir(models_dir)}")
                
                # Add any .keras files found in this directory
                for file in os.listdir(models_dir):
                    if file.endswith('.keras') or file.endswith('.h5'):
                        full_path = os.path.join(models_dir, file)
                        if full_path not in model_paths:
                            model_paths.append(full_path)
        
        print(f"🔍 Trying {len(model_paths)} model paths...")
        
        for i, model_path in enumerate(model_paths, 1):
            print(f"📥 Attempt {i}/{len(model_paths)}: {model_path}")
            
            if os.path.exists(model_path):
                try:
                    print(f"   ✅ File exists, size: {os.path.getsize(model_path)} bytes")
                    model = tf.keras.models.load_model(model_path, compile=False)
                    print(f"   🎉 Model loaded successfully from: {model_path}")
                    
                    # Test the model with a dummy prediction
                    test_input = tf.random.normal((1, 160, 160, 3))
                    test_pred = model.predict(test_input, verbose=0)
                    print(f"   ✅ Model test prediction successful: {test_pred.shape}")
                    
                    return True
                except Exception as e:
                    print(f"   ❌ Failed to load {model_path}: {str(e)[:100]}...")
                    continue
            else:
                print(f"   ❌ File does not exist: {model_path}")
        
        print("❌ No model file found in any of the attempted paths")
        return False
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
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

@app.get("/debug-detailed")
def debug_detailed():
    """Comprehensive debug endpoint"""
    import sys
    
    def safe_listdir(path):
        try:
            if os.path.exists(path):
                return os.listdir(path)
            else:
                return f"Path does not exist: {path}"
        except Exception as e:
            return f"Error accessing path: {str(e)}"
    
    def find_keras_files(directory):
        """Recursively find all .keras files"""
        keras_files = []
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.keras') or file.endswith('.h5'):
                        keras_files.append(os.path.join(root, file))
        except Exception as e:
            return [f"Error walking directory: {str(e)}"]
        return keras_files
    
    return {
        "python_version": sys.version,
        "current_directory": os.getcwd(),
        "model_loaded": model is not None,
        "model_type": str(type(model)) if model else None,
        "environment_variables": {
            "PORT": os.environ.get("PORT", "Not set"),
            "RAILWAY_ENVIRONMENT": os.environ.get("RAILWAY_ENVIRONMENT", "Not set"),
            "PWD": os.environ.get("PWD", "Not set")
        },
        "file_system": {
            "root_directory": safe_listdir('.'),
            "models_directory": safe_listdir('models'),
            "src_directory": safe_listdir('src'),
            "parent_directory": safe_listdir('..'),
        },
        "keras_files_found": find_keras_files('.'),
        "tensorflow_version": tf.__version__ if 'tensorflow' in sys.modules else "Not imported",
        "sys_path": sys.path[:5]  # First 5 entries only
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
    
    # ✅ FIX: Don't check model here, let predict_gender_direct handle it
    # if model is None:
    #     print("❌ Model is None")
    #     raise HTTPException(status_code=503, detail="Model not loaded")
    
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
            result = predict_gender_direct(temp_path)  # ✅ This will load model if needed
            print(f"📊 Prediction result: {result}")
            
            # Check if prediction failed
            if isinstance(result, dict) and 'error' in result:
                raise HTTPException(status_code=503, detail=result['error'])
            
            # Add metadata if successful
            result["filename"] = file.filename
            result["file_size"] = len(contents)
            
            return JSONResponse(content=result)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"🗑️ Cleaned up temp file")
                
    except HTTPException:
        raise
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
    # Use Railway's PORT environment variable, fallback to 8080 for local development
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)