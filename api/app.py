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
    """Load model directly without importing prediction.py"""
    global model
    
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        
        # Try multiple model paths
        model_paths = [
            'models/gender_classifier.keras',
            '../models/gender_classifier.keras',
            'src/models/gender_classifier.keras',
            './models/gender_classifier.keras'
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    print(f"📥 Loading model from: {model_path}")
                    model = tf.keras.models.load_model(model_path, compile=False)
                    print(f"✅ Model loaded successfully from: {model_path}")
                    return True
                except Exception as e:
                    print(f"❌ Failed to load {model_path}: {e}")
                    continue
        
        print("❌ No model file found")
        return False
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
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