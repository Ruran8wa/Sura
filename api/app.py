# api/app.py - Production Ready Gender Classifier API

import os
import io
import base64
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI(title="Gender Classifier API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
class_names = ['female', 'male']

def load_model():
    """Load the trained model"""
    global model
    
    if model is not None:
        return True
        
    try:
        tf.get_logger().setLevel('ERROR')
        
        # Try to load model from standard paths
        model_paths = [
            'models/gender_classifier.keras',
            'models/test_model_small.keras',
            './models/gender_classifier.keras',
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    model = tf.keras.models.load_model(model_path, compile=False)
                    return True
                except Exception:
                    continue
        
        return False
        
    except Exception:
        return False

def predict_gender(img_path):
    """Predict gender from image path"""
    if not load_model():
        return {'error': 'Model not available'}
    
    try:
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(160, 160))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[predicted_class])
        
        return {
            'predicted_gender': class_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'female': float(prediction[0]),
                'male': float(prediction[1])
            }
        }
        
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}

# Load model on startup
load_model()

@app.get("/")
def health_check():
    return {
        "message": "Gender Classifier API",
        "status": "healthy",
        "model_loaded": model is not None
    }

class ImagePayload(BaseModel):
    image: str

@app.post("/predict")
def predict_base64(payload: ImagePayload):
    try:
        # Decode base64 image
        image_data = base64.b64decode(payload.image)
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Save to temp file and predict
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image_pil.save(tmp_file.name)
            result = predict_gender(tmp_file.name)
            os.remove(tmp_file.name)
            
        if 'error' in result:
            raise HTTPException(status_code=503, detail=result['error'])
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Save to temp file and predict
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image_pil.save(tmp_file.name)
            result = predict_gender(tmp_file.name)
            os.remove(tmp_file.name)
        
        if 'error' in result:
            raise HTTPException(status_code=503, detail=result['error'])
        
        # Add file metadata
        result.update({
            "filename": file.filename,
            "file_size": len(contents)
        })
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    def retrain_task():
        try:
            # Import and run retraining
            import sys
            sys.path.append('src')
            from retrain import retrain
            
            model_path = retrain()
            
            # Reload model
            global model
            model = None
            load_model()
            
        except Exception as e:
            print(f"Retraining failed: {e}")
    
    background_tasks.add_task(retrain_task)
    return {"message": "Retraining started"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)