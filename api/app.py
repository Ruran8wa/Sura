# api/app.py

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

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.prediction import predict_gender
    from src.retrain import retrain
except ImportError:
    print("Warning: Could not import prediction/retrain modules")
    predict_gender = None
    retrain = None

app = FastAPI(
    title="Gender Classifier API",
    description="Predict gender from facial images using a pretrained MobileNetV2 model.",
    version="1.0.0"
)

# Optional: Enable CORS (for Streamlit UI or other frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/")
def read_root():
    return {
        "message": "Gender Classifier API is up and running!",
        "status": "healthy",
        "prediction_available": predict_gender is not None,
        "retrain_available": retrain is not None
    }


# Base64 Prediction Endpoint
class ImagePayload(BaseModel):
    image: str  # base64-encoded string

@app.post("/predict")
def predict_base64(payload: ImagePayload):
    if predict_gender is None:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        image_data = base64.b64decode(payload.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Use temporary file with proper cleanup
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            result = predict_gender(temp_path)
            return JSONResponse(content=result)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# File Upload Endpoint
@app.post("/predict-file")
async def predict_upload(file: UploadFile = File(...)):
    if predict_gender is None:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Use temporary file with proper cleanup
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            result = predict_gender(temp_path)
            
            # Add metadata
            if isinstance(result, dict):
                result["filename"] = file.filename
                result["file_size"] = len(contents)
            
            return JSONResponse(content=result)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    if retrain is None:
        raise HTTPException(status_code=503, detail="Retrain service not available")
    
    def retrain_task():
        try:
            model_path = retrain()
            print(f"✅ Model retrained successfully: {model_path}")
        except Exception as e:
            print(f"❌ Retraining failed: {e}")
    
    background_tasks.add_task(retrain_task)
    return {"message": "Model retraining started in background"}

@app.get("/model/status")
def model_status():
    """Get model status information"""
    return {
        "prediction_available": predict_gender is not None,
        "retrain_available": retrain is not None,
        "status": "ready" if predict_gender is not None else "not ready"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
