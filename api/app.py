# api/app.py

import os
import io
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse
from PIL import Image
from src.prediction import predict_gender
from src.retrain import retrain

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
    return {"message": "Gender Classifier API is up and running!"}


# Base64 Prediction Endpoint
class ImagePayload(BaseModel):
    image: str  # base64-encoded string

@app.post("/predict")
def predict_base64(payload: ImagePayload):
    try:
        image_data = base64.b64decode(payload.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        temp_path = "temp.jpg"
        image.save(temp_path)

        result = predict_gender(temp_path)
        os.remove(temp_path)

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# File Upload Endpoint
@app.post("/predict-file")
async def predict_upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        temp_path = "temp.jpg"
        image.save(temp_path)

        result = predict_gender(temp_path)
        os.remove(temp_path)

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/retrain")
def retrain_model():
    try:
        model_path = retrain()
        return {"message": "Model retrained successfully!", "model_path": model_path}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
