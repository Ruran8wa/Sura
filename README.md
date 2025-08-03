# Gender Classifier

A machine learning project for gender classification using deep learning techniques.

## Project Structure

```
./
├── README.md
├── Dockerfile
├── requirements.txt
├── notebook/
│   └── gender_classifier.ipynb
├── data/
│   ├── train/
│   └── test/
├── models/
│   └── gender_model.h5
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   └── retrain.py
├── api/
│   └── app.py
├── ui/
│   └── streamlit_app.py
└── locust/
    └── locustfile.py
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit UI:
   ```bash
   streamlit run ui/streamlit_app.py
   ```

3. Run the FastAPI backend:
   ```bash
   cd api
   uvicorn app:app --reload --port 8080
   ```

## Features

- **Deep Learning Model**: MobileNetV2-based gender classification with ~80-85% accuracy
- **REST API**: FastAPI backend with prediction and retraining endpoints
- **Interactive UI**: Streamlit web interface for easy image upload and prediction
- **Model Retraining**: Automated retraining capabilities with timestamped model saving
- **Production Ready**: Minimal, clean codebase optimized for deployment

## Demo
https://youtu.be/-923d6s3yCc
### Local Usage

1. **Start the API server**:
   ```bash
   cd api
   uvicorn app:app --reload --port 8080
   ```

2. **Start the Streamlit UI** (in a new terminal):
   ```bash
   streamlit run ui/streamlit_app.py
   ```

3. **Access the application**:
   - Streamlit UI: http://localhost:8501
   - API documentation: http://localhost:8080/docs

### Using the Interface

1. Upload an image using the file uploader in the Streamlit interface
2. View the gender prediction with confidence scores
3. Use the "Retrain Model" button to trigger model retraining with new data

### API Endpoints

- `GET /` - Health check and model status
- `POST /predict` - Predict gender from base64 encoded image
- `POST /predict-file` - Predict gender from uploaded file
- `POST /retrain` - Trigger model retraining (background task)

## Results

### Model Performance
- **Architecture**: MobileNetV2 with custom classification head
- **Input Size**: 160x160 RGB images
- **Accuracy**: 80-85% on test data
- **Training Time**: ~2-3 hours on standard hardware
- **Model Size**: ~9MB (optimized for deployment)

### Dataset
- Training on gender-labeled facial images
- Automatic preprocessing and augmentation
- Balanced dataset with male/female classes

### Performance Metrics
- **Precision**: ~0.83
- **Recall**: ~0.82
- **F1-Score**: ~0.82
- **Inference Time**: <100ms per image

## Notebook

The complete development process is documented in `notebook/gender_recognition.ipynb.ipynb`, including:

- Data exploration and preprocessing
- Model architecture design and training
- Performance evaluation and visualization
- Hyperparameter tuning experiments
- Model comparison and selection

To run the notebook:
```bash
jupyter notebook notebook/gender_recognition.ipynb
```

## Deployment

### Local Deployment
The application runs successfully in local environments with both API and UI components communicating properly.

### Cloud Deployment Issues

**Railway Deployment**: Attempted deployment to Railway platform encountered several challenges:

1. **TensorFlow Version Conflicts**:
   - Railway's Python environment has compatibility issues with TensorFlow 2.15+
   - Model files created with newer TensorFlow versions cannot be loaded
   - Downgrading TensorFlow breaks other dependencies

2. **Resource Limitations**:
   - Model loading requires significant memory (>512MB)
   - Railway's free tier has strict memory limits
   - Cold start times exceed platform timeouts

3. **Model File Handling**:
   - Large model files (9MB+) cause deployment package size issues
   - Git LFS integration complications with Railway

### Recommended Solutions for Production

1. **Dockerization**:
   ```dockerfile
   FROM python:3.9-slim
   # Controlled environment with specific TensorFlow version
   ```

2. **External Model Storage**:
   - Store models in cloud storage (AWS S3, Google Cloud Storage)
   - Download models at runtime or during container initialization

3. **Model Optimization**:
   - Implement model quantization to reduce size
   - Use TensorFlow Lite for mobile/edge deployment
   - Consider ONNX format for cross-platform compatibility

4. **Alternative Platforms**:
   - Google Cloud Run (better TensorFlow support)
   - Heroku with custom buildpacks
   - AWS Lambda with container images

### Load Testing Results

Attempted load testing with Locust revealed:
- Local API handles ~50-100 concurrent requests efficiently
- Memory usage scales linearly with concurrent predictions
- Response times average <200ms under normal load

## Development

### Training New Models
```bash
python src/model.py
```

### Retraining with New Data
```bash
python src/retrain.py
```

### Running Tests
```bash
python -m pytest tests/
```

## Dependencies

Key packages:
- TensorFlow 2.15+
- FastAPI
- Streamlit
- Pillow
- NumPy
- Uvicorn

See `requirements.txt` for complete dependency list.

## Future Improvements

1. **Model Enhancements**:
   - Experiment with Vision Transformers
   - Implement ensemble methods
   - Add age classification capability

2. **Production Features**:
   - User authentication system
   - Prediction history tracking
   - Batch processing capabilities
   - Model versioning and A/B testing

3. **Deployment Optimization**:
   - Implement proper Docker containerization
   - Set up CI/CD pipeline
   - Add monitoring and logging
   - Implement model caching strategies
