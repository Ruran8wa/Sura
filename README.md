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

3. Run the Flask API:
   ```bash
   python api/app.py
   ```

## Features

- Data preprocessing and model training
- REST API for predictions
- Interactive Streamlit UI
- Model retraining capabilities
- Load testing with Locust

## Usage

[Add usage instructions here]
