import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = os.path.join('models', 'gender_classifier.keras')

model = None
class_names = ['female', 'male']

def load_model_with_debug():
    global model
    
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Model file not found: {MODEL_PATH}")
        return False
    
    print(f"Attempting to load model: {MODEL_PATH}")
    print(f"File size: {os.path.getsize(MODEL_PATH)} bytes")
    
    loading_methods = [
        ("standard", lambda: load_model(MODEL_PATH)),
        ("compile=False", lambda: load_model(MODEL_PATH, compile=False)),
        ("safe_mode", lambda: load_model(MODEL_PATH, compile=False, safe_mode=False))
    ]
    
    for method_name, load_func in loading_methods:
        try:
            print(f"Trying {method_name} method...")
            model = load_func()
            print(f"✓ Model loaded successfully using {method_name}!")
            return True
        except Exception as e:
            print(f"✗ {method_name} failed: {str(e)[:100]}...")
            continue
    
    print("✗ All loading methods failed")
    model = None
    return False

model_loaded = load_model_with_debug()

def preprocess_image(img_path, target_size=(160, 160)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_gender(img_path):
    if model is None:
        return {'error': 'Model not loaded properly'}
    
    try:
        img_array = preprocess_image(img_path)
        
        with tf.device('/CPU:0'):
            prediction = model.predict(img_array, verbose=0)[0]
        
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[predicted_class])
        label = class_names[predicted_class]
        
        return {
            'predicted_gender': label, 
            'confidence': confidence,
            'probabilities': {
                'female': float(prediction[0]),
                'male': float(prediction[1])
            }
        }
    except Exception as e:
        return {'error': str(e)}

class GenderPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
        self.model_path = model_path
        self.model = None
        self.class_names = ['female', 'male']
        
    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Model file not found: {self.model_path}")
            return False
        
        try:
            self.model = load_model(self.model_path, compile=False)
            print(f"✓ Model loaded: {self.model_path}")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model = None
            return False
    
    def predict_single(self, img_input):
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        try:
            if isinstance(img_input, str):
                img_array = preprocess_image(img_input, target_size=(160, 160))
            else:
                if len(img_input.shape) == 3:
                    if img_input.shape[:2] != (160, 160):
                        pil_img = array_to_img(img_input)
                        pil_img = pil_img.resize((160, 160))
                        img_input = img_to_array(pil_img)
                    img_array = np.expand_dims(img_input, axis=0)
                else:
                    img_array = img_input
                if img_array.max() > 1.0:
                    img_array = img_array / 255.0
            
            with tf.device('/CPU:0'):
                prediction = self.model.predict(img_array, verbose=0)[0]
            
            predicted_class = np.argmax(prediction)
            confidence = float(prediction[predicted_class])
            predicted_gender = self.class_names[predicted_class]
            
            return {
                'predicted_gender': predicted_gender,
                'confidence': confidence,
                'probabilities': {
                    'female': float(prediction[0]),
                    'male': float(prediction[1])
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def predict_batch(self, img_inputs):
        results = []
        for img_input in img_inputs:
            result = self.predict_single(img_input)
            results.append(result)
        return results

if model_loaded and model is not None:
    print("✓ Model loaded successfully")
else:
    print("✗ Model failed to load")
