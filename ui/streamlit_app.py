# ui/streamlit_app.py

import streamlit as st
import requests
from PIL import Image

# Configuration
BACKEND_URL = "http://localhost:8080"
API_URL = f"{BACKEND_URL}/predict-file"

st.set_page_config(page_title="Gender Classifier", page_icon="🧠", layout="centered")

st.title("🧠 Gender Classifier")
st.markdown("Upload a facial image to predict gender with AI.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {len(uploaded_file.getvalue())} bytes")

    if st.button("🔮 Predict Gender", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                    elif 'predicted_gender' in result:
                        predicted_gender = result['predicted_gender'].capitalize()
                        confidence = result.get('confidence', 0)
                        gender_emoji = "👨" if predicted_gender.lower() == "male" else "👩"
                        
                        st.success("✅ Prediction Complete!")
                        st.markdown(f"## {gender_emoji} **{predicted_gender}**")
                        st.markdown(f"**Confidence:** {confidence:.1%}")
                        st.progress(confidence)
                        
                        if 'probabilities' in result:
                            probs = result['probabilities']
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("👩 Female", f"{probs.get('female', 0):.1%}")
                            with col2:
                                st.metric("👨 Male", f"{probs.get('male', 0):.1%}")
                    else:
                        st.error("❌ Unexpected response format")
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                        
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to API at {BACKEND_URL}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Simple status check
if st.button("Check API Status"):
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=5)
        if response.status_code == 200:
            st.success("✅ API is online")
        else:
            st.error("❌ API is not responding")
    except:
        st.error("❌ API is not accessible")

# Retrain option
if st.button("Retrain Model"):
    with st.spinner("Retraining..."):
        try:
            response = requests.post(f"{BACKEND_URL}/retrain", timeout=300)
            if response.status_code == 200:
                st.success("✅ Retraining started")
            else:
                st.error("❌ Retraining failed")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.markdown("Built with Streamlit and FastAPI")