# ui/streamlit_app.py

import streamlit as st
import requests
from PIL import Image
import io
import base64
import os

# Configuration - use Railway backend URL
BACKEND_URL = "https://sura-production.up.railway.app"
API_URL = f"{BACKEND_URL}/predict-file"

st.set_page_config(
    page_title="Gender Classifier", 
    page_icon="🧠",
    layout="centered"
)

# Header
st.title("🧠 Gender Classifier")
st.markdown("Upload a facial image, and our AI model will predict the gender with confidence scores.")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This model uses a trained CNN to classify gender from facial images.")
    st.write("**Accuracy:** ~80%")
    st.write("**Classes:** Male, Female")
    
    st.header("📊 Model Stats")
    st.metric("Overall Accuracy", "80%")
    st.metric("Avg Confidence", "87.6%")

# Main content
uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear facial image for best results"
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.write("**Image Details:**")
        st.write(f"📁 Filename: {uploaded_file.name}")
        st.write(f"📏 Size: {image.size}")
        st.write(f"🎨 Mode: {image.mode}")
        st.write(f"💾 File size: {len(uploaded_file.getvalue())} bytes")

    # Prediction button
    if st.button("🔮 Predict Gender", type="primary"):
        with st.spinner("🤖 Analyzing image..."):
            try:
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                
                # Prepare files for request
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                
                # Make API request
                response = requests.post(API_URL, files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Debug: Show the raw response
                    st.write("**Debug - API Response:**")
                    st.json(result)
                    
                    # Check if the response contains an error
                    if 'error' in result:
                        st.error(f"❌ Prediction Error: {result['error']}")
                    elif 'predicted_gender' in result:
                        # Display results
                        st.success("✅ Prediction Complete!")
                        
                        # Main prediction result
                        predicted_gender = result['predicted_gender'].capitalize()
                        confidence = result.get('confidence', 0)
                        
                        # Use different emojis based on prediction
                        gender_emoji = "👨" if predicted_gender.lower() == "male" else "👩"
                        
                        st.markdown(f"## {gender_emoji} Predicted Gender: **{predicted_gender}**")
                        
                        # Confidence meter
                        st.markdown(f"### 🎯 Confidence: **{confidence:.1%}**")
                        st.progress(confidence)
                        
                        # Detailed probabilities
                        if 'probabilities' in result:
                            st.markdown("### 📊 Detailed Probabilities:")
                            probs = result['probabilities']
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(
                                    "👩 Female", 
                                    f"{probs.get('female', 0):.1%}",
                                    delta=None
                                )
                            
                            with col2:
                                st.metric(
                                    "👨 Male", 
                                    f"{probs.get('male', 0):.1%}",
                                    delta=None
                                )
                        
                        # Confidence interpretation
                        if confidence >= 0.8:
                            st.info("🔥 High confidence prediction!")
                        elif confidence >= 0.6:
                            st.info("✨ Good confidence prediction!")
                        else:
                            st.warning("⚠️ Low confidence - image quality might affect results.")
                    else:
                        st.error("❌ Unexpected API response format")
                        st.write("Expected 'predicted_gender' in response, but got:")
                        st.json(result)
                
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.write("Response text:")
                    st.code(response.text)
                        
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to API. Make sure the FastAPI server is running at {BACKEND_URL}")
                st.info("Backend URL: " + BACKEND_URL)
            except requests.exceptions.Timeout:
                st.error("⏰ Request timed out. The server might be overloaded.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and FastAPI | Gender Classification AI")

# Instructions
with st.expander("📖 How to use"):
    st.markdown("""
    1. **Upload an image** using the file uploader above
    2. **Make sure it's a clear facial image** for best results
    3. **Click 'Predict Gender'** to get the AI prediction
    4. **View the results** with confidence scores and probabilities
    
    **Tips for better results:**
    - Use clear, well-lit photos
    - Ensure the face is clearly visible
    - Avoid heavily filtered or edited images
    - Higher resolution images generally work better
    """)

# API Status check
with st.expander("🔧 API Status"):
    if st.button("Check API Status"):
        try:
            health_response = requests.get(f"{BACKEND_URL}/", timeout=5)
            if health_response.status_code == 200:
                st.success("✅ API is online and responding")
                st.json(health_response.json())
            else:
                st.warning(f"⚠️ API responded with status: {health_response.status_code}")
        except Exception as e:
            st.error(f"❌ API is not accessible: {str(e)}")

# Retrain Model button
with st.expander("🔄 Model Management"):
    if st.button("Retrain Model"):
        with st.spinner("Retraining model..."):
            try:
                response = requests.post(f"{BACKEND_URL}/retrain", timeout=300)  # 5 min timeout
                if response.status_code == 200:
                    st.success("Model retrained successfully!")
                    st.json(response.json())
                else:
                    st.error(f"Retraining failed. Status: {response.status_code}")
                    st.code(response.text)
            except Exception as e:
                st.error(f"Retraining error: {str(e)}")