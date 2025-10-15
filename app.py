import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
from PIL import Image
import json
import os

# Configure page
st.set_page_config(
    page_title="AI Flower Identification",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS to mimic flowers.stair.center
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* General Styling */
    body {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Elements */
    .stDeployButton, footer, #MainMenu, .stDecoration {
        display: none;
    }
    
    /* Main App Container */
    .main .block-container {
        max-width: 800px;
        padding: 2rem 1.5rem 4rem 1.5rem;
    }

    /* Header */
    .header {
        text-align: center;
        margin-bottom: 2.5rem;
    }
    .header h1 {
        font-weight: 700;
        font-size: 2.75rem;
        color: #1a1a1a;
        letter-spacing: -0.03em;
    }
    .header p {
        color: #5a6c7d;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Input Area */
    .input-area {
        background-color: #f8fafb;
        border: 2px dashed #d1d9e4;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .input-area:hover {
        border-color: #14b8a6;
        background-color: #f0fdfa;
    }

    /* Results Section */
    .results-container {
        margin-top: 2.5rem;
    }
    .prediction-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        color: #1a1a1a;
        margin-bottom: 1.5rem;
        border: 1px solid #e8ecef;
        text-align: center;
    }
    .prediction-card h2 {
        font-size: 2.25rem;
        font-weight: 700;
        color: #0d9488;
        text-transform: capitalize;
    }
    .confidence-bar-container {
        background: #e6f7f5;
        border-radius: 99px;
        height: 12px;
        margin: 1rem auto;
        width: 80%;
    }
    .confidence-bar {
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
        height: 100%;
        border-radius: 99px;
    }
    .confidence-text {
        font-weight: 600;
        color: #0f766e;
    }
    
    /* Other predictions */
    .other-predictions h3 {
        text-align: center;
        font-weight: 600;
        color: #3a4b5d;
        margin-bottom: 1rem;
    }
    .prediction-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        background: #f8fafb;
        border: 1px solid #e8ecef;
    }
    .prediction-name {
        font-weight: 500;
        color: #2d3748;
        text-transform: capitalize;
    }
    .prediction-confidence {
        font-weight: 600;
        color: #5a6c7d;
        min-width: 50px;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_resources():
    """Load the model and class names, caching them for performance."""
    try:
        with open("cat_to_name.json", "r") as f:
            class_names_map = json.load(f)
        flower_classes = ["" for _ in range(102)]
        for key, name in class_names_map.items():
            index = int(key) - 1
            if 0 <= index < 102:
                flower_classes[index] = name
        
        for i, name in enumerate(flower_classes):
            if name == "":
                flower_classes[i] = f"Unnamed Class {i+1}"
                
    except Exception as e:
        st.error(f"❌ Failed to load or process class names from 'cat_to_name.json': {e}")
        return None, None
        
    model_path_keras = 'densenet201_oxfordflowers_best.keras'
    model_path_h5 = 'densenet201_oxfordflowers_best.h5'
    model_path = None

    if os.path.exists(model_path_keras):
        model_path = model_path_keras
    elif os.path.exists(model_path_h5):
        model_path = model_path_h5
        
    if model_path:
        try:
            model= tf.keras.models.load_model(model_path)
            return model, flower_classes
        except Exception as e:
            st.error(f"❌ Error loading model file '{model_path}': {str(e)}")
            return None, None
    else:
        st.error("❌ Could not find the model file.")
        st.error(f"Please ensure '{model_path_keras}' or '{model_path_h5}' is in the project folder.")
        return None, None

# --- IMAGE PROCESSING & PREDICTION ---
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocesses the image for the DenseNet201 model."""
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)

def predict_flower(model, image, flower_classes):
    """Makes a prediction and returns the top 5 results."""
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)[0]
    
    # Get top 5 predictions
    top_indices = np.argsort(predictions)[-5:][::-1]
    top_results = [
        (flower_classes[i], float(predictions[i])) for i in top_indices
    ]
    return top_results

# --- MAIN APP LAYOUT ---
# Load model and class names once
model, flower_classes = load_resources()

if not model or not flower_classes:
    st.stop()

# Header
st.markdown("""
<div class="header">
    <h1>AI Flower Identifier</h1>
    <p>Upload a photo to identify one of 102 flower types.</p>
</div>
""", unsafe_allow_html=True)

# Display the file uploader
st.markdown('<div class="input-area">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drag & Drop or Click to Upload",
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)


# Perform prediction and display results
if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file).convert("RGB")
    st.image(image_to_process, caption="Your Image", use_container_width=True)
    
    with st.spinner("🧠 AI is analyzing the flower..."):
        top_results = predict_flower(model, image_to_process, flower_classes)

    st.markdown('<div class="results-container">', unsafe_allow_html=True)

    # Display the top prediction in a special card
    top_flower, top_confidence = top_results[0]
    st.markdown(f"""
    <div class="prediction-card">
        <h2>{top_flower}</h2>
        <div class="confidence-bar-container">
            <div class="confidence-bar" style="width: {top_confidence:.1%};"></div>
        </div>
        <div class="confidence-text">Confidence: {top_confidence:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

    # Display other predictions
    if len(top_results) > 1:
        st.markdown('<div class="other-predictions"><h3>Other possibilities:</h3>', unsafe_allow_html=True)
        for flower_name, confidence in top_results[1:]:
            st.markdown(f"""
            <div class="prediction-item">
                <span class="prediction-name">{flower_name}</span>
                <span class="prediction-confidence">{confidence:.1%}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)