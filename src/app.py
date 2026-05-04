
import os
import streamlit as st
import numpy as np
import time
from typing import Any
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

st.set_page_config(
    page_title="Pneumonia/Opacity Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Source+Sans+3:wght@400;500;600&display=swap');

:root {
    --bg-1: #0b1220;
    --bg-2: #101827;
    --surface: rgba(17, 24, 39, 0.88);
    --surface-strong: #0f172a;
    --card: rgba(17, 24, 39, 0.72);
    --text: #e5e7eb;
    --muted: #9ca3af;
    --accent: #38bdf8;
    --accent-2: #22c55e;
    --warning: #f97316;
    --border: rgba(148, 163, 184, 0.22);
    --shadow: 0 12px 30px rgba(2, 6, 23, 0.35);
}

.stApp {
    background:
        radial-gradient(800px 400px at 8% -10%, rgba(56, 189, 248, 0.12), transparent 60%),
        radial-gradient(900px 600px at 100% 10%, rgba(34, 197, 94, 0.10), transparent 60%),
        linear-gradient(180deg, var(--bg-1), var(--bg-2));
    color: var(--text);
}

html, body, [class*="css"] {
    font-family: "Source Sans 3", "Segoe UI", sans-serif;
}

h1, h2, h3, h4 {
    font-family: "Space Grotesk", "Segoe UI", sans-serif;
    letter-spacing: 0.2px;
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero {
    background: linear-gradient(120deg, rgba(56, 189, 248, 0.12), rgba(34, 197, 94, 0.08));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.5rem 2rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.75rem;
}

.hero-badge {
    display: inline-block;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.2rem;
    color: var(--muted);
    margin-bottom: 0.5rem;
}

.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
}

.hero-subtitle {
    margin-top: 0.35rem;
    font-size: 1rem;
    color: var(--muted);
}

.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 0.5rem 0;
}

.section-subtitle {
    color: var(--muted);
    margin-bottom: 1rem;
}

.divider {
    height: 1px;
    background: var(--border);
    margin: 1.25rem 0 1.5rem 0;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(17, 24, 39, 0.96));
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: var(--text);
}

div[data-testid="stFileUploader"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.75rem;
    box-shadow: var(--shadow);
}

.stImage img {
    border-radius: 14px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}

div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    box-shadow: var(--shadow);
}

div[data-testid="stMetric"] label {
    color: var(--muted);
    font-weight: 600;
    letter-spacing: 0.3px;
}

div[data-testid="stMetricValue"] {
    color: var(--text);
}

div[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}

div[data-testid="stExpander"] details {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.5rem 0.75rem;
}

div[data-testid="stExpander"] summary {
    color: var(--text);
    font-weight: 600;
}

div[data-testid="stProgress"] > div {
    background-color: rgba(148, 163, 184, 0.2);
}

div[data-testid="stProgress"] > div > div {
    background-color: var(--accent);
}

#MainMenu {visibility: hidden;}
header[data-testid="stHeader"] {visibility: hidden;}
div[data-testid="stToolbar"] {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ["normal", "opacity"]



@st.cache_resource
def load_trained_model(model_path: str = MODEL_PATH):
    """Load the trained Keras model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return load_model(model_path)


def preprocess_image_file(file) -> Any:
    """Preprocess an uploaded image file for model inference."""
    img = image.load_img(file, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array



def predict_image(model, img_array: np.ndarray):
    """Return the predicted label and confidence for one image array."""
    probability = float(model.predict(img_array, verbose=0)[0][0])
    predicted_index = 1 if probability > 0.5 else 0
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = probability if predicted_index == 1 else 1 - probability
    return predicted_label, confidence, probability


st.sidebar.title("🫁 About")
st.sidebar.info(
    "This application uses a deep learning model (ResNet50) "
    "to classify chest X-ray images as either **Normal** or exhibiting **Opacity** (which can indicate conditions such as Pneumonia)."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Instructions:**")
st.sidebar.markdown("1. Upload a chest X-ray image (JPG, JPEG, PNG).")
st.sidebar.markdown("2. The model will process the image.")
st.sidebar.markdown("3. View the prediction and confidence score.")

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">Clinical Imaging - AI Triage</div>
        <div class="hero-title">Chest X-ray Opacity Classifier</div>
        <div class="hero-subtitle">Upload a chest X-ray image to evaluate whether it appears normal or shows signs of opacity.</div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-title">Image Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Accepted formats: JPG, JPEG, PNG</div>', unsafe_allow_html=True)
    file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
    
if file is not None:
    with col1:
        st.image(file, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Model confidence and probability summary</div>', unsafe_allow_html=True)
        with st.spinner("Analyzing image..."):
            try:
                time.sleep(0.5) # Slight delay for UI UX
                model = load_trained_model()
                img_array = preprocess_image_file(file)
                label, confidence, raw_probability = predict_image(model, img_array)
                
                if label == "normal":
                    st.success("✅ **Prediction: Normal**")
                else:
                    st.error("⚠️ **Prediction: Opacity Detected**")
                
                st.metric(label="Confidence", value=f"{confidence:.2%}")
                
                with st.expander("More Details"):
                    st.write(f"**Model Raw Probability (Opacity):** `{raw_probability:.4f}`")
                    st.progress(raw_probability, text="Opacity Probability")
                    
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")


