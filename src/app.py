
"""Chest X-ray Opacity Classification App

This Streamlit web app uses a ResNet50 deep learning model to classify chest X-rays as 
either Normal or showing Opacity (indicating possible pneumonia or other conditions).

Key features:
- Real-time image classification with confidence scores
- Grad-CAM heatmaps to show which image regions influenced predictions
- Privacy-aware audit logging (file hash + model output only, no raw images stored)
- Image validation to reject non-X-ray uploads
- Modern dark-themed UI with responsive layout
"""

import os
import time
import streamlit as st
import numpy as np
from keras.models import load_model
import tensorflow as tf
from data_preprocessing import is_likely_chest_xray
from utils import (
    preprocess_image_file,
    predict_image,
    get_image_bytes,
    build_gradcam_heatmap,
    overlay_gradcam,
    log_prediction_event,
)

st.set_page_config(
    page_title="Pneumonia/Opacity Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)
from custom_css import inject_custom_css

# Inject the shared custom CSS
inject_custom_css()

# ===== Configuration & Paths =====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
LOG_DIR = os.path.join(BASE_DIR, "logs")
AUDIT_LOG_PATH = os.path.join(LOG_DIR, "prediction_audit.csv")

# ResNet50 expects 224x224 images; BATCH_SIZE would go here if doing batch inference
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Model output: class 0 = normal, class 1 = opacity (pneumonia, fluid, etc.)
CLASS_NAMES = ["normal", "opacity"]



# ===== Model Loading =====
@st.cache_resource
def load_trained_model(model_path: str = MODEL_PATH):
    """Load and cache the trained Keras model.
    
    Cached so it only loads once per session (critical for performance).
    Raises FileNotFoundError if the model file doesn't exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return load_model(model_path)











# ===== Sidebar (Context & Instructions) =====
st.sidebar.title("🫁 About")
st.sidebar.info(
    "This application uses a deep learning model (ResNet50) "
    "to classify chest X-ray images as either **Normal** or exhibiting **Opacity** (which can indicate conditions such as Pneumonia)."
)
st.sidebar.warning(
    "Triage aid only: this is not a clinical diagnosis and should not replace radiologist review or medical judgment."
)
st.sidebar.info(
    "Uploads are logged in a privacy-aware audit trail using a file hash and model outputs only. Raw images are not stored."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Instructions:**")
st.sidebar.markdown("1. Upload a chest X-ray image (JPG, JPEG, PNG).")
st.sidebar.markdown("2. The model will process the image.")
st.sidebar.markdown("3. View the prediction, confidence, and model focus map.")

# ===== UI Layout: Hero Section =====
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

st.warning(
    "This tool is a triage aid only. It can highlight image regions that influenced the prediction, but it does not provide a diagnosis."
)

# ===== UI Layout: Two-Column Design =====
# Left column: image upload and preview | Right column: model prediction and Grad-CAM heatmap
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-title">Image Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Accepted formats: JPG, JPEG, PNG</div>', unsafe_allow_html=True)
    file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
    
if file is not None:
    image_bytes = get_image_bytes(file)

    # ===== Image Validation =====
    # Quick check: does this look like a chest X-ray? (grayscale, right dimensions, etc.)
    # Rejects obvious mismatches before wasting compute on invalid images.
    try:
        file.seek(0)
        valid_xray = is_likely_chest_xray(file)
    except Exception:
        valid_xray = False
    finally:
        try:
            file.seek(0)
        except Exception:
            pass

    if not valid_xray:
        # Image failed validation — show it anyway so user sees what was rejected
        with col1:
            st.image(file, caption="Uploaded Image (preview)", use_container_width=True)
            st.error(
                "Uploaded image doesn't look like a chest X-ray. Please upload a frontal chest radiograph (grayscale)."
            )
        with col2:
            st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">Model confidence and probability summary</div>', unsafe_allow_html=True)
            st.warning("Analysis skipped: uploaded image failed basic X-ray validation.")
            log_prediction_event(
                image_bytes=image_bytes,
                filename=getattr(file, "name", "unknown"),
                validation_status="failed_validation",
                label="skipped",
                confidence=0.0,
                raw_probability=0.0,
                gradcam_available=False,
                log_dir=LOG_DIR,
                audit_log_path=AUDIT_LOG_PATH,
            )
    else:
        # ===== Inference Pipeline =====
        # Valid X-ray: show on left, run model, display prediction + heatmap on right
        with col1:
            st.image(file, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">Model confidence and probability summary</div>', unsafe_allow_html=True)
            with st.spinner("Analyzing image..."):
                try:
                    time.sleep(0.5)  # Brief pause for UX polish
                    
                    # Step 1: Load model (cached after first call)
                    model = load_trained_model()
                    
                    # Step 2: Preprocess image for ResNet50
                    file.seek(0)
                    img_array = preprocess_image_file(file)
                    
                    # Step 3: Get prediction
                    label, confidence, raw_probability = predict_image(model, img_array, CLASS_NAMES)
                    
                    # Step 4: Generate interpretability heatmap
                    heatmap = build_gradcam_heatmap(model, img_array)
                    overlay_image = overlay_gradcam(image_bytes, heatmap, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)
                    
                    # ===== Display Results =====
                    if label == "normal":
                        st.success("✅ **Prediction: Normal**")
                    else:
                        st.error("⚠️ **Prediction: Opacity Detected**")
                    
                    # Show confidence as percentage (0-100%)
                    st.metric(label="Confidence", value=f"{confidence:.2%}")
                    
                    # Show heatmap overlaid on original X-ray (bright = model attention)
                    st.image(overlay_image, caption="Grad-CAM focus overlay", use_container_width=True)
                    
                    # Advanced metrics for clinicians who want more detail
                    with st.expander("More Details"):
                        st.write(f"**Model Raw Probability (Opacity):** `{raw_probability:.4f}`")
                        st.progress(raw_probability, text="Opacity Probability")
                        st.caption("Brighter regions indicate the parts of the image that contributed more to the prediction.")

                    # ===== Log for Audit & Monitoring =====
                    log_prediction_event(
                        image_bytes=image_bytes,
                        filename=getattr(file, "name", "unknown"),
                        validation_status="passed_validation",
                        label=label,
                        confidence=confidence,
                        raw_probability=raw_probability,
                        gradcam_available=True,
                        log_dir=LOG_DIR,
                        audit_log_path=AUDIT_LOG_PATH,
                    )
                        
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")


