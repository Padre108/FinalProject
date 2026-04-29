
import os
import streamlit as st
import numpy as np
from typing import Any
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

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


st.title("Chest X-ray Opacity Classifier")
st.write("Upload a chest X-ray image to predict if it is normal or has opacity.")

file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
if file is not None:
    st.image(file, caption="Uploaded Image")
    try:
        model = load_trained_model()
        img_array = preprocess_image_file(file)
        label, confidence, raw_probability = predict_image(model, img_array)
        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.2%}")
        st.caption(f"Raw opacity probability: {raw_probability:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")


