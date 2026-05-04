"""Utility functions for chest X-ray classification.

This module contains core ML and data handling functions separated from the UI layer.
Includes: image preprocessing, model inference, Grad-CAM visualization, and audit logging.
"""

import csv
import hashlib
import io
import os
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from PIL import Image
import matplotlib.cm as cm

# These will be set by the importing app module
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ["normal", "opacity"]
LOG_DIR = None
AUDIT_LOG_PATH = None


def preprocess_image_file(file, img_height: int = 224, img_width: int = 224) -> Any:
    """Preprocess an uploaded image file for ResNet50 model inference.
    
    Resizes to 224x224 and applies ImageNet normalization.
    """
    img = image.load_img(file, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_image(model, img_array: np.ndarray, class_names: Optional[list] = None) -> tuple:
    """Get prediction label and confidence for an image array.
    
    Returns: (predicted_label, confidence, raw_probability)
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    probability = float(model.predict(img_array, verbose=0)[0][0])
    predicted_index = 1 if probability > 0.5 else 0
    predicted_label = class_names[predicted_index]
    confidence = probability if predicted_index == 1 else 1 - probability
    return predicted_label, confidence, probability


def get_image_bytes(uploaded_file) -> bytes:
    """Extract raw bytes from uploaded file without saving to disk."""
    uploaded_file.seek(0)
    return uploaded_file.getvalue()


def build_gradcam_heatmap(model, img_array: np.ndarray) -> np.ndarray:
    """Generate a Grad-CAM heatmap showing regions that influenced the prediction.
    
    Brighter regions = model paid more attention to that area when predicting opacity.
    Returns normalized heatmap (0-1 range).
    """
    # Find the last convolutional layer (needed for Grad-CAM)
    conv_layer_name = None
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                conv_layer_name = layer.name
                break
        except Exception:
            continue

    if conv_layer_name is None:
        raise ValueError("No convolutional layer found for Grad-CAM.")

    # Build a model that outputs both conv layer output and final prediction
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(conv_layer_name).output, model.output],
    )

    # Compute gradients: how much did each conv output contribute to the prediction?
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # Loss w.r.t. opacity probability

    grads = tape.gradient(loss, conv_outputs)
    
    # Average importance across spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    
    # Weight each conv channel by its average gradient
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)  # ReLU to remove negative contributions
    
    # Normalize to 0-1 range
    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(max_value > 0, heatmap / max_value, heatmap)
    
    return heatmap.numpy()


def overlay_gradcam(
    image_bytes: bytes, 
    heatmap: np.ndarray, 
    alpha: float = 0.42,
    img_width: int = 224,
    img_height: int = 224
) -> Image.Image:
    """Overlay a Grad-CAM heatmap on the original X-ray for visualization.
    
    Args:
        image_bytes: Raw image file bytes
        heatmap: Normalized heatmap (0-1)
        alpha: Blend factor (0=original image, 1=pure heatmap)
        img_width, img_height: Target image dimensions
    
    Returns: PIL Image with heatmap overlay
    """
    # Load and resize base image
    base_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    base_image = base_image.resize((img_width, img_height))
    base_array = np.asarray(base_image).astype(np.float32) / 255.0

    # Convert heatmap to color (jet colormap: blue=low, red=high)
    heatmap_rgb = cm.get_cmap("jet")(heatmap)[..., :3]
    heatmap_rgb = np.uint8(255 * heatmap_rgb)
    heatmap_image = Image.fromarray(heatmap_rgb).resize((img_width, img_height))
    heatmap_array = np.asarray(heatmap_image).astype(np.float32) / 255.0

    # Blend: original + heatmap overlay
    blended = (1 - alpha) * base_array + alpha * heatmap_array
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)


def log_prediction_event(
    image_bytes: bytes,
    filename: str,
    validation_status: str,
    label: str,
    confidence: float,
    raw_probability: float,
    gradcam_available: bool,
    log_dir: Optional[str] = None,
    audit_log_path: Optional[str] = None,
) -> None:
    """Log a prediction event to CSV for audit trail and drift monitoring.
    
    Privacy-first approach: only logs file hash (not image), prediction, and confidence.
    Useful for detecting model drift, performance degradation, and auditing predictions.
    
    Args:
        image_bytes: Raw image bytes (used to compute file hash)
        filename: Original filename
        validation_status: "passed_validation" or "failed_validation"
        label: Predicted label ("normal" or "opacity")
        confidence: Model confidence (0-1)
        raw_probability: Raw opacity probability from model
        gradcam_available: Whether Grad-CAM heatmap was generated
        log_dir: Path to logs directory
        audit_log_path: Path to audit CSV file
    """
    if log_dir is None:
        log_dir = LOG_DIR
    if audit_log_path is None:
        audit_log_path = AUDIT_LOG_PATH
    
    if log_dir is None:
        raise ValueError("log_dir must be set")
    if audit_log_path is None:
        raise ValueError("audit_log_path must be set")
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Hash the image file (privacy-preserving identifier)
    file_hash = hashlib.sha256(image_bytes).hexdigest()
    file_ext = os.path.splitext(filename)[1].lower().lstrip(".") if filename else "unknown"
    
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "file_hash": file_hash,
        "file_extension": file_ext or "unknown",
        "validation_status": validation_status,
        "prediction": label,
        "confidence": f"{confidence:.6f}",
        "opacity_probability": f"{raw_probability:.6f}",
        "gradcam_available": str(bool(gradcam_available)),
    }
    
    # Append to CSV (create header if file doesn't exist)
    file_exists = os.path.exists(audit_log_path)
    with open(audit_log_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
