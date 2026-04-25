import argparse
import os

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input



BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ["normal", "opacity"]


def load_trained_model(model_path: str = MODEL_PATH):
    """Load the trained Keras model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return load_model(model_path)


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess an image for model inference."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array


def predict_image(model, image_path: str):
    """Return the predicted label and confidence for one image."""
    processed_image = preprocess_image(image_path)
    probability = float(model.predict(processed_image, verbose=0)[0][0])

    predicted_index = 1 if probability > 0.5 else 0
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = probability if predicted_index == 1 else 1 - probability

    return predicted_label, confidence, probability


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a chest X-ray image."
    )
    parser.add_argument(
        "image_path",
        help="Path to the image file to classify.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="Optional path to a .keras model file.",
    )
    args = parser.parse_args()

    model = load_trained_model(args.model)
    label, confidence, raw_probability = predict_image(model, args.image_path)

    print(f"Image: {args.image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Raw opacity probability: {raw_probability:.4f}")


if __name__ == "__main__":
    main()
