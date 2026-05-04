# Data Preprocessing for Chest X-ray Classification (ResNet50 model-ready)
# This module creates data generators for training, validation, and testing.
# It handles image resizing, augmentation, and normalization compatible with ResNet50.

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np

# Define base directory structure for data organization
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Image parameters: ResNet50 expects 224x224 RGB input
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32  # Balance between memory efficiency and gradient stability

# ========== Data Augmentation for Training ==========
# Augmentation only applied to training data to simulate realistic variations
# in X-ray acquisition (patient positioning, angle, centering, distance).
# This artificially increases dataset diversity and reduces overfitting.
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # ImageNet normalization for ResNet50
    rotation_range=15,                        # ±15° rotation (patient positioning)
    width_shift_range=0.1,                    # ±10% horizontal shift
    height_shift_range=0.1,                   # ±10% vertical shift
    shear_range=0.1,                          # ±10% shear (angle variation)
    zoom_range=0.1,                           # ±10% zoom (distance variation)
    horizontal_flip=True,                     # Symmetric chest X-rays; flip is realistic
    fill_mode='nearest'                       # Avoid black borders during transforms
)

# ========== Validation and Test Data: No Augmentation ==========
# Only ResNet50 normalization applied; no synthetic augmentation.
# Validation/test sets must reflect real-world data distribution.
val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# ========== Training Generator ==========
# Reads images from TRAIN_DIR/{normal, opacity} subdirectories.
# Shuffles batches to prevent order bias in gradient descent.
train_generator = train_datagen.flow_from_directory(
	TRAIN_DIR,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	batch_size=BATCH_SIZE,
	class_mode='binary',  # Binary classification: normal (0) vs. opacity (1)
	color_mode='rgb',     # ResNet50 expects 3-channel RGB input
	shuffle=True          # Randomize batch order for better convergence
)

# ========== Validation Generator ==========
# Used during training for monitoring performance and early stopping.
# No shuffling to ensure consistent evaluation across epochs.
val_generator = val_test_datagen.flow_from_directory(
	VAL_DIR,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	batch_size=BATCH_SIZE,
	class_mode='binary',
	color_mode='rgb',
	shuffle=False
)

# ========== Test Generator ==========
# Held-out test set for final model evaluation.
# Reflects real-world inference conditions (no augmentation, no shuffling).
test_generator = val_test_datagen.flow_from_directory(
	TEST_DIR,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	batch_size=BATCH_SIZE,
	class_mode='binary',
	color_mode='rgb',
	shuffle=False
)

# ========== Debug Information ==========
# Print class mapping and sample batch shape for verification.
if __name__ == "__main__":
	print("Class indices:", train_generator.class_indices)
	batch = next(train_generator)
	print("Batch X shape:", batch[0].shape)  # (batch_size, 224, 224, 3)
	print("Batch y shape:", batch[1].shape)  # (batch_size, 1) for binary classification


def is_likely_chest_xray(image_input, colorfulness_threshold: float = 15.0) -> bool:
	"""Basic heuristic check whether an image is likely a chest X-ray.

	This function uses lightweight, fast heuristics (no ML) to filter out
	obvious non-X-ray uploads such as color photos. Heuristics used:
	  - Colorfulness metric (Hasler & Suesstrunk): chest X-rays are
		typically grayscale/low-color images.
	  - Aspect ratio sanity check: extremely wide/tall images are
		unlikely to be standard radiographs.

	Args:
		image_input: a PIL.Image.Image instance or a file-like object
		colorfulness_threshold: threshold above which image is considered
			"colorful" (i.e., likely not an X-ray).

	Returns:
		True if the image passes the heuristics and is likely a chest X-ray.
	"""
	# Accept both PIL images and file-like objects
	if not isinstance(image_input, Image.Image):
		try:
			img = Image.open(image_input)
		except Exception:
			return False
	else:
		img = image_input

	# Quick aspect ratio check
	width, height = img.size
	if width == 0 or height == 0:
		return False
	aspect = float(width) / float(height)
	if aspect > 3.0 or aspect < 0.25:
		return False

	# Colorfulness metric (fast, no external deps beyond numpy)
	try:
		arr = np.asarray(img.convert('RGB')).astype('float32')
	except Exception:
		return False

	# Compute rg and yb components
	R = arr[:, :, 0]
	G = arr[:, :, 1]
	B = arr[:, :, 2]
	rg = R - G
	yb = 0.5 * (R + G) - B

	std_rg = np.std(rg)
	std_yb = np.std(yb)
	mean_rg = np.mean(rg)
	mean_yb = np.mean(yb)

	colorfulness = np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)

	# If colorfulness is high, image is likely a color photo (not X-ray)
	if colorfulness > colorfulness_threshold:
		return False

	# Passed heuristics -> likely chest X-ray (or at least grayscale radiograph-like)
	return True

