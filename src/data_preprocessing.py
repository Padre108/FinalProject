# Data Preprocessing for Chest X-ray Classification (ResNet-ready)
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Paths to your data folders
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
	rescale=1./255,
	rotation_range=15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.1,
	zoom_range=0.1,
	horizontal_flip=True,
	fill_mode='nearest'
)

# Validation and test data: only rescale
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Training generator
train_generator = train_datagen.flow_from_directory(
	TRAIN_DIR,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	batch_size=BATCH_SIZE,
	class_mode='binary',
	color_mode='rgb',
	shuffle=True
)

# Validation generator
val_generator = val_test_datagen.flow_from_directory(
	VAL_DIR,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	batch_size=BATCH_SIZE,
	class_mode='binary',
	color_mode='rgb',
	shuffle=False
)

# Test generator
test_generator = val_test_datagen.flow_from_directory(
	TEST_DIR,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	batch_size=BATCH_SIZE,
	class_mode='binary',
	color_mode='rgb',
	shuffle=False
)

if __name__ == "__main__":
	print("Class indices:", train_generator.class_indices)
	batch = next(train_generator)
	print("Batch X shape:", batch[0].shape)
	print("Batch y shape:", batch[1].shape)
