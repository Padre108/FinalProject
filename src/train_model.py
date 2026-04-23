# Model Training Script: ResNet50 Transfer Learning
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Import data generators from data_preprocessiong.py

from data_preprocessing import train_generator, val_generator

# --- Debug: Print class indices and class distribution ---
print("Class indices:", train_generator.class_indices)
import numpy as np
unique, counts = np.unique(train_generator.classes, return_counts=True)
print("Samples per class (train):", dict(zip(unique, counts)))
unique_val, counts_val = np.unique(val_generator.classes, return_counts=True)
print("Samples per class (val):", dict(zip(unique_val, counts_val)))

# Model parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 1  # Binary classification

# Load ResNet50 as feature extractor
base_model = ResNet50(
	weights='imagenet',
	include_top=False,
	input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base_model.trainable = False  # Freeze backbone for initial training

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
	optimizer=Adam(learning_rate=1e-4),
	loss='binary_crossentropy',
	metrics=['accuracy']
)

# Callbacks
checkpoint = ModelCheckpoint(
	'models/best_model.keras',
	monitor='val_accuracy',
	save_best_only=True,
	mode='max',
	verbose=1
)
early_stop = EarlyStopping(
	monitor='val_accuracy',
	patience=5,
	restore_best_weights=True
)

# Training
EPOCHS = 15
history = model.fit(
	train_generator,
	validation_data=val_generator,
	epochs=EPOCHS,
	callbacks=[checkpoint, early_stop]
)

# Optionally, unfreeze some layers and fine-tune
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
history_finetune = model.fit(train_generator, validation_data=val_generator, epochs=5)

# Save final model
model.save('models/final_model.keras')
