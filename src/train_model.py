# Model Training Script: ResNet50 Transfer Learning
from sklearn.utils.class_weight import compute_class_weight
from keras.applications import ResNet50
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

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
base_model.trainable = False

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
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
    class_weight=class_weight_dict
)
# Fine-tuning (optional: you can keep or remove class_weight here)
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
history_finetune = model.fit(train_generator, validation_data=val_generator, epochs=5)

model.save('models/final_model.keras')
