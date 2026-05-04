# Model Training Script: ResNet50 Transfer Learning
# This module implements a two-stage training pipeline:
# Stage 1: Train classifier head on frozen ResNet50 backbone (15 epochs)
# Stage 2: Fine-tune last 30 ResNet blocks with lower learning rate (8 epochs)
# Decision threshold is calibrated on validation set to optimize recall (95% target).

import os
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from keras.metrics import AUC, BinaryAccuracy, Precision, Recall
from keras.models import Model
from keras.optimizers import Adam

from data_preprocessing import test_generator, train_generator, val_generator

# ========== Data Distribution Analysis ==========
# Print class balance to guide class weighting strategy.
# Imbalanced datasets benefit from balanced class weights.
print("Class indices:", train_generator.class_indices)
unique, counts = np.unique(train_generator.classes, return_counts=True)
print("Samples per class (train):", dict(zip(unique, counts)))
unique_val, counts_val = np.unique(val_generator.classes, return_counts=True)
print("Samples per class (val):", dict(zip(unique_val, counts_val)))

# ========== Model Configuration ==========
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 1                          # Binary classification output
POSITIVE_CLASS_NAME = "opacity"          # Label of positive class (lesion/opacity)
TARGET_RECALL = 0.95                     # Threshold optimization target (95% sensitivity)
HEAD_EPOCHS = 15                         # Initial training of classification head
FINE_TUNE_EPOCHS = 8                     # Fine-tuning of backbone layers
UNFREEZE_LAST_N = 30                     # Number of ResNet50 layers to unfreeze
POSITIVE_CLASS_WEIGHT_BOOST = 1.25       # Boost weight of positive class 25%


def build_metrics():
    """Build list of metrics to monitor during training.
    
    Returns:
        list: Metrics including accuracy, recall, precision, ROC-AUC, and PR-AUC.
    
    Justification:
        - Accuracy: Overall correctness (but misleading on imbalanced data)
        - Recall: True positive rate (critical for medical: missed cases are costly)
        - Precision: False positive rate (controls unnecessary radiologist reviews)
        - ROC-AUC & PR-AUC: Threshold-independent performance evaluation
    """
    return [
        BinaryAccuracy(name="accuracy"),
        Recall(name="recall"),
        Precision(name="precision"),
        AUC(name="roc_auc"),
        AUC(name="pr_auc", curve="PR"),
    ]


def compile_model(model_to_compile, learning_rate):
    """Compile model with Adam optimizer and binary crossentropy loss.
    
    Args:
        model_to_compile: Keras model to compile
        learning_rate (float): Learning rate for Adam optimizer
    
    Justification of choices:
        - Adam: Adaptive learning rate per parameter; handles sparse gradients well
        - Binary Crossentropy: Standard loss for binary classification
        - Metrics: Comprehensive evaluation across all aspects
    """
    model_to_compile.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=build_metrics(),
    )


def select_threshold_for_recall(y_true, y_prob, target_recall):
    """Calibrate decision threshold to achieve target recall on validation set.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        target_recall (float): Target recall value (e.g., 0.95)
    
    Returns:
        tuple: (optimal_threshold, achieved_recall, achieved_precision)
    
    Justification:
        In medical imaging, recall (sensitivity) is prioritized: missing a lesion
        is worse than a false positive (which gets reviewed). This function selects
        the threshold that maximizes precision among thresholds meeting the recall constraint.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0

    precision_curve = precisions[:-1]
    recall_curve = recalls[:-1]
    candidate_idx = np.where(recall_curve >= target_recall)[0]

    if len(candidate_idx) > 0:
        best = candidate_idx[np.argmax(precision_curve[candidate_idx])]
    else:
        best = int(np.argmax(recall_curve))

    return float(thresholds[best]), float(recall_curve[best]), float(precision_curve[best])


def evaluate_with_threshold(model, generator, split_name, threshold):
    """Evaluate model on a data split using a specific decision threshold.
    
    Args:
        model: Trained model instance
        generator: Data generator (train, val, or test)
        split_name (str): Name for logging (e.g., "Validation")
        threshold (float): Decision threshold for binary classification
    
    Returns:
        tuple: (y_true, y_prob) for further analysis
    
    Prints:
        Confusion matrix, sensitivity, specificity, precision at given threshold.
    """
    generator.reset()
    y_true = generator.classes.astype(np.int32)
    y_prob = model.predict(generator).ravel()
    y_pred = (y_prob >= threshold).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-8)  # Recall for positive class (opacity)
    specificity = tn / (tn + fp + 1e-8)  # True negative rate
    precision = tp / (tp + fp + 1e-8)    # Positive predictive value

    print(f"\n[{split_name}] threshold={threshold:.4f}")
    print(f"[{split_name}] TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"[{split_name}] sensitivity/recall={sensitivity:.4f}")
    print(f"[{split_name}] specificity={specificity:.4f}")
    print(f"[{split_name}] precision={precision:.4f}")

    return y_true, y_prob


# ========== ARCHITECTURE: Transfer Learning with ResNet50 ==========
# Load ResNet50 pre-trained on ImageNet as a frozen feature extractor.
# ImageNet pre-training has learned general visual patterns (edges, textures)
# transferable to X-ray classification.
base_model = ResNet50(
    weights="imagenet",
    include_top=False,                    # Remove ImageNet classification head
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
)
base_model.trainable = False              # Freeze backbone for Stage 1

# ========== Custom Classification Head ==========
# Build a lightweight head on top of ResNet50 backbone features.
# GlobalAveragePooling: Compress spatial dims (7x7x2048 -> 2048)
# Dense + ReLU: Learn task-specific patterns
# Dropout: Regularization for limited medical data
# Dense(1, Sigmoid): Binary classification output [0, 1]
x = base_model.output
x = GlobalAveragePooling2D()(x)           # Global average pooling: reduce spatial dims
x = Dropout(0.5)(x)                       # Dropout: prevent co-adaptation
x = Dense(128, activation="relu")(x)      # Hidden layer: 128 neurons
x = Dropout(0.5)(x)                       # Dropout: additional regularization
output = Dense(NUM_CLASSES, activation="sigmoid")(x)  # Binary output: [0, 1]

# Create functional model
model = Model(inputs=base_model.input, outputs=output)
compile_model(model, learning_rate=1e-4)  # Higher learning rate for head initialization

# ========== Callbacks for Training Stability ==========
# ModelCheckpoint: Save best models based on val_recall and val_accuracy
# EarlyStopping: Stop if val_recall plateaus (patience=6 epochs)
# ReduceLROnPlateau: Reduce learning rate if val_recall stalls

checkpoint_recall = ModelCheckpoint(
    "models/best_model.keras",
    monitor="val_recall",                 # Primary metric: recall (sensitivity)
    save_best_only=True,
    mode="max",
    verbose=1,
)
checkpoint_accuracy = ModelCheckpoint(
    "models/best_model_accuracy.keras",
    monitor="val_accuracy",               # Secondary metric: overall accuracy
    save_best_only=True,
    mode="max",
    verbose=1,
)
early_stop = EarlyStopping(
    monitor="val_recall",
    mode="max",
    patience=6,                           # Stop if no improvement in 6 epochs
    restore_best_weights=True,            # Restore best weights when stopping
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_recall",
    mode="max",
    factor=0.5,                           # Multiply LR by 0.5
    patience=2,                           # After 2 epochs without improvement
    min_lr=1e-7,
    verbose=1,
)

# ========== Class Weighting ==========
# Address class imbalance: if dataset has 60% normal and 40% opacity,
# balanced weights upweight the minority class.
# Additional 1.25x boost to opacity class reflects medical priority:
# missing a lesion is costlier than a false alarm.
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes,
)
class_weight_dict = dict(enumerate(class_weights))
positive_idx = train_generator.class_indices.get(POSITIVE_CLASS_NAME)
if positive_idx is not None and positive_idx in class_weight_dict:
    class_weight_dict[positive_idx] *= POSITIVE_CLASS_WEIGHT_BOOST
print("Class weights:", class_weight_dict)

# ========== STAGE 1: Train Classifier Head Only ==========
# Backbone frozen; only train custom head (Global pooling, Dense, Dropout layers).
# Allows head to adapt to X-ray data without disrupting pre-trained features.
# Higher learning rate (1e-4) since we're starting from random initialization.
print("\n=== Stage 1: Training classifier head ===")
history_head = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=HEAD_EPOCHS,
    callbacks=[checkpoint_recall, checkpoint_accuracy, early_stop, reduce_lr],
    class_weight=class_weight_dict,  # Apply class weights to handle imbalance
)

# ========== STAGE 2: Fine-Tune Backbone ==========
# Unfreeze last 30 ResNet50 layers; keep early layers frozen (ImageNet is universal).
# Keep BatchNormalization layers frozen to maintain feature distribution stability.
# Lower learning rate (1e-5) to avoid disrupting pre-trained weights.
print("\n=== Stage 2: Fine-tuning backbone layers ===")
base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAST_N]:
    layer.trainable = False               # Keep early layers frozen
for layer in base_model.layers[-UNFREEZE_LAST_N:]:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False           # Freeze BN for stability

compile_model(model, learning_rate=1e-5)  # Lower LR for fine-tuning
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[checkpoint_recall, checkpoint_accuracy, early_stop, reduce_lr],
    class_weight=class_weight_dict,
)

# ========== Threshold Calibration ==========
# On validation set, find decision threshold that achieves 95% recall (sensitivity).
# This threshold balances detecting lesions (high recall) with acceptable precision.
print("\n=== Calibrating decision threshold ===")
val_generator.reset()
val_true = val_generator.classes.astype(np.int32)
val_prob = model.predict(val_generator).ravel()
decision_threshold, achieved_recall, achieved_precision = select_threshold_for_recall(
    val_true, val_prob, TARGET_RECALL
)
print(
    f"\nCalibrated threshold={decision_threshold:.4f} "
    f"(val recall={achieved_recall:.4f}, val precision={achieved_precision:.4f})"
)

# Save threshold for later inference
os.makedirs("models", exist_ok=True)
with open("models/decision_threshold.txt", "w", encoding="utf-8") as f:
    f.write(f"{decision_threshold:.6f}\n")

# ========== Evaluation on Validation and Test Sets ==========
print("\n=== Final Evaluation ===")
evaluate_with_threshold(model, val_generator, "Validation", decision_threshold)
evaluate_with_threshold(model, test_generator, "Test", decision_threshold)

# ========== Save Final Model ==========
model.save("models/final_model.keras")
print("\nModel saved to models/final_model.keras")

