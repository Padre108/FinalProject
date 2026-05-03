# Model Training Script: ResNet50 Transfer Learning
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

# --- Debug: Print class indices and class distribution ---
print("Class indices:", train_generator.class_indices)
unique, counts = np.unique(train_generator.classes, return_counts=True)
print("Samples per class (train):", dict(zip(unique, counts)))
unique_val, counts_val = np.unique(val_generator.classes, return_counts=True)
print("Samples per class (val):", dict(zip(unique_val, counts_val)))

# Model parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 1  # Binary classification
POSITIVE_CLASS_NAME = "opacity"
TARGET_RECALL = 0.95
HEAD_EPOCHS = 15
FINE_TUNE_EPOCHS = 8
UNFREEZE_LAST_N = 30
POSITIVE_CLASS_WEIGHT_BOOST = 1.25


def build_metrics():
    return [
        BinaryAccuracy(name="accuracy"),
        Recall(name="recall"),
        Precision(name="precision"),
        AUC(name="roc_auc"),
        AUC(name="pr_auc", curve="PR"),
    ]


def compile_model(model_to_compile, learning_rate):
    model_to_compile.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=build_metrics(),
    )


def select_threshold_for_recall(y_true, y_prob, target_recall):
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


def evaluate_with_threshold(generator, split_name, threshold):
    generator.reset()
    y_true = generator.classes.astype(np.int32)
    y_prob = model.predict(generator, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-8)  # recall for positive class (opacity)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)

    print(f"\n[{split_name}] threshold={threshold:.4f}")
    print(f"[{split_name}] TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"[{split_name}] sensitivity/recall={sensitivity:.4f}")
    print(f"[{split_name}] specificity={specificity:.4f}")
    print(f"[{split_name}] precision={precision:.4f}")

    return y_true, y_prob


# Load ResNet50 as feature extractor
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
)
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)
compile_model(model, learning_rate=1e-4)

# Callbacks
checkpoint_recall = ModelCheckpoint(
    "models/best_model.keras",
    monitor="val_recall",
    save_best_only=True,
    mode="max",
    verbose=1,
)
checkpoint_accuracy = ModelCheckpoint(
    "models/best_model_accuracy.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)
early_stop = EarlyStopping(
    monitor="val_recall",
    mode="max",
    patience=6,
    restore_best_weights=True,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_recall",
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1,
)

# Class weighting
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

# Stage 1: train classifier head only
history_head = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=HEAD_EPOCHS,
    callbacks=[checkpoint_recall, checkpoint_accuracy, early_stop, reduce_lr],
    class_weight=class_weight_dict,
)

# Stage 2: fine-tune upper ResNet blocks (keep BN frozen for stability)
base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAST_N]:
    layer.trainable = False
for layer in base_model.layers[-UNFREEZE_LAST_N:]:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

compile_model(model, learning_rate=1e-5)
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[checkpoint_recall, checkpoint_accuracy, early_stop, reduce_lr],
    class_weight=class_weight_dict,
)

# Threshold calibration on validation set to favor recall
val_generator.reset()
val_true = val_generator.classes.astype(np.int32)
val_prob = model.predict(val_generator, verbose=0).ravel()
decision_threshold, achieved_recall, achieved_precision = select_threshold_for_recall(
    val_true, val_prob, TARGET_RECALL
)
print(
    f"\nCalibrated threshold={decision_threshold:.4f} "
    f"(val recall={achieved_recall:.4f}, val precision={achieved_precision:.4f})"
)

os.makedirs("models", exist_ok=True)
with open("models/decision_threshold.txt", "w", encoding="utf-8") as f:
    f.write(f"{decision_threshold:.6f}\n")

evaluate_with_threshold(val_generator, "Validation", decision_threshold)
evaluate_with_threshold(test_generator, "Test", decision_threshold)

model.save("models/final_model.keras")
