# Model Evaluation Script
# This module loads the best trained model and evaluates it on the test set.
# Generates confusion matrix, classification report, and visualizations.
# Note: Uses 0.5 threshold; for optimized performance use threshold from decision_threshold.txt

import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Import test data generator from preprocessing module
from data_preprocessing import test_generator

# ========== Data Distribution Analysis ==========
# Print test set class balance to understand evaluation conditions.
print("Test class indices:", test_generator.class_indices)
unique, counts = np.unique(test_generator.classes, return_counts=True)
print("Samples per class (test):", dict(zip(unique, counts)))

# ========== Load Best Model ==========
# Load the best model saved during training (checkpoint based on val_recall).
model = load_model('models/best_model.keras')
print("Model loaded from models/best_model.keras")

# ========== Inference on Test Set ==========
# Generate predictions for all test samples.
# Note: Default threshold is 0.5; for medical precision, use calibrated threshold
# from models/decision_threshold.txt (optimized for 95% recall).
print("\n=== Generating predictions ===")
test_generator.reset()
pred_probs = model.predict(test_generator, verbose=1)  # Probabilities [0, 1]
print("Sample predicted probabilities:", pred_probs[:20].flatten())
print("Min prob:", np.min(pred_probs), "Max prob:", np.max(pred_probs), "Mean prob:", np.mean(pred_probs))

# Convert probabilities to binary predictions using 0.5 threshold
preds = (pred_probs > 0.5).astype(int).flatten()

# ========== Ground Truth Labels ==========
# Extract true labels and class names from generator.
true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# ========== Classification Report ==========
# Precision, Recall, F1-score for each class.
# For medical imaging:
#   - Recall: % of actual lesions detected (sensitivity)
#   - Precision: % of predicted lesions that are correct
#   - F1: Harmonic mean (balanced metric)
print("\n=== Classification Report ===")
print(classification_report(true_labels, preds, target_names=class_labels))

# ========== Confusion Matrix ==========
# TP (True Positive): Correctly predicted opacity
# FP (False Positive): Incorrectly predicted opacity (false alarm)
# TN (True Negative): Correctly predicted normal
# FN (False Negative): Missed opacity (dangerous in medical context)
cm = confusion_matrix(true_labels, preds)
print("\n=== Confusion Matrix ===")
print(cm)

# ========== Visualization ==========
# Plot confusion matrix as a heatmap.
# Ideally: high values on diagonal (TP, TN), low on off-diagonal (FP, FN).
print("\n=== Plotting Confusion Matrix ===")
plt.figure(figsize=(5, 5))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix (Test Set, threshold=0.5)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0, 1], class_labels)
plt.yticks([0, 1], class_labels)
for i in range(2):
	for j in range(2):
		plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.tight_layout()
plt.show()

print("\nEvaluation complete. Consider using calibrated threshold from models/decision_threshold.txt for production.")

