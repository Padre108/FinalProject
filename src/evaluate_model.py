# Model Evaluation Script
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Import test data generator
from data_preprocessing import test_generator

# Load the best model
model = load_model('models/best_model.h5')

# Predict on test data
test_generator.reset()
pred_probs = model.predict(test_generator, verbose=1)
preds = (pred_probs > 0.5).astype(int).flatten()

# True labels
true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Classification report
print("\nClassification Report:")
print(classification_report(true_labels, preds, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(true_labels, preds)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(5,5))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0,1], class_labels)
plt.yticks([0,1], class_labels)
for i in range(2):
	for j in range(2):
		plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.tight_layout()
plt.show()
