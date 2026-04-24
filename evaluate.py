import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_DIR = os.path.join(BASE_DIR, "archive", "test")
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "models", "class_names.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 32

# Load class names
with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=False
)

# Normalize
test_dataset = test_dataset.map(lambda x, y: (x / 255.0, y))

# Collect true labels
y_true = np.concatenate([y.numpy() for _, y in test_dataset], axis=0)

# Predict
y_pred_probs = model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.show()