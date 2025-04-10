import numpy as np
import os
import shutil
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- SETTINGS ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "mycelium_labeled"  # your labeled data folder
CLASSES = ['1', '2', '3', '4', '5', '6', '7', '8', '9','10', '11', '12', '13', '14', '15', '16', '17', '18', '19']  # subfolder names
SEED = 42

# --- Step 1: Load file paths and labels ---
filepaths = []
labels = []

for label in CLASSES:
    class_path = os.path.join(DATA_DIR, label)
    for fname in os.listdir(class_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepaths.append(os.path.join(class_path, fname))
            labels.append(label)

# --- Step 2: Create a stratified train/val split ---
train_files, val_files, y_train, y_val = train_test_split(
    filepaths, labels, test_size=0.2, stratify=labels, random_state=SEED
)

# --- Step 3: Move to temp folders ---
def setup_split_dir(split_dir, files, labels):
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    for label in CLASSES:
        os.makedirs(os.path.join(split_dir, label), exist_ok=True)
    for f, label in zip(files, labels):
        dst = os.path.join(split_dir, label, os.path.basename(f))
        shutil.copy(f, dst)

setup_split_dir("split/train", train_files, y_train)
setup_split_dir("split/val", val_files, y_val)

# --- Step 4: Image generators ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "split/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Get the class labels from your training generator
y_train = train_generator.classes

# Count samples per class
class_counts = Counter(y_train)
majority_class_count = max(class_counts.values())
total_samples = sum(class_counts.values())

# Calculate baseline accuracy
baseline_accuracy = majority_class_count / total_samples

print("Class distribution:", class_counts)
print(f"Baseline accuracy (majority class): {baseline_accuracy:.2f}")
