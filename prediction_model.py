import os
import shutil
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Define constants
DATA_DIR: str = 'mycelium_labeled'
MODEL_DIR: str = 'model'
TEMP_DIR: str = 'temp_data'
PLOT_DIR: str = 'plots'
IMG_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32
NUM_CLASSES: int = 2

NOTREADY: str = "2"
READY: str = "1"

CLASSES = [READY, NOTREADY]

# Prepare directory structure
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

os.makedirs(os.path.join(TEMP_DIR, READY), exist_ok=True)
os.makedirs(os.path.join(TEMP_DIR, NOTREADY), exist_ok=True)

for fname in os.listdir(DATA_DIR):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        class_name = fname.split('_')[0]
        if class_name in CLASSES:
            src_path = os.path.join(DATA_DIR, fname)
            dst_path = os.path.join(TEMP_DIR, class_name, fname)
            shutil.copyfile(src_path, dst_path)

# Load data with training and test split
datagen = ImageDataGenerator(validation_split=0.2, rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    TEMP_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

test_generator = datagen.flow_from_directory(
    TEMP_DIR,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Load and modify VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-1]:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=test_generator, epochs=10)

# Save the model
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, 'vgg16_mycelium_model.h5'))

# Predictions and evaluation
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])


# Extract TP, TN, FP, FN
# Ensure 2x2 shape
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
else:
    tn = fp = fn = tp = 0
    if y_true[0] == 0:
        tn = cm[0][0]
    else:
        tp = cm[0][0]

accuracy = (tp + tn) / np.sum(cm)

# Plot and save confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[NOTREADY, READY])
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}\nTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')

os.makedirs(PLOT_DIR, exist_ok=True)
plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrix.png'))
plt.show()
plt.close()

# Clean up temporary data
shutil.rmtree(TEMP_DIR)
