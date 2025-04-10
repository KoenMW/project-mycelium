import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, VGG19 , VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight

# --- SETTINGS ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "mycelium_labeled"  # your labeled data folder
CLASSES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23' ,'24','25', '26'] # subfolder names
SEED = 42
MODEL = "vgg16"

# --- Set seeds for reproducibility ---
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Step 1: Load file paths and labels ---
filepaths = []
labels = []

def load_resnet50(freeze: bool = False, pooling: str = "avg"):
    base_model = ResNet50(weights='imagenet', include_top=False, pooling=pooling, input_shape=(224, 224, 3))
    if freeze:
        for layer in base_model.layers:
            layer.trainable = False
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

def load_vgg16(freeze: bool = False):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    if freeze:
        for layer in base_model.layers:
            layer.trainable = False
    model = Model(inputs=base_model.input, outputs=output)
    return model

def load_vgg19(freeze: bool = False):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    if freeze:
        for layer in base_model.layers:
            layer.trainable = False
    model = Model(inputs=base_model.input, outputs=output)
    return model

for label in CLASSES:
    class_path = os.path.join(DATA_DIR, label)
    for fname in os.listdir(class_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepaths.append(os.path.join(class_path, fname))
            labels.append(label)

def load_pretrained_model(model_name: str, **kwargs):
    model_loaders = {
        "resnet50": load_resnet50,
        "vgg16": load_vgg16,
        "vgg19": load_vgg19,
    }
    
    model_name = model_name.lower()
    if model_name not in model_loaders:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(model_loaders.keys())}")
    
    return model_loaders[model_name](**kwargs)


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

val_generator = val_datagen.flow_from_directory(
    "split/val",
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)

# --- Step 5: Compute class weights ---
y_train_labels = train_generator.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# --- Step 6: Load Model base ---
base_model = load_pretrained_model(MODEL)
for layer in base_model.layers[:-4]:  # Fine-tune last 4 layers
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(len(CLASSES), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# --- Step 7: Train the model ---
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights
)

# --- Step 8: Evaluate the model ---
y_true = val_generator.classes
y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

print(classification_report(y_true, y_pred, target_names=CLASSES))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(cmap=plt.cm.Blues)
plt.title("Validation Confusion Matrix")
plt.show()
