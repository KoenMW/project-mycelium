# --- IMPORTS ---
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam

# --- SETTINGS ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "mycelium_labeled"

# Only keep classes from day 0 to day 14
CLASSES = [str(i) for i in range(15)]  # Class names as day 0, 1, ..., 14
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- LOAD FILE PATHS ---
filepaths, labels = [], []
for label in CLASSES:
    class_path = os.path.join(DATA_DIR, label)
    if os.path.exists(class_path):  # Check if class folder exists
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepaths.append(os.path.join(class_path, fname))
                labels.append(label)

# --- TRAIN/VAL SPLIT ---
train_files, val_files, y_train, y_val = train_test_split(
    filepaths, labels, test_size=0.2, stratify=labels, random_state=SEED
)

# --- SETUP SPLIT DIRECTORIES ---
def setup_split_dir(split_dir, files, labels):
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    for label in CLASSES:
        os.makedirs(os.path.join(split_dir, label), exist_ok=True)
    for f, label in zip(files, labels):
        shutil.copy(f, os.path.join(split_dir, label, os.path.basename(f)))

setup_split_dir("split/train", train_files, y_train)
setup_split_dir("split/val", val_files, y_val)

# --- IMAGE GENERATORS ---
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
    "split/train", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", classes=CLASSES
)
val_generator = val_datagen.flow_from_directory(
    "split/val", target_size=IMG_SIZE, batch_size=1, class_mode="categorical", shuffle=False, classes=CLASSES
)

# --- CLASS WEIGHTS ---
y_train_labels = train_generator.classes
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights = dict(enumerate(weights))

# --- AUTOENCODER MODEL ---
def build_autoencoder(input_shape=(224, 224, 3)):
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

# --- AUTOENCODER DATA GENERATOR ---
def autoencoder_data_generator(generator):
    while True:
        x, _ = next(generator)
        yield x, x

# --- TRAIN AUTOENCODER ---
autoencoder, encoder = build_autoencoder()
autoencoder.fit(
    autoencoder_data_generator(train_generator),
    steps_per_epoch=len(train_generator),
    validation_data=autoencoder_data_generator(val_generator),
    validation_steps=len(val_generator),
    epochs=20
)

# --- HYBRID MODEL (VGG16 + ENCODER) ---
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in vgg_base.layers[:-4]:
    layer.trainable = False

vgg_features = Flatten()(vgg_base.output)
encoder_input = Input(shape=(224, 224, 3))
encoder_features = encoder(encoder_input)
encoder_flat = Flatten()(encoder_features)

combined = Concatenate()([vgg_features, encoder_flat])
x = Dense(256, activation='relu')(combined)
output = Dense(len(CLASSES), activation='softmax')(x)

hybrid_model = Model(inputs=[vgg_base.input, encoder_input], outputs=output)
hybrid_model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# --- TF.DATA WRAPPER FOR DUAL INPUTS ---
def make_dual_input_dataset(generator):
    output_signature = (
        (tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)),
        tf.TensorSpec(shape=(None, len(CLASSES)), dtype=tf.float32)
    )

    def gen():
        while True:
            x, y = next(generator)
            yield (x, x), y

    return tf.data.Dataset.from_generator(gen, output_signature=output_signature)

train_dataset = make_dual_input_dataset(train_generator).repeat()
val_dataset = make_dual_input_dataset(val_generator)

# --- TRAIN HYBRID MODEL ---
hybrid_model.fit(
    train_dataset,
    steps_per_epoch=len(train_generator),
    validation_data=val_dataset,
    validation_steps=len(val_generator),
    epochs=EPOCHS,
    class_weight=class_weights
)

# --- EVALUATION ---
y_true = val_generator.classes
y_pred_probs = hybrid_model.predict(val_dataset, steps=len(val_generator))
y_pred = np.argmax(y_pred_probs, axis=1)

print(classification_report(y_true, y_pred, target_names=CLASSES))
cm = confusion_matrix(y_true, y_pred)

# --- FUZZY ACCURACY ---
def fuzzy_accuracy(y_true, y_pred, threshold_day=9):
    correct = 0
    for true, pred in zip(y_true, y_pred):
        tolerance = 2 if true >= threshold_day else 1
        if abs(true - pred) <= tolerance:
            correct += 1
    return correct / len(y_true)

fuzzy_acc = fuzzy_accuracy(y_true, y_pred)
print(f"Fuzzy Accuracy (±1 day, ±2 from day 9): {fuzzy_acc:.4f}")

# --- PLOT CONFUSION MATRIX ---
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(cmap=plt.cm.Blues)
plt.title("Validation Confusion Matrix")
plt.show()
