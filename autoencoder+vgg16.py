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
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "mycelium_labeled"
CLASSES = [str(i) for i in range(15)]
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load image paths and labels
filepaths, labels = [], []
for label in CLASSES:
    class_path = os.path.join(DATA_DIR, label)
    if os.path.exists(class_path):
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepaths.append(os.path.join(class_path, fname))
                labels.append(label)

# Train-validation split
train_files, val_files, y_train, y_val = train_test_split(
    filepaths, labels, test_size=0.2, stratify=labels, random_state=SEED
)

# Create directory structure for ImageDataGenerator
def setup_split_dir(split_dir, files, labels):
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    for label in CLASSES:
        os.makedirs(os.path.join(split_dir, label), exist_ok=True)
    for f, label in zip(files, labels):
        shutil.copy(f, os.path.join(split_dir, label, os.path.basename(f)))

setup_split_dir("split/train", train_files, y_train)
setup_split_dir("split/val", val_files, y_val)

# Data generators
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

# Compute class weights
y_train_labels = train_generator.classes
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weights = dict(enumerate(weights))

# Build autoencoder
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

def autoencoder_data_generator(generator):
    while True:
        x, _ = next(generator)
        yield x, x

# Train autoencoder
autoencoder, encoder = build_autoencoder()
autoencoder.fit(
    autoencoder_data_generator(train_generator),
    steps_per_epoch=len(train_generator),
    validation_data=autoencoder_data_generator(val_generator),
    validation_steps=len(val_generator),
    epochs=20
)

# Hybrid model (VGG16 + Encoder)
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

# Prepare tf.data.Dataset from ImageDataGenerator
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

# Train hybrid model
hybrid_model.fit(
    train_dataset,
    steps_per_epoch=len(train_generator),
    validation_data=val_dataset,
    validation_steps=len(val_generator),
    epochs=EPOCHS,
    class_weight=class_weights
)

# Save model
hybrid_model.save("hybrid_model.h5")
encoder.save("encoder_model.h5")
autoencoder.save("autoencoder_model.h5")

# Evaluate model with correct inputs
val_generator.reset()
X_val, y_val_true = [], []
for i in range(len(val_generator)):
    x, y = val_generator.next()
    X_val.append(x)
    y_val_true.append(y)
X_val = np.concatenate(X_val)
y_val_true = np.argmax(np.concatenate(y_val_true), axis=1)

# Predict using hybrid model
y_pred_probs = hybrid_model.predict([X_val, X_val], batch_size=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_val_true, y_pred, target_names=CLASSES))
cm = confusion_matrix(y_val_true, y_pred)

# Fuzzy accuracy metric
def fuzzy_accuracy(y_true, y_pred, max_day=14):
    return np.mean([max(0, true - 1) <= pred <= min(max_day, true + 1) for true, pred in zip(y_true, y_pred)])

print(f"Exact Accuracy: {np.mean(y_val_true == y_pred):.4f}")
print(f"Fuzzy Accuracy (±1 day): {fuzzy_accuracy(y_val_true, y_pred):.4f}")

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
ax.set_title("Validation Confusion Matrix")
ax.set_xlabel("Predicted Day")
ax.set_ylabel("True Day")
plt.tight_layout()
plt.show()