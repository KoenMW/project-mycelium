import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

def extract_features(img_path: str, model) -> np.ndarray:
    img = image.load_img(img_path, target_size=(224, 224))  # ResNet50 expects 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize input
    
    features = model.predict(img_array)  # Extract features
    return features.flatten()  # Flatten to 1D vector

# Example usage
features = extract_features("example.jpg", feature_extractor)
