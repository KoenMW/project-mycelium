import os
import re
import numpy as np
import cv2
from keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input # type: ignore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from yellowbrick.cluster import KElbowVisualizer

# Set base folder
BASE_FOLDER: str = "mycelium"

# Default parameters
DEFAULT_ANGLES: List[int] = [1]  # Selectable: [1, 2, 3, 4]
DEFAULT_TESTS: List[int] = list(range(1, 7))  # Tests 1-6
DEFAULT_TIME_WINDOW: int = 5  # Hours per feature map

# CNN model for feature extraction
model: ResNet50 = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Regex pattern to match filenames
FILENAME_PATTERN = re.compile(r"test(\d+)_h(\d+)_(\d)\.jpg")


def extract_features(image_path: str) -> Optional[np.ndarray]:
    """Extract features from an image using ResNet50."""
    img: Optional[np.ndarray] = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features: np.ndarray = model.predict(img)
    
    return features.flatten()


def parse_filename(filename: str) -> Optional[Tuple[int, int, int]]:
    """
    Extract test number, hours, and angle from the filename.

    Args:
        filename (str): Image filename.

    Returns:
        Optional[Tuple[int, int, int]]: (test_number, hours, angle) or None if invalid.
    """
    match = FILENAME_PATTERN.match(filename)
    if match:
        test_number, hours, angle = map(int, match.groups())
        return test_number, hours, angle
    return None


def group_images_by_time_window(images: List[Tuple[str, int]], time_window: int) -> Dict[int, List[str]]:
    """
    Groups images into bins based on the selected time window.

    Args:
        images (List[Tuple[str, int]]): List of (image_path, hours).
        time_window (int): Number of hours per feature map.

    Returns:
        Dict[int, List[str]]: Grouped images, key = time window index.
    """
    grouped_images: Dict[int, List[str]] = {}
    
    for image_path, hours in images:
        time_bin = hours // time_window  # Assign to a time window
        if time_bin not in grouped_images:
            grouped_images[time_bin] = []
        grouped_images[time_bin].append(image_path)

    return grouped_images


def process_images(angles: List[int] = DEFAULT_ANGLES, tests: List[int] = DEFAULT_TESTS, time_window: int = DEFAULT_TIME_WINDOW):
    """Processes images, extracts features, clusters them, and visualizes results."""
    image_data: List[Tuple[str, int]] = []  # Store (image_path, hours)

    # Scan files in the output folder
    for filename in os.listdir(BASE_FOLDER):
        file_path: str = os.path.join(BASE_FOLDER, filename)

        if not os.path.isfile(file_path):
            continue  # Skip non-files

        parsed_data = parse_filename(filename)
        if not parsed_data:
            continue  # Skip invalid filenames

        test_number, hours, angle = parsed_data

        # Filter by selected tests and angles
        if test_number in tests and angle in angles:
            image_data.append((file_path, hours))

    # Group images by time window
    grouped_images: Dict[int, List[str]] = group_images_by_time_window(image_data, time_window)

    # Extract and aggregate features per time window
    time_features: Dict[int, np.ndarray] = {}

    for time_bin, image_paths in grouped_images.items():
        feature_list: List[np.ndarray] = []

        for image_path in image_paths:
            features = extract_features(image_path)
            if features is not None:
                feature_list.append(features)

        if feature_list:
            time_features[time_bin] = np.mean(feature_list, axis=0)  # Aggregate features

    if not time_features:
        print("No valid images found for processing.")
        return

    # Prepare data for clustering
    time_bins: List[int] = list(time_features.keys())
    feature_matrix: np.ndarray = np.array(list(time_features.values()))

    # Reduce dimensions using PCA
    pca: PCA = PCA(n_components=2)
    features_pca: np.ndarray = pca.fit_transform(feature_matrix)

    # Clustering with K-Means
    num_clusters: int = 2  # Change if needed
    kmeans: KMeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters: np.ndarray = kmeans.fit_predict(feature_matrix)

    # Assign clusters to time bins
    time_clusters: Dict[int, int] = dict(zip(time_bins, clusters))
    print("Cluster Assignments:", time_clusters)

    elbowPlot(features_pca, kmeans, (2,10))

    # Visualization
    visualize2D(features_pca, clusters, time_bins)
    


def visualize2D(feature_matrix: np.ndarray, clusters: np.ndarray, time_bins: list[int]): 
    plt.figure(figsize=(8, 6))
    for cluster in set(clusters):
        indices: np.ndarray = np.where(clusters == cluster)
        plt.scatter(feature_matrix[indices, 0], feature_matrix[indices, 1], label=f"Cluster {cluster}")

    for i, time_bin in enumerate(time_bins):
        plt.annotate(f"T{time_bin}", (feature_matrix[i, 0], feature_matrix[i, 1]))

    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    plt.legend()
    plt.title("Clusters of Image Feature Maps Over Time")
    plt.savefig(f"plots/cluster-pce-h{DEFAULT_TIME_WINDOW}.png")
    plt.show()


def elbowPlot(data, model, kRange):

    # set the plotting stage for later
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, constrained_layout=True,figsize=(15,5))

    # Using the Distortion measure:
    visualizer = KElbowVisualizer(model, k=kRange, metric='distortion', ax=ax1, n_init=10)
    chPlot = visualizer.fit(data)
    ax1.set_title('Distortion')

    # Using the Calinski-Harabasz measure
    visualizer = KElbowVisualizer(model, k=kRange, metric='calinski_harabasz', ax=ax2, n_init=10)
    chPlot = visualizer.fit(data)
    ax2.set_title('Calinski-Harabasz')

    # Using the Silhouette measure
    visualizer = KElbowVisualizer(model, k=kRange, metric='silhouette', ax=ax3, n_init=10)
    chPlot = visualizer.fit(data)
    ax3.set_title('Silhouette')

    # Show the results
    plt.savefig(f"plots/elbowplot-pca-h{DEFAULT_TIME_WINDOW}.png")
    plt.show()

# Run with default parameters
if __name__ == "__main__":
    process_images()
