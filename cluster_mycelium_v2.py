import os
import re
import numpy as np
import cv2
from keras.applications import ResNet50, VGG19 , VGG16, MobileNetV2
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
import pickle
import hdbscan
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from yellowbrick.cluster import KElbowVisualizer
import scipy.cluster.hierarchy as sch
import shutil

# Set base folder
BASE_FOLDER: str = "mycelium"

# Default parameters
DEFAULT_ANGLES: List[int] = [1]
DEFAULT_TESTS: List[int] = list(range(1, 7))
DEFAULT_TIME_WINDOW: int = 2
DEFAULT_MODEL: str = "vgg16"
DEFAULT_CLUSTER: str = "kmeans"
DEFAULT_N_CLUSTERS: int = 2
DEFAULT_LABELS: bool = True
DEFAULT_DIMENSIONS: int = 2

# Regex pattern to match filenames
FILENAME_PATTERN = re.compile(r"test(\d+)_h(\d+)_(\d)\.jpg")

FEATURES_FILENAME = f"results/{DEFAULT_MODEL}_h{DEFAULT_TIME_WINDOW}_t{'_'.join(map(str, DEFAULT_TESTS))}_a{'_'.join(map(str, DEFAULT_ANGLES))}.pkl"


if (not os.path.exists(BASE_FOLDER)):
    raise SystemError(f"No {BASE_FOLDER} found")
    
if (not os.path.exists("plots")):
    os.makedirs("plots")

def save_time_features(time_features: np.ndarray, filename: str = FEATURES_FILENAME):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(time_features, f)

def lead_time_features(filename: str = FEATURES_FILENAME):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            features: Dict[int, np.ndarray] = pickle.load(f)
            return features
    return None

def load_resnet50(freeze: bool = False, pooling: str = "avg"):
    base_model = ResNet50(weights='imagenet', include_top=False, pooling=pooling)
    if freeze:
        for layer in base_model.layers:
            layer.trainable = False
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model, resnet_preprocess

def load_vgg16(freeze: bool = False):
    base_model = VGG16(weights='imagenet', include_top=False)
    if freeze:
        for layer in base_model.layers:
            layer.trainable = False
    output = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=output)
    return model, vgg16_preprocess

def load_vgg19(freeze: bool = False):
    base_model = VGG19(weights='imagenet', include_top=False)
    if freeze:
        for layer in base_model.layers:
            layer.trainable = False
    output = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=output)
    return model, vgg19_preprocess


def load_mobilenetv2(freeze: bool = False, alpha: float = 1.0):
    base_model = MobileNetV2(weights='imagenet', include_top=False, alpha=alpha)
    if freeze:
        for layer in base_model.layers:
            layer.trainable = False
    output = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=output)
    return model, mobilenet_preprocess

def load_pretrained_model(model_name: str, **kwargs):
    model_loaders = {
        "resnet50": load_resnet50,
        "vgg16": load_vgg16,
        "vgg19": load_vgg19,
        "mobilenetv2": load_mobilenetv2
    }
    
    model_name = model_name.lower()
    if model_name not in model_loaders:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(model_loaders.keys())}")
    
    return model_loaders[model_name](**kwargs)


def get_cluster(cluster_name: str, data: np.ndarray):
    """
    Select and apply a clustering algorithm.
    """
    cluster_methods = {
        "kmeans": KMeans(n_clusters=DEFAULT_N_CLUSTERS, random_state=42),
        "dbscan": DBSCAN(eps=0.5, min_samples=5),
        "hierarchical": AgglomerativeClustering(n_clusters=DEFAULT_N_CLUSTERS),
        "hdbscan": hdbscan.HDBSCAN(min_cluster_size=5),
        "meanshift": MeanShift()
    }
    
    if cluster_name.lower() not in cluster_methods:
        raise ValueError(f"Clustering method {cluster_name} not supported. Choose from: {list(cluster_methods.keys())}")
    
    model = cluster_methods[cluster_name.lower()]
    clusters = model.fit_predict(data)
    
    if cluster_name.lower() == "kmeans" or cluster_name.lower() == "hierarchical":
        elbowPlot(data, model, (2, 10))
    
    return clusters


def elbowPlot(data, model, kRange):
    """
    Generate elbow plot for cluster analysis.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5))
    
    # Using the Distortion measure:
    visualizer = KElbowVisualizer(model, k=kRange, metric='distortion', ax=ax1, n_init=10)
    visualizer.fit(data)
    ax1.set_title('Distortion')
    
    # Using the Calinski-Harabasz measure
    visualizer = KElbowVisualizer(model, k=kRange, metric='calinski_harabasz', ax=ax2, n_init=10)
    visualizer.fit(data)
    ax2.set_title('Calinski-Harabasz')
    
    # Using the Silhouette measure
    visualizer = KElbowVisualizer(model, k=kRange, metric='silhouette', ax=ax3, n_init=10)
    visualizer.fit(data)
    ax3.set_title('Silhouette')
    
    plt.savefig(f"plots/elbowplot-pca_{DEFAULT_MODEL}_{DEFAULT_CLUSTER}_h{DEFAULT_TIME_WINDOW}_v2.png")
    plt.show()


def extract_features(image_path: str, model: Model, preprocess_function) -> Optional[np.ndarray]:
    """Extract features from an image using a pre-trained CNN."""
    img: Optional[np.ndarray] = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_function(img)
    features: np.ndarray = model.predict(img)
    
    return features.flatten()


def parse_filename(filename: str) -> Optional[Tuple[int, int, int]]:
    """
    Extract test number, hours, and angle from the filename.
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

def label_and_copy_images(clusters: np.ndarray, time_bins: List[int], image_data: List[Tuple[str, int]], time_window: int):
    """
    Copies images into a labeled folder with the cluster ID prefixed to the filename.
    """
    labeled_folder = "mycelium_labeled"
    os.makedirs(labeled_folder, exist_ok=True)

    # Map time_bin to cluster
    time_bin_to_cluster = dict(zip(time_bins, clusters))

    for image_path, hours in image_data:
        time_bin = hours // time_window
        if time_bin in time_bin_to_cluster:
            cluster_label = time_bin_to_cluster[time_bin] + 1
            filename = os.path.basename(image_path)
            new_filename = f"{cluster_label}_{filename}"
            dest_path = os.path.join(labeled_folder, new_filename)
            shutil.copy(image_path, dest_path)

def process_images(model_name: str = DEFAULT_MODEL, angles: List[int] = DEFAULT_ANGLES, tests: List[int] = DEFAULT_TESTS, time_window: int = DEFAULT_TIME_WINDOW):
    """Processes images, extracts features, clusters them, and visualizes results."""

    time_features = lead_time_features()
    time_bins: List[int] = []

    if time_features is None:
        model, preprocess_function = load_pretrained_model(model_name)
        image_data: List[Tuple[str, int]] = []

        for filename in os.listdir(BASE_FOLDER):
            file_path: str = os.path.join(BASE_FOLDER, filename)
            if not os.path.isfile(file_path):
                continue

            parsed_data = parse_filename(filename)
            if not parsed_data:
                continue

            test_number, hours, angle = parsed_data
            if test_number in tests and angle in angles:
                image_data.append((file_path, hours))

        if not image_data:
            print("No valid images found for processing.")
            return
        # Group images by time window
        grouped_images: Dict[int, List[str]] = group_images_by_time_window(image_data, time_window)

        # Extract and aggregate features per time window
        time_features: Dict[int, np.ndarray] = {}

        for time_bin, image_paths in grouped_images.items():
            feature_list: List[np.ndarray] = []

            for image_path in image_paths:
                features = extract_features(image_path, model, preprocess_function)
                if features is not None:
                    feature_list.append(features)

            if feature_list:
                time_features[time_bin] = np.mean(feature_list, axis=0)  # Aggregate features

        if not time_features:
            print("No valid images found for processing.")
            return
    
    
    save_time_features(time_features)

    # Prepare data for clustering
    time_bins = list(time_features.keys())

    feature_matrix = np.array(list(time_features.values()))

    pca = PCA(n_components=DEFAULT_DIMENSIONS)
    features_pca = pca.fit_transform(feature_matrix)
    plotData(DEFAULT_DIMENSIONS, features_pca, time_bins=time_bins)

    # Only label and copy if images were processed fresh
    clusters = get_cluster(DEFAULT_CLUSTER, feature_matrix)
    if image_data:
        label_and_copy_images(clusters, time_bins, image_data, time_window)
    

def plotData(dimensions: int, feature_matrix: np.ndarray, time_bins: list[int] = []):
    plot_loaders = {
        1: visualize1D,
        2: visualize2D
    }

    if dimensions not in plot_loaders:
        raise ValueError(f"{dimensions}d plot not supported. Choose from: {list(plot_loaders.keys())}")
    
    return plot_loaders[dimensions](feature_matrix=feature_matrix, time_bins=time_bins)


def visualize1D(feature_matrix: np.ndarray, method: str = 'ward', time_bins: list[int] = []):
    """
    Plots a 1D dendrogram for hierarchical clustering using PCA-reduced data.
    
    Parameters:
    - feature_matrix (np.ndarray): A 2D array of shape (n_samples, 1).
    - method (str): The linkage method to use (e.g., 'ward', 'single', 'complete', 'average').
    - time_bins (list[int]): A list of labels for the data points.
    """
    if not isinstance(feature_matrix, np.ndarray) or feature_matrix.ndim != 2 or feature_matrix.shape[1] != 1:
        raise ValueError("Input data must be a 2D numpy array with shape (n_samples, 1) from PCA.")
    
    if len(time_bins) != feature_matrix.shape[0]:
        raise ValueError("time_bins must have the same length as the number of samples in feature_matrix.")
    
    # Perform hierarchical clustering
    linkage_matrix = sch.linkage(feature_matrix, method=method)
    
    # Plot the dendrogram with time_bins as labels
    plt.figure(figsize=(10, 4))
    sch.dendrogram(linkage_matrix, orientation='top', distance_sort='ascending', 
                   show_leaf_counts=True, labels=time_bins)
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.title("1D Dendrogram of PCA-Reduced Data")
    plt.show()

def visualize2D(feature_matrix: np.ndarray, time_bins: list[int]): 
    clusters = get_cluster(DEFAULT_CLUSTER, feature_matrix)

    plt.figure(figsize=(8, 6))
    for cluster in set(clusters):
        indices = np.where(clusters == cluster)
        plt.scatter(feature_matrix[indices, 0], feature_matrix[indices, 1], label=f"Cluster {cluster}")

    if DEFAULT_LABELS:
        for i, time_bin in enumerate(time_bins):
            plt.annotate(f"H{time_bin}", (feature_matrix[i, 0], feature_matrix[i, 1]))


    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    plt.legend()
    plt.title("Clusters of Image Feature Maps Over Time")
    plt.savefig(f"plots/cluster_plot_{DEFAULT_MODEL}__{DEFAULT_CLUSTER}-h{DEFAULT_TIME_WINDOW}_dnc{DEFAULT_N_CLUSTERS}_{'labels_' if DEFAULT_LABELS else ''}v2.png")
    plt.show()

# Run with default parameters
if __name__ == "__main__":
    process_images()