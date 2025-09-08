import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

def plot_city_map(place_name, latitude, longitude, box_size_km=2, poi_tags=None):
    """
    Visualize geographic data on a map.
    
    Args:
        place_name (str): Name of the location
        latitude (float): Center latitude
        longitude (float): Center longitude
        box_size_km (float): Size of bounding box in km
        poi_tags (dict): OSM tags for points of interest
    """
    # Get data
    pois = get_osm_datapoints(latitude, longitude, box_size_km, poi_tags)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot points
    pois.plot(ax=ax, color='red', markersize=10, alpha=0.7)
    
    # Customize plot
    ax.set_title(f"OSM Points of Interest - {place_name}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.show()

def get_osm_features(latitude, longitude, box_size_km=2, tags=None):
    """
    Get raw OSM data as a GeoDataFrame.
    
    Args:
        latitude (float): Center latitude
        longitude (float): Center longitude
        box_size_km (float): Size of bounding box in km
        tags (dict): OSM tags to filter features
        
    Returns:
        geopandas.GeoDataFrame: Raw OSM features
    """
    return get_osm_datapoints(latitude, longitude, box_size_km, tags)

def get_feature_vector(latitude, longitude, box_size_km=2, features=None):
    """
    Quantify geographic features into a numerical vector.
    
    Args:
        latitude (float): Center latitude
        longitude (float): Center longitude
        box_size_km (float): Size of bounding box in km
        features (list): List of feature types to count
        
    Returns:
        numpy.ndarray: Feature vector
    """
    if features is None:
        features = ['amenity', 'building', 'shop', 'tourism', 'leisure']
    
    # Get OSM data
    pois = get_osm_datapoints(latitude, longitude, box_size_km)
    
    # Create feature vector
    feature_vector = []
    
    for feature in features:
        # Count occurrences of each feature type
        if feature in pois.columns:
            count = pois[feature].notna().sum()
        else:
            count = 0
        feature_vector.append(count)
    
    # Add total count as additional feature
    feature_vector.append(len(pois))
    
    return np.array(feature_vector)

def visualize_feature_space(X, y, method='PCA'):
    """
    Visualize data distribution and separability using dimensionality reduction.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Labels
        method (str): Dimensionality reduction method ('PCA' or 't-SNE')
    """
    if method == 'PCA':
        reducer = PCA(n_components=2)
        title = "PCA Visualization"
    elif method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=42)
        title = "t-SNE Visualization"
    else:
        raise ValueError("Method must be 'PCA' or 't-SNE'")
    
    # Reduce dimensionality
    X_reduced = reducer.fit_transform(X)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot each class with different color
    unique_labels = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.7)
    
    plt.title(title)
    plt.xlabel(f"{method} Component 1")
    plt.ylabel(f"{method} Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
