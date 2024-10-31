import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.utils import resample
import warnings
#from cdbw import CDbw
import dbcv

def remove_duplicates(X, y):
    unique_rows, indices = np.unique(X, axis=0, return_index=True)
    X_unique = X[indices]
    y_unique = y[indices]
    removed_count = X.shape[0] - X_unique.shape[0]
    print(f"Removed {removed_count} duplicate rows.")
    return X_unique, y_unique

def stratified_sample(X, labels, sample_size):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    samples_per_cluster = max(sample_size // n_clusters, 1)

    sampled_X = []
    sampled_labels = []

    for label in unique_labels:
        cluster_X = X[labels == label]
        cluster_size = len(cluster_X)

        if cluster_size <= samples_per_cluster:
            sampled_X.append(cluster_X)
            sampled_labels.extend([label] * cluster_size)
        else:
            cluster_sample = resample(cluster_X, n_samples=samples_per_cluster, replace=False)
            sampled_X.append(cluster_sample)
            sampled_labels.extend([label] * samples_per_cluster)

    return np.vstack(sampled_X), np.array(sampled_labels)

def calculate_dunn_index(X, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1:
        return 0

    min_inter_cluster_distance = np.inf
    max_intra_cluster_distance = 0

    for i, label in enumerate(unique_labels):
        cluster_points = X[labels == label]

        if len(cluster_points) > 1:
            distances = pdist(cluster_points)
            max_intra_cluster_distance = max(max_intra_cluster_distance, np.max(distances))

        for other_label in unique_labels[i+1:]:
            other_cluster_points = X[labels == other_label]
            inter_distances = cdist(cluster_points, other_cluster_points)
            min_inter_cluster_distance = min(min_inter_cluster_distance, np.min(inter_distances))

    return min_inter_cluster_distance / max_intra_cluster_distance if max_intra_cluster_distance > 0 else 0

def improved_dunn_index(X, labels, sample_size=1000, n_runs=1):
    dunn_indices = []

    for _ in range(n_runs):
        sampled_X, sampled_labels = stratified_sample(X, labels, sample_size)
        dunn_index = calculate_dunn_index(sampled_X, sampled_labels)
        dunn_indices.append(dunn_index)

    return np.mean(dunn_indices)

def evaluate_clustering(X, labels):
    X = np.array(X)
    labels = np.array(labels)

    # Remove duplicates
    X, labels = remove_duplicates(X, labels)

    noise_labels = [-1, -2]
    label_counts = Counter(labels)
    valid_labels = [label for label, count in label_counts.items() if count > 1 and label not in noise_labels]
    n_clusters = len(valid_labels)

    # Initialize results dictionary with NA values
    results = {
        "n_clusters": n_clusters,
        "dbcv_score": np.nan,
        "cdbw_score": np.nan,
        "silhouette_score": np.nan,
        "calinski_harabasz_score": np.nan,
        "davies_bouldin_score": np.nan,
        "dunn_index": np.nan,
        "cluster_sizes": dict(label_counts),
        "valid_cluster_sizes": {},
        "noise_points": sum(label_counts[l] for l in noise_labels),
    }

    if n_clusters <= 1:
        print("Warning: No valid clusters found (clusters with more than one point, excluding noise).")
        return results

    mask = np.isin(labels, valid_labels)
    X_filtered = X[mask]
    labels_filtered = labels[mask]

    if len(X_filtered) == 0:
        print("Warning: No valid data points left after filtering.")
        results["n_clusters"] = 0
        return results

    label_counts_filtered = Counter(labels_filtered)
    results["valid_cluster_sizes"] = dict(label_counts_filtered)

    # Calculate DBCV score
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            results["dbcv_score"] = dbcv.dbcv(X_filtered, labels_filtered, n_processes=25, batch_size=100000, noise_id=-1)
    except Exception as e:
        print(f"Warning: DBCV calculation failed. Error: {str(e)}")

    return results

# This function can be called from the main script
def run_evaluation(X, y):
    """
    Run the clustering evaluation on the given data.

    Parameters:
    X : numpy array
        The feature matrix
    y : numpy array
        The cluster labels

    Returns:
    dict
        A dictionary containing all the evaluation metrics
    """
    return evaluate_clustering(X, y)
