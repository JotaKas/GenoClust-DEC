# -*- coding: utf-8 -*-
from time import time
import os, csv
import numpy as np
import keras.backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras import callbacks
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE as sklearn_TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from denmune import DenMune
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from keras.models import load_model
from joblib import Parallel, delayed
#import dbcv WHILE I DONT SOLVE THE NUMPY PROBLEM
from cdbw import CDbw
from collections import Counter
import warnings

K.set_image_data_format('channels_last')  # This remains unchanged

class DEC(object):
    def __init__(self, dims, init='glorot_uniform'):
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.pretrained = False
        self.autoencoder = None
        self.encoder = None

    def initialize_model(self, save_dir):
        """Check if pretrained models exist, load them if available, otherwise create new models."""
        autoencoder_path = os.path.join(save_dir, 'autoencoder_model.h5')
        encoder_path = os.path.join(save_dir, 'encoder_model.h5')

        if os.path.exists(autoencoder_path) and os.path.exists(encoder_path):
            print("Loading existing models...")
            self.autoencoder = load_model(autoencoder_path)
            self.encoder = load_model(encoder_path)
            self.pretrained = True
            print("Models loaded successfully.")
        else:
            print("No existing models found. Creating new models...")
            self.autoencoder, self.encoder = autoencoder(self.dims, init=self.init)
            self.pretrained = False

    def pretrain(self, x, x_val, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp', verbose=1):
        if self.pretrained:
            print("Models already pretrained. Skipping pretraining.")
            return None

        print('Pretraining......')
        self.autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')

        csv_logger = callbacks.CSVLogger(os.path.join(save_dir, 'pretrain_log.csv'))
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        cb = [csv_logger, early_stopping]

        t0 = time()
        history = self.autoencoder.fit(x, x, validation_data=(x_val, x_val),  batch_size=batch_size, epochs=epochs, callbacks=cb, verbose=verbose)
        print('Pretraining time: ', time() - t0)

        self.autoencoder.save_weights(os.path.join(save_dir, 'ae_weights.h5'))
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)

        autoencoder_path = os.path.join(save_dir, 'autoencoder_model.h5')
        encoder_path = os.path.join(save_dir, 'encoder_model.h5')
        self.autoencoder.save(autoencoder_path)
        self.encoder.save(encoder_path)
        print(f'Full autoencoder and encoder models are saved to {save_dir}')

        self.pretrained = True
        return history

    def stratified_sample(self, X, labels, sample_size):
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Determine samples per cluster
        samples_per_cluster = max(sample_size // n_clusters, 1)

        sampled_X = []
        sampled_labels = []

        for label in unique_labels:
            cluster_X = X[labels == label]
            cluster_size = len(cluster_X)

            # If cluster is smaller than samples_per_cluster, take all points
            if cluster_size <= samples_per_cluster:
                sampled_X.append(cluster_X)
                sampled_labels.extend([label] * cluster_size)
            else:
                cluster_sample = resample(cluster_X, n_samples=samples_per_cluster, replace=False)
                sampled_X.append(cluster_sample)
                sampled_labels.extend([label] * samples_per_cluster)

        return np.vstack(sampled_X), np.array(sampled_labels)

    def calculate_dunn_index(self, X, labels):
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        if n_clusters == 1:
            return 0

        min_inter_cluster_distance = np.inf
        max_intra_cluster_distance = 0

        for i, label in enumerate(unique_labels):
            cluster_points = X[labels == label]

            if len(cluster_points) > 1:
                # Calculate max intra-cluster distance
                distances = pdist(cluster_points)
                max_intra_cluster_distance = max(max_intra_cluster_distance, np.max(distances))

            # Calculate min inter-cluster distance
            for other_label in unique_labels[i+1:]:
                other_cluster_points = X[labels == other_label]
                inter_distances = cdist(cluster_points, other_cluster_points)
                min_inter_cluster_distance = min(min_inter_cluster_distance, np.min(inter_distances))

        return min_inter_cluster_distance / max_intra_cluster_distance if max_intra_cluster_distance > 0 else 0

    def improved_dunn_index(self, X, labels, sample_size=1000, n_runs=1):
        dunn_indices = []

        for _ in range(n_runs):
            sampled_X, sampled_labels = self.stratified_sample(X, labels, sample_size)
            dunn_index = self.calculate_dunn_index(sampled_X, sampled_labels)
            dunn_indices.append(dunn_index)

        return np.mean(dunn_indices)

    def evaluate_clustering(self, X, labels):
        # Ensure X and labels are numpy arrays
        X = np.array(X)
        labels = np.array(labels)

        noise_labels = [-1, -2]  # Consider both -1 and -2 as noise
        label_counts = Counter(labels)
        valid_labels = [label for label, count in label_counts.items() if count > 1 and label not in noise_labels]
        n_clusters = len(valid_labels)

        if n_clusters <= 1:
            print("Warning: No valid clusters found (clusters with more than one point, excluding noise).")
            return {
                "n_clusters": n_clusters,
                "dbcv_score": np.nan,
                "cdbw_score": np.nan,
                "cluster_sizes": dict(label_counts),
                "valid_cluster_sizes": {},
                "noise_points": sum(label_counts[l] for l in noise_labels),
                "dbcv_error": "Not enough valid clusters",
                "cdbw_error": "Not enough valid clusters"
            }

        # Filter out single-point clusters and noise points
        mask = np.isin(labels, valid_labels)
        X_filtered = X[mask]
        labels_filtered = labels[mask]

        # Check if we have any valid data points left
        if len(X_filtered) == 0:
            print("Warning: No valid data points left after filtering.")
            return {
                "n_clusters": 0,
                "dbcv_score": np.nan,
                "cdbw_score": np.nan,
                "cluster_sizes": dict(label_counts),
                "valid_cluster_sizes": {},
                "noise_points": sum(label_counts[l] for l in noise_labels),
                "dbcv_error": "No valid data points after filtering",
                "cdbw_error": "No valid data points after filtering"
            }

        # Recalculate label_counts after filtering
        label_counts_filtered = Counter(labels_filtered)

        # Calculate DBCV score
        dbcv_error = None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                dbcv_score = dbcv.dbcv(X_filtered, labels_filtered, n_processes=1, noise_id=None)
        except Exception as e:
            print(f"Warning: DBCV calculation failed. Error: {str(e)}")
            dbcv_score = np.nan
            dbcv_error = str(e)

        # Calculate CDbw score
        cdbw_error = None
        try:
            cdbw_calculator = CDbw(X_filtered, labels_filtered, metric="euclidean", alg_noise='comb', intra_dens_inf=False, s=3, multipliers=False)
            cdbw_score = cdbw_calculator
        except Exception as e:
            print(f"Warning: CDbw calculation failed. Error: {str(e)}")
            cdbw_score = np.nan
            cdbw_error = str(e)

        return {
            "n_clusters": n_clusters,
            "dbcv_score": dbcv_score,
            "cdbw_score": cdbw_score,
            "cluster_sizes": dict(label_counts),
            "valid_cluster_sizes": dict(label_counts_filtered),
            "noise_points": sum(label_counts[l] for l in noise_labels),
            "dbcv_error": dbcv_error,
            "cdbw_error": cdbw_error
        }

    def fit(self, x, knn_values, seeds, save_dir='./results/temp'):
        print('******************** Use DenMune to Cluster ************************')

        features = self.encoder.predict(x)
        np.savetxt(os.path.join(save_dir, 'full_features.txt'), features)

        n_components = 2
        perplexity = 30

        results = {}

        for seed in seeds:
            print(f"Start t-SNE for seed = {seed}")
            tsne_instance = sklearn_TSNE(n_components=n_components, perplexity=perplexity, random_state=seed, n_jobs=-1)
            features_transformed = tsne_instance.fit_transform(features)

            tsne_file = os.path.join(save_dir, f'tsne_features_seed_{seed}.txt')
            np.savetxt(tsne_file, features_transformed)

            seed_results = {}

            for knn in knn_values:
                print(f"Clustering for k = {knn}, seed = {seed}")
                X_df = pd.DataFrame(features_transformed)
                dm = DenMune(
                    train_data=X_df,
                    k_nearest=knn,
                    file_2d=tsne_file
                )
                y_pred, _ = dm.fit_predict(show_analyzer=False, show_noise=True)

                if y_pred is None or 'train' not in y_pred or len(y_pred['train']) == 0:
                    print(f"Warning: Clustering failed for k={knn}, seed={seed}")
                    seed_results[knn] = {
                        "n_clusters": 0,
                        "dbcv_score": np.nan,
                        "cdbw_score": np.nan,
                        "cluster_sizes": {},
                        "valid_cluster_sizes": {},
                        "noise_points": 0,
                        "dbcv_error": "Clustering failed",
                        "cdbw_error": "Clustering failed"
                    }
                else:
                    y_pred = np.array(y_pred['train'])  # Ensure y_pred is a numpy array
                    np.savetxt(os.path.join(save_dir, f'predicted_clusters_knn_{knn}_seed_{seed}.txt'), y_pred)
                    #eval_metrics = self.evaluate_clustering(features, y_pred)
                    #seed_results[knn] = eval_metrics

                # Save clustering results
                #with open(os.path.join(save_dir, f'clustering_results_seed_{seed}_knn_{knn}.csv'), 'w', newline='') as csvfile:
                #    writer = csv.writer(csvfile)
                #    writer.writerow(['metric', 'value'])
                #    for key, value in eval_metrics.items():
                #        writer.writerow([key, value])

                # Plotting
                plt.figure(figsize=(10, 8))
                plt.scatter(features_transformed[:, 0], features_transformed[:, 1], c=y_pred, cmap='viridis', s=0.5, alpha=0.5)
                plt.title(f't-SNE and Clustering Results (k={knn}, seed={seed}, n_clusters={eval_metrics["n_clusters"]})')
                plt.savefig(os.path.join(save_dir, f'2D_plot_knn_{knn}_seed_{seed}.png'), dpi=600)
                plt.close()

            results[seed] = seed_results

        # Print summary of results
        print("\nClustering Evaluation Summary:")
        for seed, seed_results in results.items():
            print(f"\nFor seed={seed}:")
            for knn, metrics in seed_results.items():
                print(f"  k={knn}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value}")

        return results
