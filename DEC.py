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
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from denmune import DenMune
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from keras.models import load_model
from joblib import Parallel, delayed
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

    def fit(self, x, knn_values, seeds, save_dir='./results/temp'):
        print('******************** Use DenMune to Cluster ************************')

        #features = self.encoder.predict(x)
        #np.savetxt(os.path.join(save_dir, 'full_features.txt'), features)

       	full_features_file = os.path.join(save_dir, 'full_features.txt')
       	if os.path.exists(full_features_file):
       	    print("Loading existing full features")
       	    features = np.loadtxt(full_features_file)
       	else:
       	    print("Generating full features")
       	    features = self.encoder.predict(x)
       	    np.savetxt(full_features_file, features)

        n_components = 2
        perplexity = 30

        results = {}

#        for seed in seeds:
#            print(f"Start t-SNE for seed = {seed}")
#            tsne_instance = sklearn_TSNE(n_components=n_components, perplexity=perplexity, random_state=seed, n_jobs=-1)
#            features_transformed = tsne_instance.fit_transform(features)

#            tsne_file = os.path.join(save_dir, f'tsne_features_seed_{seed}.txt')
#            np.savetxt(tsne_file, features_transformed)

#            seed_results = {}

        for seed in seeds:
            tsne_file = os.path.join(save_dir, f'tsne_features_seed_{seed}.txt')
            if os.path.exists(tsne_file):
                print(f"Loading existing t-SNE for seed = {seed}")
                features_transformed = np.loadtxt(tsne_file)
            else:
                print(f"Start t-SNE for seed = {seed}")
                tsne_instance = sklearn_TSNE(n_components=n_components, perplexity=perplexity, random_state=seed, n_jobs=-1)
                features_transformed = tsne_instance.fit_transform(features)
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

                if y_pred is not None and 'train' in y_pred and len(y_pred['train']) > 0:
                    y_pred = np.array(y_pred['train'])
                    np.savetxt(os.path.join(save_dir, f'predicted_clusters_knn_{knn}_seed_{seed}.txt'), y_pred)

                    # Plotting
                    plt.figure(figsize=(10, 8))
                    plt.scatter(features_transformed[:, 0], features_transformed[:, 1], c=y_pred, cmap='viridis', s=0.5, alpha=0.5)
                    plt.title(f't-SNE and Clustering Results (k={knn}, seed={seed})')
                    plt.savefig(os.path.join(save_dir, f'2D_plot_knn_{knn}_seed_{seed}.png'), dpi=600)
                    plt.close()
                else:
                    print(f"Warning: Clustering failed for k={knn}, seed={seed}")

        print("Clustering completed.")
