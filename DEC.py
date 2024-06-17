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
#from mpl_toolkits.mplot3d import Axes3D
#import plotly.graph_objects as go
from sklearn.manifold import TSNE as sklearn_TSNE
#from openTSNE import TSNE as openTSNE
#from MulticoreTSNE import MulticoreTSNE as MulticoreTSNE
#from DenPeakcode import DenPeakCluster
from denmune import DenMune
import pandas as pd
K.set_image_data_format('channels_last')  # This remains unchanged

def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric, adapted for genomic data.
    """
    n_stacks = len(dims) - 1
    x = Input(shape=(dims[0],), name='input')
    h = x

    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)

    y = h
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')

class DEC(object):
    def __init__(self, dims, init='glorot_uniform'):
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.pretrained = False
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)

    def pretrain(self, x, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp', verbose=1):
        """
        Pretraining method adapted for genomic data.
        """
        print('Pretraining......')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]

        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb, verbose=verbose)
        print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True

    def fit(self, x, save_dir='./results/temp'):
        """
        Clustering method adapted for genomic data.
        """
        t1 = time()
        print('******************** Use Denmune to Cluster ************************')

        features = self.encoder.predict(x)
        np.savetxt(os.path.join(save_dir, 'full_features.txt'), features)

        # Define the t-SNE configurations
        n_components = 2
        perplexity = 30
        random_state = 42
        n_jobs=40

        # Define t-SNE approaches to benchmark
        tsne_approaches = {
            #'MulticoreTSNE': {'class': MulticoreTSNE, 'features': None, 'time': None},
            'scikit-learn': {'class': sklearn_TSNE, 'features': None, 'time': None},
#            'openTSNE': {'class': openTSNE, 'features': None, 'time': None},
        }

        # Benchmark each t-SNE approach
        for name, tsne_info in tsne_approaches.items():
            start_time = time()
            if name == 'openTSNE':
                tsne_instance = tsne_info['class'](n_components=n_components, perplexity=perplexity, random_state=random_state, n_jobs=n_jobs)
                features_transformed = tsne_instance.fit(features)
            else:
                tsne_instance = tsne_info['class'](n_components=n_components, perplexity=perplexity, random_state=random_state, n_jobs=n_jobs)
                features_transformed = tsne_instance.fit_transform(features)

            tsne_time = time() - start_time
            tsne_info['features'] = features_transformed
            tsne_info['time'] = tsne_time
            np.savetxt(os.path.join(save_dir, f'features_{name}.txt'), features_transformed)
            print(f"{name} t-SNE took {tsne_time:.2f} seconds.")

        # Process each t-SNE approach's results
        for name, tsne_info in tsne_approaches.items():
            features_transformed = tsne_info['features']
            #y_pred, center_num, dc_percent, dc = DenPeakCluster(features_transformed)
            print("Start clustering with DenMune")
            knn = 20 # k-nearest neighbor, the only parameter required by the algorithm
            X_train = pd.DataFrame(features_transformed)
#            print(X_train.shape)
#            print(features.shape)

            dm = DenMune(train_data=X_train, k_nearest=knn)
            y_pred, validity = dm.fit_predict(show_analyzer=False, show_noise=True)
            y_pred = y_pred['train']
            # Plotting
            cmap = 'viridis'

            plt.cla()
            plt.scatter(features_transformed[:, 0], features_transformed[:, 1], c=y_pred, cmap=cmap, s=0.5, alpha=0.5)
            plt.savefig(os.path.join(save_dir, f'2D_{name}.png'), dpi=600)

            # Code for 3D plotting
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')  # Creating a 3D subplot

            # Scatter plot in 3D, assuming features_transformed has at least three columns for x, y, z coordinates
            #ax.scatter(features_transformed[:, 0], features_transformed[:, 1], features_transformed[:, 2], c=y_pred, s=0.5)  # Adjust marker size as needed

            # Saving the 3D plot. Adjust file name as needed.
            #plt.savefig(os.path.join(save_dir, f'3D_{name}.png'), dpi=600)

            # Clear the current figure's plotting area to prevent overlap with future plots (optional)
            #plt.clf()

            #fig = go.Figure(data=[go.Scatter3d(
            #            x=features_transformed[:, 0],  # X coordinates
            #            y=features_transformed[:, 1],  # Y coordinates
            #            z=features_transformed[:, 2],  # Z coordinates
            #            mode='markers',
            #            marker=dict(
            #                        size=5,  # Marker size
            #                        color=y_pred,  # Assigning colors based on y_pred
            #                        colorscale='Viridis',  # Color scale for markers
            #                        opacity=0.8  # Marker opacity
            #            )
           # )])

            # Customize the layout
            #fig.update_layout(
            #            title='3D Scatter Plot',
            #            scene=dict(
            #                        xaxis_title='X Axis',
            #                        yaxis_title='Y Axis',
            #                        zaxis_title='Z Axis'
            #            ),
            #            autosize=False,
            #            width=800,
            #            height=600,
            #)

            # The file path for saving
            #file_path = os.path.join(save_dir, f'3D_{name}.html')

            # Save the plot as an interactive HTML file
            #fig.write_html(file_path)

            # Saving data
            #np.savetxt(os.path.join(save_dir, f'features_{name}.txt'), features_transformed)
            np.savetxt(os.path.join(save_dir, f'predicted_clusters_{name}.txt'), y_pred)

            # Log clustering info for each approach
            #log_info = {
            #    'TSNE Approach': name,
            #    'Total Clusters': center_num,
            #    'DC Percent': dc_percent,
            #    'DC Value': dc,
            #    'n jobs': n_jobs,
            #    'Clustering Time': tsne_info['time'],
            #    'Total Time': time() - t1
            #}

            #with open(os.path.join(save_dir, f'log_{name}.csv'), 'a', newline='') as logfile:
            #        logwriter = csv.DictWriter(logfile, fieldnames=log_info.keys())
            #        if logfile.tell() == 0:  # Check if file is empty to write header
            #                logwriter.writeheader()
            #        logwriter.writerow(log_info)

        print('Clustering completed and results saved.')

        return y_pred
