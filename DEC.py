# -*- coding: utf-8 -*-
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras import callbacks
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import metrics
from DenPeakcode import DenPeakCluster

def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric, suitable for genomic data.
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
        super(DEC, self).__init__()
        self.dims = dims
        self.autoencoder, self.encoder = autoencoder(dims, init=init)

    def pretrain(self, x, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp', verbose=1):
        print('Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')
        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger], verbose=verbose)
        print('Pretraining time: ', time())
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def fit(self, x, save_dir='./results/temp'):
        print('******************** Use Denpeak to Cluster ************************')
        features = self.extract_features(x)
        features = TSNE(n_components=2).fit_transform(features)
        y_pred, _, center_num, _, _ = DenPeakCluster(features)
        print('saving picture to:', save_dir + '/2D.png')
        plt.scatter(features[:, 0], features[:, 1], c=y_pred, s=0.5, alpha=0.5)
        plt.savefig(save_dir + '/2D.png')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['acc', 'nmi', 'ari', 'center_num'])
        logwriter.writeheader()

        # Assuming y (true labels) is available for evaluation
        # acc = np.round(metrics.acc(y, y_pred), 5)
        # nmi = np.round(metrics.nmi(y, y_pred), 5)
        # ari = np.round(metrics.ari(y, y_pred), 5)
        # logwriter.writerow({'acc': acc, 'nmi': nmi, 'ari': ari, 'center_num': center_num})

        logfile.close()
        print('Clustering completed and results saved.')
