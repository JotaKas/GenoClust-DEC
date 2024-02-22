import os
import csv
import numpy as np
import pandas as pd  # Add pandas for easy CSV handling
from time import time
from keras.initializers import VarianceScaling
from ConvDEC import ConvDEC

def load_genomic_data(filepath):
    # Load data using pandas, skipping the first row (gene names) and first column (genomeID)
    data = pd.read_csv(filepath, index_col=0).values
    return data

def run_exp(dataset_paths, expdir, ae_weights_dir=None, trials=5, verbose=0):
    if not os.path.exists(expdir):
        os.makedirs(expdir)
    logfile = open(os.path.join(expdir, 'results.csv'), 'a')
    logwriter = csv.DictWriter(logfile, fieldnames=['dataset', 'trial', 'acc', 'nmi', 'ari', 'time'])
    logwriter.writeheader()

    for dataset_name, filepath in dataset_paths.items():
        data = load_genomic_data(filepath)
        x = data  # Assuming y (labels) are not used for unsupervised clustering
        input_shape = (x.shape[1],)  # Ensure input shape is correct for the DAE model
        save_dataset_dir = os.path.join(expdir, dataset_name)
        os.makedirs(save_dataset_dir, exist_ok=True)

        for trial in range(trials):
            t0 = time()
            save_trial_dir = os.path.join(save_dataset_dir, f'trial{trial}')
            os.makedirs(save_trial_dir, exist_ok=True)

            model = ConvDEC(input_shape=input_shape, units=[512, 256, 128])
            
            if ae_weights_dir:
                weights_path = os.path.join(ae_weights_dir, dataset_name, f'trial{trial}', 'ae_weights.h5')
                model.autoencoder.load_weights(weights_path)
            else:
                model.pretrain(x, epochs=500, save_dir=save_trial_dir, verbose=verbose)
            
            y_pred = model.fit(x, save_dir=save_trial_dir)
            t1 = time() - t0

            logwriter.writerow({'dataset': dataset_name, 'trial': trial, 'acc': '-', 'nmi': '-', 'ari': '-', 'time': t1})

    logfile.close()

if __name__ == "__main__":
    # Define the path to your datasets here
    data_folder = 'data'
    dataset_paths = {
        'genomic_dataset_1': os.path.join(data_folder, 'your_dataset_1.csv'),
        # Add more datasets if you have them
    }
    expdir = 'results/genomic_clustering'
    run_exp(dataset_paths, expdir, trials=5, verbose=1)
