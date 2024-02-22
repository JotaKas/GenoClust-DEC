import os
import csv
import numpy as np
from time import time
from keras.initializers import VarianceScaling
from ConvDEC import ConvDEC

def run_exp(datasets, expdir, ae_weights_dir=None, trials=5, verbose=0):
    if not os.path.exists(expdir):
        os.makedirs(expdir)
    logfile = open(os.path.join(expdir, 'results.csv'), 'a')
    logwriter = csv.DictWriter(logfile, fieldnames=['dataset', 'trial', 'acc', 'nmi', 'ari', 'time'])
    logwriter.writeheader()

    for dataset_name, data in datasets.items():
        x, y = data  # Assuming x is the one-hot encoded genomic data, y are the labels (if available)
        input_shape = x.shape[1:]
        save_dataset_dir = os.path.join(expdir, dataset_name)
        if not os.path.exists(save_dataset_dir):
            os.makedirs(save_dataset_dir)

        # Initialize results storage
        results = np.zeros((trials, 4))  # Storing acc, nmi, ari, time for each trial
        
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

            # Optionally: Evaluate clustering performance using true labels
            # acc = metrics.acc(y, y_pred)
            # nmi = metrics.nmi(y, y_pred)
            # ari = metrics.ari(y, y_pred)
            # results[trial] = [acc, nmi, ari, t1]
            
            # Log results for this trial
            logwriter.writerow({'dataset': dataset_name, 'trial': trial, 'acc': '-', 'nmi': '-', 'ari': '-', 'time': t1})

        # After all trials, optionally calculate mean performance
        # mean_results = np.mean(results, axis=0)
        # logwriter.writerow({'dataset': dataset_name, 'trial': 'mean', 'acc': mean_results[0], 'nmi': mean_results[1], 'ari': mean_results[2], 'time': mean_results[3]})

    logfile.close()

if __name__ == "__main__":
    # Example: Load your genomic data here
    # datasets should be a dict with dataset names as keys and (data, labels) tuples as values
    datasets = {
        'genomic_dataset_1': (np.random.rand(100, 784), np.random.randint(0, 2, 100)),  # Example data
    }
    expdir = 'results/genomic_clustering'
    run_exp(datasets, expdir, trials=5, verbose=1)
