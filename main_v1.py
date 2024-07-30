import argparse
from DenseDEC import DenseDEC
import os
import csv
import gc
from datasets import load_genomic_data
from keras.initializers import VarianceScaling
import numpy as np
from time import time
from sklearn.model_selection import KFold
from keras.models import load_model
import tensorflow as tf

def clear_session():
    tf.keras.backend.clear_session()
    gc.collect()

def run_autoencoder_mode(dbs, expdir, trials=5, n_splits=4, verbose=0, encoding_dims=[64, 128, 256]):
    if not os.path.exists(expdir):
        os.makedirs(expdir)
    logfile = open(os.path.join(expdir, 'autoencoder_training_summary.csv'), 'w', newline='')
    logwriter = csv.DictWriter(logfile, fieldnames=['db', 'trial', 'fold', 'encoding_dim', 'train_loss', 'val_loss'])
    logwriter.writeheader()

    for db in dbs:
        db_dir = os.path.join(expdir, db)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

        x = load_genomic_data(os.path.join('data/', db + '.parquet'))
        input_dim = x.shape[1]
        init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform', seed=123)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for trial in range(trials):
            trial_dir = os.path.join(db_dir, f'trial{trial}')
            if not os.path.exists(trial_dir):
                os.makedirs(trial_dir)

            for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
                fold_dir = os.path.join(trial_dir, f'fold{fold}')
                if not os.path.exists(fold_dir):
                    os.makedirs(fold_dir)

                x_train, x_val = x[train_idx], x[val_idx]

                for encoding_dim in encoding_dims:
                    dim_dir = os.path.join(fold_dir, f'dim{encoding_dim}')
                    if not os.path.exists(dim_dir):
                        os.makedirs(dim_dir)

                    # Clear TensorFlow session and garbage collect
                    clear_session()

                    model = DenseDEC(input_dim=input_dim, encoding_dim=encoding_dim, init=init)
                    history = model.pretrain(x_train, x_val, optimizer='adam', epochs=200, batch_size=256, save_dir=dim_dir, verbose=verbose)

                    train_loss = history.history['loss'][-1]
                    val_loss = history.history['val_loss'][-1]

                    logwriter.writerow({
                        'db': db,
                        'trial': trial,
                        'fold': fold,
                        'encoding_dim': encoding_dim,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    })

                    del model, history
                    clear_session()

    logfile.close()

def run_clustering_mode(dbs, expdir, knn_values=[5, 10, 15, 20], seeds=[42, 123, 456, 789, 101], encoding_dims=[64, 128, 256]):
    for db in dbs:
        db_dir = os.path.join(expdir, db)

        x = load_genomic_data(os.path.join('data/', db + '.parquet'))
        input_dim = x.shape[1]
        init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform', seed=123)

        for trial in os.listdir(db_dir):
            trial_dir = os.path.join(db_dir, trial)
            if not os.path.isdir(trial_dir):
                continue

            for fold in os.listdir(trial_dir):
                fold_dir = os.path.join(trial_dir, fold)
                if not os.path.isdir(fold_dir):
                    continue

                for encoding_dim in encoding_dims:
                    dim_dir = os.path.join(fold_dir, f'dim{encoding_dim}')
                    if not os.path.isdir(dim_dir):
                        continue

                    autoencoder_path = os.path.join(dim_dir, 'autoencoder_model.h5')
                    encoder_path = os.path.join(dim_dir, 'encoder_model.h5')

                    if not (os.path.exists(autoencoder_path) and os.path.exists(encoder_path)):
                        print(f"Skipping {dim_dir} - trained models not found")
                        continue

                    clear_session()

                    model = DenseDEC(input_dim=input_dim, encoding_dim=encoding_dim, init=init)
                    model.autoencoder = load_model(autoencoder_path)
                    model.encoder = load_model(encoder_path)

                    clustering_results = model.fit(x, knn_values=knn_values, seeds=seeds, save_dir=dim_dir)

                    # Save clustering results
                    with open(os.path.join(dim_dir, 'clustering_results.csv'), 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        first_seed = list(clustering_results.keys())[0]
                        first_knn = list(clustering_results[first_seed].keys())[0]
                        metric_keys = list(clustering_results[first_seed][first_knn].keys())
                        writer.writerow(['seed', 'knn', 'encoding_dim'] + metric_keys)
                        for seed, seed_results in clustering_results.items():
                            for knn, metrics in seed_results.items():
                                row = [seed, knn, encoding_dim]
                                for key in metric_keys:
                                    if key in ['cluster_sizes', 'valid_cluster_sizes']:
                                        row.append(str(metrics[key]))  # Convert dict to string for CSV
                                    else:
                                        row.append(metrics[key])
                                writer.writerow(row)

                    clear_session()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run autoencoder training or clustering on genomic data.")
    parser.add_argument("mode", choices=["autoencoder", "clustering"], help="Mode to run: 'autoencoder' or 'clustering'")
    parser.add_argument("--dbs", nargs='+', default=['prokka_onehot_nay60k_1000', 'prokka_onehot_allGUT_combined', 'humags_prokka_onehotencoded_dataset'], help="List of databases to process")
    parser.add_argument("--expdir", default='results/genomic_exp', help="Directory to save results")
    parser.add_argument("--trials", type=int, default=2, help="Number of trials (for autoencoder mode)")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--knn_values", nargs='+', type=int, default=[10, 20, 50, 100, 200], help="KNN values for clustering")
    parser.add_argument("--seeds", nargs='+', type=int, default=[42, 123, 456, 789, 101], help="Seeds for clustering")
    parser.add_argument("--encoding_dims", nargs='+', type=int, default=[2, 64, 128, 256, 1024], help="Encoding dimensions to try")

    args = parser.parse_args()

    if args.mode == "autoencoder":
        run_autoencoder_mode(args.dbs, expdir=args.expdir, verbose=args.verbose, trials=args.trials, encoding_dims=args.encoding_dims)
    elif args.mode == "clustering":
        run_clustering_mode(args.dbs, expdir=args.expdir, knn_values=args.knn_values, seeds=args.seeds, encoding_dims=args.encoding_dims)
