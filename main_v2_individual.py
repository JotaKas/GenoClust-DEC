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

def run_autoencoder_mode(dbs, expdir, trials=5, n_splits=4, verbose=0, encoding_dims=[64, 128, 256], specific_trial=None, specific_fold=None, specific_dim=None):
    print(f"Starting run_autoencoder_mode with parameters:")
    print(f"dbs: {dbs}")
    print(f"expdir: {expdir}")
    print(f"trials: {trials}")
    print(f"n_splits: {n_splits}")
    print(f"verbose: {verbose}")
    print(f"encoding_dims: {encoding_dims}")
    print(f"specific_trial: {specific_trial}")
    print(f"specific_fold: {specific_fold}")
    print(f"specific_dim: {specific_dim}")

    if not os.path.exists(expdir):
        os.makedirs(expdir)
        print(f"Created expdir: {expdir}")

    for db in dbs:
        print(f"Processing database: {db}")
        db_dir = os.path.join(expdir, db)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            print(f"Created db_dir: {db_dir}")

        print(f"Loading data from: {os.path.join('data/', db + '.parquet')}")
        x = load_genomic_data(os.path.join('data/', db + '.parquet'))
        input_dim = x.shape[1]
        print(f"Data loaded. Shape: {x.shape}")

        init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform', seed=123)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        log_file = os.path.join(db_dir, 'autoencoder_training_summary.csv')
        log_exists = os.path.exists(log_file)
        print(f"Log file: {log_file}")
        print(f"Log exists: {log_exists}")

        with open(log_file, 'a', newline='') as logfile:
            logwriter = csv.DictWriter(logfile, fieldnames=['db', 'trial', 'fold', 'encoding_dim', 'train_loss', 'val_loss'])
            if not log_exists:
                logwriter.writeheader()

            # If specific_trial is set, only process that trial
            trials_to_process = [specific_trial] if specific_trial is not None else range(trials)

            for trial in trials_to_process:
                print(f"Processing trial: {trial}")
                trial_str = f'trial{trial}'
                trial_dir = os.path.join(db_dir, trial_str)
                if not os.path.exists(trial_dir):
                    os.makedirs(trial_dir)
                    print(f"Created trial_dir: {trial_dir}")

                # If specific_fold is set, only process that fold
                folds_to_process = [specific_fold] if specific_fold is not None else range(n_splits)

                for fold in folds_to_process:
                    print(f"Processing fold: {fold}")
                    fold_str = f'fold{fold}'
                    fold_dir = os.path.join(trial_dir, fold_str)
                    if not os.path.exists(fold_dir):
                        os.makedirs(fold_dir)
                        print(f"Created fold_dir: {fold_dir}")

                    train_idx, val_idx = list(kf.split(x))[fold]
                    x_train, x_val = x[train_idx], x[val_idx]

                    # If specific_dim is set, only process that dimension
                    dims_to_process = [specific_dim] if specific_dim is not None else encoding_dims

                    for encoding_dim in dims_to_process:
                        print(f"Processing encoding_dim: {encoding_dim}")
                        dim_dir = os.path.join(fold_dir, f'dim{encoding_dim}')
                        if not os.path.exists(dim_dir):
                            os.makedirs(dim_dir)
                            print(f"Created dim_dir: {dim_dir}")

                        if os.path.exists(os.path.join(dim_dir, 'autoencoder_model.h5')):
                            print(f"Skipping {dim_dir} - already processed")
                            continue

                        print(f"Processing {dim_dir}")
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
                        logfile.flush()
                        print(f"Logged results for trial {trial}, fold {fold}, encoding_dim {encoding_dim}")

                        del model, history
                        clear_session()

    print("Autoencoder training completed.")

def run_clustering_mode(dbs, expdir, knn_values=[5, 10, 15, 20], seeds=[42, 123, 456, 789, 101], encoding_dims=[64, 128, 256], specific_trial=None, specific_fold=None, specific_dim=None):
    for db in dbs:
        db_dir = os.path.join(expdir, db)

        x = load_genomic_data(os.path.join('data/', db + '.parquet'))
        input_dim = x.shape[1]
        init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform', seed=123)

        for trial in os.listdir(db_dir):
            if specific_trial is not None and int(trial.replace('trial', '')) != specific_trial:
                continue

            trial_dir = os.path.join(db_dir, trial)
            if not os.path.isdir(trial_dir):
                continue

            for fold in os.listdir(trial_dir):
                if specific_fold is not None and int(fold.replace('fold', '')) != specific_fold:
                    continue

                fold_dir = os.path.join(trial_dir, fold)
                if not os.path.isdir(fold_dir):
                    continue

                for encoding_dim in encoding_dims:
                    if specific_dim is not None and encoding_dim != specific_dim:
                        continue

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

                    # Check which seed and knn combinations need to be run
                    combinations_to_run = []
                    for seed in seeds:
                        tsne_file = os.path.join(dim_dir, f'tsne_features_seed_{seed}.txt')
                        if not os.path.exists(tsne_file):
                            combinations_to_run.extend([(seed, knn) for knn in knn_values])
                            continue

                        for knn in knn_values:
                            clustering_results_file = os.path.join(dim_dir, f'clustering_results_seed_{seed}_knn_{knn}.csv')
                            predicted_clusters_file = os.path.join(dim_dir, f'predicted_clusters_knn_{knn}_seed_{seed}.txt')

                            if not (os.path.exists(clustering_results_file) and os.path.exists(predicted_clusters_file)):
                                combinations_to_run.append((seed, knn))

                    if not combinations_to_run:
                        print(f"Skipping {dim_dir} - all outputs already exist")
                        continue

                    print(f"Running clustering for {dim_dir} with {len(combinations_to_run)} combinations")
                    clustering_results = model.fit(x, knn_values=[knn for _, knn in combinations_to_run],
                                                   seeds=[seed for seed, _ in combinations_to_run],
                                                   save_dir=dim_dir)

                    # Save clustering results
                    for seed, seed_results in clustering_results.items():
                        for knn, metrics in seed_results.items():
                            results_file = os.path.join(dim_dir, f'clustering_results_seed_{seed}_knn_{knn}.csv')
                            with open(results_file, 'w', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow(['metric', 'value'])
                                for key, value in metrics.items():
                                    writer.writerow([key, value])

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
    parser.add_argument("--specific_trial", type=int, help="Specific trial to run")
    parser.add_argument("--specific_fold", type=int, help="Specific fold to run")
    parser.add_argument("--specific_dim", type=int, help="Specific encoding dimension to run")

    args = parser.parse_args()

    if args.mode == "autoencoder":
        run_autoencoder_mode(args.dbs, expdir=args.expdir, verbose=args.verbose, trials=args.trials, encoding_dims=args.encoding_dims,
                             specific_trial=args.specific_trial, specific_fold=args.specific_fold, specific_dim=args.specific_dim)
    elif args.mode == "clustering":
        run_clustering_mode(args.dbs, expdir=args.expdir, knn_values=args.knn_values, seeds=args.seeds, encoding_dims=args.encoding_dims,
                            specific_trial=args.specific_trial, specific_fold=args.specific_fold, specific_dim=args.specific_dim)
