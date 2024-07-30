# Genomic Data Analysis Project

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Detailed Component Description](#detailed-component-description)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Autoencoder Mode](#autoencoder-mode)
   - [Clustering Mode](#clustering-mode)
6. [Data Requirements](#data-requirements)
7. [Output Details](#output-details)
8. [Customization and Extension](#customization-and-extension)
9. [Troubleshooting](#troubleshooting)
10. [License](#license)
11. [Contact](#contact)

## Introduction

This project implements a sophisticated Deep Embedded Clustering (DEC) approach for analyzing genomic data. It combines the power of autoencoders for dimensionality reduction with advanced clustering techniques to uncover patterns in high-dimensional genomic datasets. The project is particularly designed to work with one-hot encoded genomic data, making it suitable for a wide range of genomic analysis tasks.

The main goals of this project are:
1. To reduce the dimensionality of complex genomic data using deep autoencoders
2. To perform clustering on the reduced-dimension data to identify meaningful groups or patterns
3. To visualize the clustered data using t-SNE for intuitive interpretation
4. To evaluate the quality of clustering using various metrics

This project is ideal for researchers and data scientists working in bioinformatics, genomics, and related fields who need to analyze large-scale genomic datasets efficiently.

## Project Structure

The project is organized into several Python scripts, each handling a specific aspect of the analysis pipeline:

```
.
├── main_v2_individual.py
├── DenseDEC.py
├── DEC.py
├── datasets.py
└── data/
    ├── prokka_onehot_nay60k_1000.parquet
    ├── prokka_onehot_allGUT_combined.parquet
    └── humags_prokka_onehotencoded_dataset.parquet
```

## Detailed Component Description

### main_v2_individual.py

This is the main entry point of the project. It orchestrates the entire analysis pipeline and provides a command-line interface for running the autoencoder training and clustering processes. Key features include:

- Flexible configuration through command-line arguments
- Support for multiple trials and cross-validation folds
- Ability to run specific combinations of trials, folds, and encoding dimensions
- Comprehensive logging and result saving

### DenseDEC.py

This script implements the Dense Deep Embedded Clustering model, which is a variation of the DEC algorithm optimized for dense data like genomic sequences. It includes:

- A custom dense autoencoder architecture
- Methods for initializing and pretraining the autoencoder
- Integration with the base DEC class for clustering

### DEC.py

This script contains the base implementation of the Deep Embedded Clustering algorithm. It provides:

- Methods for model initialization and pretraining
- Implementation of the DenMune clustering algorithm
- Functions for evaluating clustering quality using various metrics (DBCV, CDbw)
- Utilities for data sampling and visualization

### datasets.py

This utility script handles data loading and preprocessing. It includes:

- Functions for loading one-hot encoded genomic data from Parquet files
- Data validation and error checking to ensure input data quality
- Conversion of data to formats suitable for TensorFlow processing

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/JotaKas/GenoClust-DEC.git
   cd genomic-data-analysis
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install tensorflow keras numpy pandas scikit-learn matplotlib denmune
   ```

   Note: Depending on your system and CUDA compatibility, you might need to install a specific version of TensorFlow. Refer to the [TensorFlow installation guide](https://www.tensorflow.org/install) for more details.

## Usage

The project supports two main modes of operation: Autoencoder Mode and Clustering Mode.

### Autoencoder Mode

This mode is used for training the autoencoder to reduce the dimensionality of the input genomic data.

```bash
python main_v2_individual.py autoencoder [options]
```

Options:
- `--dbs`: List of databases to process. Default: `['prokka_onehot_nay60k_1000', 'prokka_onehot_allGUT_combined', 'humags_prokka_onehotencoded_dataset']`
- `--expdir`: Directory to save results. Default: `'results/genomic_exp'`
- `--trials`: Number of trials to run. Default: `2`
- `--verbose`: Verbosity level (0, 1, or 2). Default: `1`
- `--encoding_dims`: List of encoding dimensions to try. Default: `[2, 64, 128, 256, 1024]`
- `--specific_trial`: Run a specific trial (optional)
- `--specific_fold`: Run a specific fold (optional)
- `--specific_dim`: Run a specific encoding dimension (optional)

Example:
```bash
python main_v2_individual.py autoencoder --dbs prokka_onehot_nay60k_1000 --trials 3 --encoding_dims 64 128 --verbose 2
```

This command will run the autoencoder training on the 'prokka_onehot_nay60k_1000' database for 3 trials, using encoding dimensions of 64 and 128, with maximum verbosity.

### Clustering Mode

This mode performs clustering on the reduced-dimension data obtained from the autoencoder.

```bash
python main_v2_individual.py clustering [options]
```

Options:
- `--dbs`: List of databases to process
- `--expdir`: Directory to save results. Default: `'results/genomic_exp'`
- `--knn_values`: List of KNN values for clustering. Default: `[10, 20, 50, 100, 200]`
- `--seeds`: List of random seeds for reproducibility. Default: `[42, 123, 456, 789, 101]`
- `--encoding_dims`: List of encoding dimensions to use. Default: `[2, 64, 128, 256, 1024]`
- `--specific_trial`: Run a specific trial (optional)
- `--specific_fold`: Run a specific fold (optional)
- `--specific_dim`: Run a specific encoding dimension (optional)

Example:
```bash
python main_v2_individual.py clustering --dbs prokka_onehot_allGUT_combined --knn_values 15 30 --seeds 42 789 --encoding_dims 128
```

This command will perform clustering on the 'prokka_onehot_allGUT_combined' database using KNN values of 15 and 30, seeds 42 and 789, and an encoding dimension of 128.

## Data Requirements

The project expects genomic data in Parquet format. The data should be:

1. One-hot encoded
2. Stored in the `data/` directory relative to the project root
3. Named according to the database names used in the command-line arguments

Each Parquet file should contain a table where:
- Rows represent individual genomes or samples
- Columns represent genes or features
- The first column is typically an identifier (which is ignored during processing)
- All other columns contain binary (0 or 1) values representing the one-hot encoding

Example of expected data structure:

| ID | Gene1 | Gene2 | Gene3 | ... | GeneN |
|----|-------|-------|-------|-----|-------|
| 1  | 0     | 1     | 0     | ... | 1     |
| 2  | 1     | 0     | 1     | ... | 0     |
| ...| ...   | ...   | ...   | ... | ...   |

## Output Details

The project generates various outputs in the specified experiment directory:

1. **Trained Models**:
   - `autoencoder_model.h5`: The full autoencoder model
   - `encoder_model.h5`: The encoder part of the autoencoder
   - `ae_weights.h5`: Weights of the trained autoencoder

2. **Training Logs**:
   - `pretrain_log.csv`: Log of the autoencoder pretraining process

3. **Encoded Features**:
   - `full_features.txt`: Encoded features for the entire dataset

4. **t-SNE Visualizations**:
   - `tsne_features_seed_{seed}.txt`: t-SNE transformed features for each seed

5. **Clustering Results**:
   - `predicted_clusters_knn_{knn}_seed_{seed}.txt`: Predicted cluster labels
   - `clustering_results_seed_{seed}_knn_{knn}.csv`: Detailed clustering metrics

6. **Plots**:
   - `2D_plot_knn_{knn}_seed_{seed}.png`: 2D scatter plot of clustered data

These outputs allow for comprehensive analysis and interpretation of the results, including model performance, clustering quality, and visual representation of the data structure.

## Customization and Extension

The project is designed to be modular and extensible. Here are some ways you can customize or extend the functionality:

1. **Adding New Databases**: Simply add new Parquet files to the `data/` directory and include their names in the `--dbs` argument.

2. **Modifying the Autoencoder Architecture**: Edit the `DenseAutoencoder` function in `DenseDEC.py` to change the network architecture.

3. **Implementing New Clustering Algorithms**: Extend the `DEC` class in `DEC.py` to include additional clustering methods.

4. **Adding New Evaluation Metrics**: Incorporate additional clustering quality metrics in the `evaluate_clustering` method of `DEC.py`.

5. **Customizing Visualizations**: Modify the plotting code in `DEC.py` to create different types of visualizations.

## Troubleshooting

Common issues and their solutions:

1. **Out of Memory Errors**: If you encounter memory issues, try reducing the batch size or using a smaller encoding dimension.

2. **CUDA Errors**: Ensure that your TensorFlow installation is compatible with your CUDA version. You may need to install a specific TensorFlow version.

3. **Data Loading Errors**: Verify that your Parquet files are in the correct format and location. Use the `--verbose` flag to get more detailed error messages.

4. **Poor Clustering Results**: Experiment with different KNN values and encoding dimensions. Consider preprocessing your data further if results are consistently poor.

## License

GPL-3.0 license

## Contact

Jonas C Kasmanas
Email: jonas.kasmanas@ufz.de

For bug reports and feature requests, please open an issue on the GitHub repository.

## References

Abbas, M., El-Zoghabi, A., & Shoukry, A. (2021). DenMune: Density peak based clustering using mutual nearest neighbors. Pattern Recognition, 109(107589), 107589. https://doi.org/10.1016/j.patcog.2020.107589

Ren, Y., Wang, N., Li, M., & Xu, Z. (2020). Deep density-based image clustering. Knowledge-Based Systems, 197(105841), 105841. https://doi.org/10.1016/j.knosys.2020.105841


