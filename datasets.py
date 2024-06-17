import pandas as pd
import numpy as np

def load_genomic_data(file_path):
    """
    Load one-hot encoded genomic data, ensuring it is suitable for TensorFlow.

    Parameters:
    - file_path: Path to the genomic data file. Assumes data in Parquet format.

    Returns:
    - X: The loaded genomic data as a NumPy array, with genomes as rows and genes as columns.
    """
    # Load the data with Pandas
    df = pd.read_parquet(file_path).iloc[:, 1:]

    # Check for non-numeric columns and report them
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_columns) > 0:
        raise ValueError(f"Non-numeric columns found: {non_numeric_columns}. Please preprocess these columns before loading.")

    # Check for missing values
    if df.isnull().values.any():
        raise ValueError("Missing values detected. Please impute or remove missing values before loading.")

    # Convert to NumPy array and ensure it's of floating-point type
    X = df.values.astype(np.float32)

    # Debugging: print out its shape and data type to confirm it's as expected
    print(f"Data shape: {X.shape}, Data type: {X.dtype}")

    return X
