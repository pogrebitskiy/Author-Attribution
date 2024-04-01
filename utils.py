"""
Utils for CS4120 Project
By: David Pogrebitskiy and Jacob Ostapenko
Date: 03/30/2023
"""

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import scipy as sp


def get_feature_matrix(file_path: str) -> np.array:
    """
    Read a file and return a feature matrix.
    Args:
        file_path: path to the file
    """
    if file_path.endswith('.npy'):
        return np.load(file_path, allow_pickle=True)
    elif file_path.endswith('.npz'):
        return sp.sparse.load_npz(file_path).toarray()


def get_dense_matrix(sparse_matrix: sp.sparse._csr.csr_matrix) -> np.array:
    """
    Convert a sparse matrix to a dense matrix.
    Args:
        sparse_matrix: the sparse matrix
    Returns:
        the dense matrix
    """
    return sparse_matrix.toarray()


def get_prfa(y_true: list, y_pred: list) -> dict:
    """
    Calculate precision, recall, f1, and accuracy for a given set of predictions and labels.
    Args:
        y_true: list of true labels
        y_pred: list of predictions
        verbose: whether to print the metrics
    Returns:
        tuple of precision, recall, f1, and accuracy
    """

    precision_val = precision_score(y_true, y_pred, average='weighted')
    recall_val = recall_score(y_true, y_pred, average='weighted')
    f1_val = f1_score(y_true, y_pred, average='weighted')
    accuracy_val = accuracy_score(y_true, y_pred)

    return {'Precision': precision_val,
            'Recall': recall_val,
            'F1': f1_val,
            'Accuracy': accuracy_val}


def get_label(encoder, encoding):
    """
    Get the label from the encoding.
    Args:
        encoder: the label encoder
        encoding: the encoding
    Returns:
        the label
    """
    return encoder.inverse_transform([encoding])

