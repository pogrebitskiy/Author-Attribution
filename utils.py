"""
Utils for CS4120 Project
By: David Pogrebitskiy and Jacob Ostapenko
Date: 03/30/2023
"""

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import scipy as sp
from sklearn.preprocessing import StandardScaler


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


def scale_feature_matrix(feature_matrix: np.array) -> np.array:
    """
    Scale the feature matrix.
    Args:
        feature_matrix: the feature matrix
    Returns:
        the scaled feature matrix
    """
    scaler = StandardScaler()
    return scaler.fit_transform(feature_matrix)
