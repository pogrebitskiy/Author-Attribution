"""
This class implements a wrapper for scikit-learn classifiers. It inherits from the BaseClassifier class

By: David Pogrebitskiy and Jacob Ostapenko
Date: 03/31/2023
"""
from models.BaseClassifier import BaseClassifier
import utils
import numpy as np
from typing import Any, Dict


class SklearnClassifier(BaseClassifier):
    """
    This class implements a wrapper for scikit-learn classifiers. It inherits from the BaseClassifier class.

    Attributes:
        model: The scikit-learn classifier model.
    """

    def __init__(self, model, **kwargs: Any) -> None:
        """
        Initialize the SklearnClassifier.

        Args:
            model (ClassifierMixin): The scikit-learn classifier model.
            **kwargs: Additional keyword arguments to be passed to the model constructor.
        """
        super().__init__(model(**kwargs))

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model.

        Args:
            X_train (ndarray): Training data.
            y_train (ndarray): Target labels.

        Returns:
            None
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X_test (ndarray): Test data.

        Returns:
            ndarray: Predicted labels.
        """
        return self.model.predict(X_test)

    def top_k_accuracy(self, X_test: np.ndarray, y_test: np.ndarray, k: int) -> float:
        """
        Calculate top k accuracy.

        Args:
            X_test (ndarray): Test data.
            y_test (ndarray): True labels.
            k (int): Value of k for top k accuracy.

        Returns:
            float: Top k accuracy.
        """
        classes = self.model.classes_
        y_pred_proba = self.model.predict_proba(X_test)
        y_test_encoded = np.array([list(classes).index(label) for label in y_test])
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        match_array = np.logical_or.reduce(y_test_encoded.reshape(-1, 1) == top_k_preds, axis=1)
        return np.mean(match_array)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, top_k: int = 3) -> Dict[str, Any]:
        """
        Evaluate the model using precision, recall, f1, accuracy, and top k accuracy.

        Args:
            X_test (ndarray): Test data.
            y_test (ndarray): True labels.
            top_k (int): Value of k for top k accuracy. Default is 3.

        Returns:
            dict: A dictionary of metrics.
        """
        y_pred = self.predict(X_test)
        metrics = utils.get_prfa(y_test, y_pred)
        metrics['Top-k Accuracy'] = self.top_k_accuracy(X_test, y_test, top_k)
        return metrics
