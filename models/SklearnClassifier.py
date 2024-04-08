"""
This class implements a wrapper for scikit-learn classifiers. It inherits from the BaseClassifier class

By: David Pogrebitskiy and Jacob Ostapenko
Date: 03/31/2023
"""
from models.BaseClassifier import BaseClassifier
import utils
import numpy as np
from typing import Any, Dict
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


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

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model using precision, recall, and f1, accuracy

        Args:
            X_test (ndarray): Test data.
            y_test (ndarray): True labels.

        Returns:
            dict: A dictionary of metrics.
        """
        y_pred = self.predict(X_test)
        metrics = utils.get_prfa(y_test, y_pred)
        return metrics

    def get_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray, title: str) -> None:
        """
        Get the confusion matrix.

        Args:
            X_test (ndarray): Test data.
            y_test (ndarray): True labels.
            title (str): Title for the confusion matrix.

        Returns:
            ndarray: The confusion matrix.
        """
        y_pred = self.predict(X_test)
        mat = confusion_matrix(y_test, y_pred)

        # Visualize the confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(title)
        plt.show()