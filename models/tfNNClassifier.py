import tensorflow as tf
from models.BaseClassifier import BaseClassifier
import numpy as np
import utils
from typing import Any, Dict


class tfNNClassifier(BaseClassifier):
    """
    This class implements a wrapper for TensorFlow neural network classifiers. It inherits from the BaseClassifier class.

    Attributes:
        model: The TensorFlow neural network model.
    """

    def __init__(self, model: tf.keras.models.Model, **kwargs: Any) -> None:
        """
        Initialize the tfNNClassifier.

        Args:
            model (tf.keras.models.Model): The TensorFlow neural network model.
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
        #optimizer/loss/metric can be changed upon desire
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=10)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X_test (ndarray): Test data.

        Returns:
            ndarray: Predicted labels.
        """
        return self.model.predict(X_test)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, top_k: int) -> float:
        """
        Evaluate the model.

        Args:
            X_test (ndarray): Test data.
            y_test (ndarray): True labels.
            top_k (int): Value of k for top k accuracy.

        Returns:
            float: Top k accuracy.
        """
        y_pred = self.predict(X_test)
        metrics = utils.get_prfa(y_test, y_pred)
        metrics['Top-k Accuracy'] = self.top_k_accuracy(X_test, y_test, top_k)
        return metrics
    
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
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['top_k_categorical_accuracy'])
        return self.model.evaluate(X_test, y_test, verbose=0)[1]