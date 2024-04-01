"""
The BaseClassifier class is an abstract class that all classifiers must inherit from. It provides a common interface
By: David Pogrebitskiy and Jacob Ostapenko
Date: 03/31/2023
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import utils


class BaseClassifier(ABC):
    """
    The BaseClassifier class is an abstract class that all classifiers must inherit from. It provides a common interface.

    Attributes:
        model: The underlying classifier model.
    """

    def __init__(self, model: Any) -> None:
        """
        Initialize the BaseClassifier.

        Args:
            model (object): The underlying classifier model.
        """
        self.model = model

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> None:
        """
        Abstract method for training that all classifiers must implement.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        pass

    @abstractmethod
    def predict(self, X_test: Any) -> Any:
        """
        Abstract method for prediction that all classifiers must implement.

        Args:
            X_test (array-like or sparse matrix): Test samples.

        Returns:
            Any: Predicted labels for X_test.
        """
        pass

    @abstractmethod
    def top_k_accuracy(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method for top k accuracy.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Top k accuracy.
        """
        pass

    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, Any]:
        """
        Method to evaluate the classifier.

        Args:
            X_test (array-like or sparse matrix): Test samples.
            y_test (array-like): True labels for X_test.

        Returns:
            dict: A dictionary containing precision, recall, f1-score, and accuracy.
        """
        y_pred = self.predict(X_test)
        return utils.get_prfa(y_test, y_pred)
