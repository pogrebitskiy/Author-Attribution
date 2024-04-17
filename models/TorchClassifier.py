"""
This class implements a wrapper for Torch classifiers. It inherits from the BaseClassifier class

By: David Pogrebitskiy and Jacob Ostapenko
Date: 04/08/2023
"""
from typing import Any

import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from models.BaseClassifier import BaseClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class TorchClassifier(BaseClassifier):
    """
    This class implements a PyTorch classifier. It inherits from the BaseClassifier class.
    The classifier can be trained on a GPU if one is available.
    """

    def __init__(self, model: torch.nn.Module, **kwargs: Any) -> None:
        """
        Initialize the TorchClassifier.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
        """
        super().__init__(model(**kwargs))
        # Check if a GPU is available and if so, move the model to the GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 10, batch_size: int = 32,
              lr: float = 0.001, weight_decay: float = 0.0) -> None:
        """
        Train the model.

        Args:
            X_train (torch.Tensor): Training data.
            y_train (torch.Tensor): Training labels.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Batch size for training.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
        """
        # Create a DataLoader for the training data
        train_data = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        # Define a loss function and an optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Train the model
        self.model.train()
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                # Move the inputs and labels to the GPU if one is available
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            # Print the progress of the training
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def predict(self, X_test: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.

        Args:
            X_test (torch.Tensor): Test data.

        Returns:
            torch.Tensor: Predicted labels.
        """
        # Move the test data to the GPU if one is available
        X_test = X_test.to(self.device)

        # Set the model to evaluation mode and make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)

        # Return the predicted label
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu()

    def get_misclassified_stats(self, misclassified_indices: torch.Tensor) -> dict:
        """
        Get the misclassified stats.

        Args:
            misclassified_indices (torch.Tensor): Indices of the misclassified samples.

        Returns:
            dict: A dictionary containing the misclassified stats.
        """
        preprocessesed_data = pd.read_pickle('data/cleaned_articles.pkl')
        misclassified_rows = preprocessesed_data.iloc[misclassified_indices]
        misclassified_stats = {}
        misclassified_stats['mean_article_length'] = np.mean([len(row['clean_content'].split()) for index, row in misclassified_rows.iterrows()])
        misclassified_stats['publishers'] = misclassified_rows['publication'].value_counts()
        misclassified_stats['authors'] = misclassified_rows['author'].value_counts()
        return misclassified_stats

    def get_confusion_matrix(self, X_test: torch.Tensor, y_test: torch.Tensor, title: str,
                             test_index: torch.Tensor) -> None:
        """
        Compute and visualize the confusion matrix.

        Args:
            X_test (torch.Tensor): Test data.
            y_test (torch.Tensor): True labels.
            title (str): Title for the confusion matrix plot.
            test_index (torch.Tensor): Index of the test data.
        """
        # Predict the labels
        y_pred = self.predict(X_test)
        mat = confusion_matrix(y_test, y_pred)

        # misclassification stats
        misclassified_indices_raw = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]
        misclassified_indices_original_dataset = test_index[misclassified_indices_raw]
        misclassified_stats = self.get_misclassified_stats(misclassified_indices_original_dataset)
        print(misclassified_stats)

        # Visualize the confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(title)
        plt.show()
