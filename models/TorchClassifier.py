import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from BaseClassifier import BaseClassifier
import matplotlib.pyplot as plt
import seaborn as sns


class TorchClassifier(BaseClassifier):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 10) -> None:
        """ Train the model.
        Args:
            X_train (Tensor): Training data.
            y_train (Tensor): Training labels.
            epochs (int): Number of epochs to train the model.
        """
        # Create a DataLoader for the training data
        train_data = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # Define a loss function and an optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        # Train the model
        self.model.train()
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

    def predict(self, X_test: torch.Tensor) -> torch.Tensor:
        # Pass the test data through the model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
        # Return the predicted labels
        _, predicted = torch.max(outputs, 1)
        return predicted

    def confusion_matrix(self, X_test: torch.Tensor, y_test: torch.Tensor, title: str) -> None:
        # Predict the labels for the test data
        y_pred = self.predict(X_test)
        mat = confusion_matrix(y_test, y_pred)

        # Visualize the confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(title)
        plt.show()
