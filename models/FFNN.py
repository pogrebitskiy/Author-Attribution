"""
Simple class for a feed-forward neural network (FFNN) model.

By: David Pogrebitskiy and Jacob Ostapenko
Date: 04/08/2023
"""

import torch.nn as nn
import torch


class FFNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        """
        Initialize the FFNN.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden layer.
            num_classes (int): The number of classes.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FFNN.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output data.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
