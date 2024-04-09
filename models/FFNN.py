"""
Simple class for a feed-forward neural network (FFNN) model.

By: David Pogrebitskiy and Jacob Ostapenko
Date: 04/08/2023
"""

import torch.nn as nn
import torch


class FFNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """
        Initialize the FFNN.

        Args:
            input_dim (int): The size of the input features.
            hidden_dim (int): The size of the hidden layer.
            output_dim (int): The number of classes.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

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
