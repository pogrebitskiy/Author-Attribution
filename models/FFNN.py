"""
Simple class for a feed-forward neural network (FFNN) model.

By: David Pogrebitskiy and Jacob Ostapenko
Date: 04/08/2023
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class FFNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5) -> None:
        """
        Initialize the FFNN.

        Args:
            input_dim (int): The size of the input features.
            hidden_dim (int): The size of the hidden layer.
            output_dim (int): The number of classes.
            dropout (float): The dropout rate. Defaults to 0.5
        """
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FFNN.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output data.
        """
        x = F.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
