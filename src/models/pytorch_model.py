import torch.nn as nn
from typing import List, Optional


class HousingPriceModel(nn.Module):
    """
    Fully-connected neural network for LogPrice regression.

    Architecture is built dynamically from hidden_dims, with optional
    Dropout after each hidden layer. Output is a single scalar (LogPrice).
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def get_model(
    input_dim: int = 9,
    hidden_dims: Optional[List[int]] = None,
    output_dim: int = 1,
    dropout_rate: float = 0.0,
) -> HousingPriceModel:
    """Instantiate and return a HousingPriceModel with the given configuration."""
    return HousingPriceModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
    )