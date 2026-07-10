import torch.nn as nn


class MLP(nn.Module):
    """
    A simple MLP allowing for arbitrary:
    - i/o dimensions
    - hidden size
    - number of layers

    Parameters
    ----------
    in_dim: int
        Input dimension, equal to the number of features in the dataset
    out_dim: int
        Output dimension. 1 in regression, the number of classes in classification
    hidden: int
        Hidden size (the same for all the hidden layers)
    n_layers: int
        Number of layers
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        layers = [[nn.Linear(in_dim, hidden), nn.ReLU()]]
        layers.extend(
            [[nn.Linear(hidden, hidden), nn.ReLU()] for _ in range(n_layers - 1)]
        )
        layers.append([nn.Linear(hidden, out_dim)])
        self.net = nn.Sequential(*[layer for block in layers for layer in block])

    def forward(self, x):
        return self.net(x)
