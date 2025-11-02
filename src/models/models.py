import sys

import torch
import torch.nn as nn

from src.logger import get_logger

DEFAULT_FCNN = {
    "input_size": 40,
    "hidden_sizes": [100, 100, 100],
    "output_size": 10,
    "activation_fn": {"type": "ReLU"}
}
DEFAULT_OPT = {
    "type": "SGD"
}


class FCNN(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, activation_fn: dict) -> None:
        """Fully-Connected Neural Network.

        Args:
            input_size (int): Size of input layer.
            hidden_sizes (list): List of sizes of hidden layers.
            output_size (int): Size of ouput layer.
        """
        
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_fn = self.get_activation_fn(activation_fn)
        
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        self.layers = nn.ModuleList()
        self.bn = nn.ModuleList()
        
        self.activation_out = nn.Softmax(dim=1)

        for i, layer_size in enumerate(layer_sizes[:-1]):
            self.layers.append(nn.Linear(layer_size, layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.bn.append(nn.BatchNorm1d(layer_sizes[i + 1])) 

    def forward(self, x):
        for linear, bn in zip(self.layers[:-1], self.bn):
            x = self.activation_fn(bn(linear(x)))
        x = self.activation_out(self.layers[-1](x))
        
        return x

    def get_activation_fn(self, config):
        logger = get_logger(__name__)
        fn_name = config.get("type", {})
        if fn_name and hasattr(nn, fn_name):
            params = config.get("params", {})
            fn = getattr(nn, fn_name)
            try:
                activation_fn = fn(**params)
            except Exception as err:
                logger.error(f"Unexpected error when trying to get activation function\n\tError details: {err=}, {type(err)=}")
                sys.exit(1)
                
        return activation_fn