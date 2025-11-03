import sys

import torch
import torch.nn as nn

from src.logger import get_logger


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
        
        self.activation_out = nn.LogSoftmax(dim=1)

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
    
    
class CNN(nn.Module):
    def __init__(self, n_in, n_out=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.pool1 = nn.MaxPool2d((3, 1), stride=2)
        self.conv2 = nn.Conv2d(100, 32, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.pool2 = nn.MaxPool2d((2, 1), stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.pool3 = nn.MaxPool2d((2, 1), stride=2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        
        if n_in == 13:
            self.fc1 = nn.Linear(1600, 1024) # hardcoded for input features of shape (13, 150)
        elif n_in == 20:
            self.fc1 = nn.Linear(3200, 1024) # hardcoded for input features of shape (20, 200)
        else:
            logger = get_logger(__name__)
            logger.error(f"The CNN module currently supports input features with shapes of (13, 200) or (20, 200).")
            sys.exit(1)
            
        self.fc2 = nn.Linear(1024, 512)            
        self.fc3 = nn.Linear(512, n_out)

    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        x = torch.nn.ReLU()(x)
        x = self.pool1(x)

        # Pass data through conv2
        x = self.conv2(x)
        x = torch.nn.ReLU()(x)
        x = self.pool2(x)
        
        # Pass data through conv3
        x = self.conv3(x)
        x = torch.nn.ReLU()(x)
        x = self.pool3(x)
        
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        
        # Pass data through fc1
        x = self.fc1(x)
        x = torch.nn.ReLU()(x)
        
        # Pass data through fc2
        x = self.fc2(x)
        x = torch.nn.ReLU()(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        output = torch.nn.LogSoftmax(dim=1)(x)
        
        return output