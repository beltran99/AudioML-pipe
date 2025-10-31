import torch
import torch.nn as nn

device = torch.device("cpu")

class FCNN(nn.Module):
  def __init__(self, input_size: int = 40, hidden_sizes: list = [128, 256, 256, 128], output_size: int = 10, activation_fn = nn.ReLU()):
    """ Instantiates a Fully-Connected Neural Network

      Args:
          input_size (int): _description_
          hidden_sizes (list): _description_
          output_size (int): _description_
    """
      
    super().__init__()
      
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    self.layers = nn.ModuleList()
    self.bn = nn.ModuleList()
    
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)

    for i, layer_size in enumerate(layer_sizes[:-1]):
        self.layers.append(nn.Linear(layer_size, layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            self.bn.append(nn.BatchNorm1d(layer_sizes[i + 1])) 

  def forward(self, x):
    for linear, bn in zip(self.layers[:-1], self.bn):
        x = self.relu(bn(linear(x)))
    x = self.softmax(self.layers[-1](x))
    
    return x

class CNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_sizes):
        pass
    
    def forward(self, x):
        pass