import torch
import torch.nn as nn


class FullyConnectedModule(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], **kwargs):
        """
        Parameters
        ----------
        input_dim : int
        output_dim : int
        hidden_dims : list-like
        kwargs : dict, keyword arguments for BuildFullyConnected function (e.g. dropout_rate)
        """
        super(FullyConnectedModule, self).__init__()
        
        dlayers = [input_dim] + hidden_dims + [output_dim]
        self.module = BuildFullyConnected(dlayers, **kwargs)
    
    def forward(self, x):
        return self.module(x)
    

def BuildFullyConnected(layers, activation="relu", dropout_rate=0.0, batchnorm=False):
    """
    Parameters
    ----------
    layers : int
    activation : str
    dropout_rate : float
    batchnorm: boolean
    """
    net = []
    for i in range(1, len(layers)-1):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU(inplace=True))
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if dropout_rate != 0.0:
            net.append(nn.Dropout(dropout_rate))
        if batchnorm:
            net.append(nn.BatchNorm1d(layers[i]))
    net.append(nn.Linear(layers[-2], layers[-1]))
    return nn.Sequential(*net)