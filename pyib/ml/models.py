import numpy as np
import torch.nn as nn
import torch

from .layers import NonLinear

class Autoencoder(nn.Module):
    """
    Encoderdimensions(list) : Encoder dimensions inputted as (Input_d, dim1, dim2 ,....)
    hidden_dim(int)         : The dimension of the RC 
    Decoderdimensions(list) : Decoder dimensions inputted as (dim1, dim2, .., Output_d)
    """
    def __init__(self, Encoderdimensions:list, hidden_dim:int, Decoderdimensions:list, activation=nn.ReLU()):
        super().__init__()
        self.Encoderdimensions = Encoderdimensions
        self.Decoderdimensions = Decoderdimensions
        self.hidden_dim        = hidden_dim
        self.Nencoder          = len(self.Encoderdimensions)
        self.Ndecoder          = len(self.Decoderdimensions)
        self.activation        = activation
        assert self.Nencoder >=1  , "Number of layers must be larger or equal to 1 as it consist of (intput_dim, dim1,..)"
        assert self.Ndecoder >=1  , "Number of layers must be larger or equal to 1 as it consist of (dim1, dim2, ... ouput_dim"

        self.encoder = self._encoder_init()
        self.decoder = self._decoder_init()

    def _encoder_init(self):
        """
        Initialization of the encoder 
        """
        layers = []
        for i in range(self.Nencoder-1):
            layers.append(NonLinear(self.Encoderdimensions[i], self.Encoderdimensions[i+1], bias=True, activation=self.activation))
        layers.append(NonLinear(self.Encoderdimensions[-1], self.hidden_dim, bias=True, activation=None))
        
        return nn.Sequential(*layers)

    def _decoder_init(self):
        """
        Initialization of the decoder
        """
        tempDim = [self.hidden_dim] + list(self.Decoderdimensions)
        # We know N>=2
        N = len(tempDim)

        layers = []
        for i in range(N-2):
            layers.append(NonLinear(tempDim[i], tempDim[i+1], bias=True, activation=self.activation))
        layers.append(NonLinear(tempDim[-2], tempDim[-1], bias=True, activation=None))
        
        return nn.Sequential(*layers)

    def forward(self, X):
        """
        Input:
        -----
            X(torch.tensor)     : A (N,d) array that contains the d-dimensional input data 
        
        Output:
        -------
            y(torch.tensor)     : A (N,d) array that contains the d-dimensional output data 
        """
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)

        return decoded

    def evaluate(self, X):
        """
        Evaluate returns the decoded output

        Input:
        ------
            X(torch.tensor)     : A(N,d) array that contains the d-dimensional input data

        Output:
        -------
            decoded(torch.tensor): A (N,d) array that contains the d-dimensional output data 
        """
        with torch.no_grad():
            encoded = self.encoder(X)
            decoded = self.decoder(encoded)

            return decoded
    
    def evaluate_hidden_dim(self, X):
        with torch.no_grad():
            encoded = self.encoder(X)

            return encoded
        

class VAE(nn.modules):
    def __init__(self, Encoderdimensions:list, hidden_dim:int, Decoderdimensions:list, activation=nn.ReLU(), prior='VampPrior'):
        super().__init__()
        self.Encoderdimensions = Encoderdimensions
        self.Decoderdimensions = Decoderdimensions
        self.hidden_dim        = hidden_dim
        self.Nencoder          = len(self.Encoderdimensions)
        self.Ndecoder          = len(self.Decoderdimensions)
        self.activation        = activation
        self.prior             = prior
        assert self.Nencoder >=1  , "Number of layers must be larger or equal to 1 as it consist of (intput_dim, dim1,..)"
        assert self.Ndecoder >=1  , "Number of layers must be larger or equal to 1 as it consist of (dim1, dim2, ... ouput_dim"

        self.encoder = self._encoder_init()
        self.decoder = self._decoder_init()

    def _encoder_init(self):
        """
        Initialization of the encoder 
        """
        layers = []
        for i in range(self.Nencoder-1):
            layers.append(NonLinear(self.Encoderdimensions[i], self.Encoderdimensions[i+1], bias=True, activation=self.activation))
        layers.append(NonLinear(self.Encoderdimensions[-1], self.hidden_dim, bias=True, activation=None))
        
        return nn.Sequential(*layers)

    def _decoder_init(self):
        """
        Initialization of the decoder
        """
        tempDim = [self.hidden_dim] + list(self.Decoderdimensions)
        # We know N>=2
        N = len(tempDim)

        layers = []
        for i in range(N-2):
            layers.append(NonLinear(tempDim[i], tempDim[i+1], bias=True, activation=self.activation))
        layers.append(NonLinear(tempDim[-2], tempDim[-1], bias=True, activation=None))
        
        return nn.Sequential(*layers)