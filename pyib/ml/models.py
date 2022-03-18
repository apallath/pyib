import numpy as np
import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    """
    Args:
        Encoderdimensions(list) : Encoder dimensions inputted as (Input_d, dim1, dim2 ,....)
        hidden_dim(int)         : The dimension of the RC 
        Decoderdimensions(list) : Decoder dimensions inputted as (dim1, dim2, .., Output_d)


    Attributes:
        Encoderdimensions(list) : Encoder dimensions inputted as (Input_d, dim1, dim2 ,....)
        hidden_dim(int)         : The dimension of the RC 
        Decoderdimensions(list) : Decoder dimensions inputted as (dim1, dim2, .., Output_d)
    """
    def __init__(self, Encoderdimensions:list, hidden_dim:int, Decoderdimensions:list):
        super().__init__()
        self.Encoderdimensions = Encoderdimensions
        self.Decoderdimensions = Decoderdimensions
        self.hidden_dim        = hidden_dim
        self.Nencoder          = len(self.Encoderdimensions)
        self.Ndecoder          = len(self.Decoderdimensions)
        assert self.Nencoder >=1  , "Number of layers must be larger or equal to 1 as it consist of (intput_dim, dim1,..)"
        assert self.Ndecoder >=1  , "Number of layers must be larger or equal to 1 as it consist of (dim1, dim2, ... ouput_dim"

        self.encoder = self._encoder_init()
        self.decoder = self._decoder_init()

    def _encoder_init(self):
        """
        Initialization of the encoder 

        Returns:
            layers(torch.nn.Sequential)   : Sequential model for the encoder  
        """
        layers = []
        for i in range(self.Nencoder-1):
            layers.append(nn.Linear(self.Encoderdimensions[i], self.Encoderdimensions[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.Encoderdimensions[-1], self.hidden_dim))
        
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
            layers.append(nn.Linear(tempDim[i], tempDim[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(tempDim[-2], tempDim[-1]))
        
        return nn.Sequential(*layers)

    def forward(self, X):
        """
        Args:
            X(torch.tensor)     : A (N,d) array that contains the d-dimensional input data 
        
        Returns:
            y(torch.tensor)     : A (N,d) array that contains the d-dimensional output data 
        """
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)

        return decoded

    def evaluate(self, X):
        """
        Evaluate returns the decoded output

        Args:
            X(torch.tensor)     : A(N,d) array that contains the d-dimensional input data

        Returns:
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

class VAE(nn.Module):
    def __init__(self, Encoderdimensions:list, hidden_dim:int, Decoderdimensions:list, device="cpu"):
        super().__init__()
        self.Encoderdimensions = Encoderdimensions
        self.Decoderdimensions = Decoderdimensions
        self.hidden_dim        = hidden_dim
        self.Nencoder          = len(self.Encoderdimensions)
        self.Ndecoder          = len(self.Decoderdimensions)
        self.device            = device

        assert self.Nencoder >=1  , "Number of layers must be larger or equal to 1 as it consist of (intput_dim, dim1,..)"
        assert self.Ndecoder >=1  , "Number of layers must be larger or equal to 1 as it consist of (dim1, dim2, ... ouput_dim"

        self.encoder, self.meanLayer, self.logvarLayer = self._encoder_init()
        self.decoder = self._decoder_init()

    def _encoder_init(self):
        """
        Initialization of the encoder 
        """
        layers = []
        MeanLayer   = []
        logVarLayer = []
        for i in range(self.Nencoder-1):
            layers.append(nn.Linear(self.Encoderdimensions[i], self.Encoderdimensions[i+1]))
            layers.append(nn.ReLU())
        
        MeanLayer.append(self.Encoderdimensions[-1], self.hidden_dim)
        logVarLayer.append(self.Encoderdimensions[-1], self.hidden_dim)

        return nn.Sequential(*layers), MeanLayer, logVarLayer

    def _decoder_init(self):
        """
        Initialization of the decoder
        """
        tempDim = [self.hidden_dim] + list(self.Decoderdimensions)
        # We know N>=2
        N = len(tempDim)

        layers = []
        for i in range(N-2):
            layers.append(nn.Linear(tempDim[i], tempDim[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(tempDim[-2], tempDim[-1]))
        
        return nn.Sequential(*layers)
    
    def forward(self, X):
        encoded = self.encoder(X)

        # mu and logvar should both be of shape (N,hidden_dim)
        mu      = self.meanLayer(encoded)
        logvar  = self.logvarLayer(encoded)

        z       = torch.exp(0.5 * logvar) * torch.rand_like(mu) + mu

        output  = self.decoder(z)

        return output