import numpy as np
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Dimensions(list)    :   list of dimension for encoder inputted as (input_dim, dim1, dim2, ..., RC_dim), 
                            decoder dimension will be assumed to be symmetric
    """
    def __init__(self, dimensions):
        super().__init__()
        self.dimensions = dimensions
        self.N          = len(self.dimensions)
        assert self.N > 1 , "Number of layers must be larger than 1 as it consist of (intput_dim, dim1, .. , RC_dim)"

        self.encoder = self._encoder_init()
        self.decoder = self._decoder_init()

    def _encoder_init(self):
        """
        Initialization of the encoder 
        """
        layers = []
        for i in range(self.N - 1):
            layers.append(nn.Linear(self.dimensions[i], self.dimensions[i+1]))
            layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)

    def _decoder_init(self):
        """
        Initialization of the decoder
        """
        layers = []

        # If the number of layers is larger than 2 --> e.g. 3 layers [10,5,1], then the layer going from 5-->1 does not require a Relu
        # have to be careful with that
        if self.N > 2:
            for i in range(self.N-1, 1,-1):
                layers.append(nn.Linear(self.dimensions[i], self.dimensions[i-1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(self.dimensions[1], self.dimensions[0]))
        else:
            layers.append(nn.Linear(self.dimensions[1] , self.dimensions[0]))
        
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