import numpy as np
import torch.nn as nn
import torch

from .layers import NonLinear
from .Distributions import log_Normal_diag

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

        Returns:
            layers(torch.nn.Sequential)   : Sequential model for the encoder  
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
    """
    Encoderdimensions(list)     : The encoder dimension passed in as a list e.g [2,3]
    """
    def __init__(self, Encoderdimensions:list, hidden_dim:int, Decoderdimensions:list, RepresentativeD:int,\
         activation=nn.ReLU(), prior='VampPrior'):
        super().__init__()
        self.Encoderdimensions = Encoderdimensions
        self.Decoderdimensions = Decoderdimensions
        self.hidden_dim        = hidden_dim
        self.Nencoder          = len(self.Encoderdimensions)
        self.Ndecoder          = len(self.Decoderdimensions)
        self.activation        = activation

        # Name of the prior --> either "VampPrior" or "Normal"
        self.prior             = prior

        # The dimension of the representative inputs for VampPrior
        self._representativeD  = RepresentativeD

        # The first element of encoder dimension is the input dimension
        self.input_dim         = self.Encoderdimensions[0]

        # The last element of the decoder dimension is the output dimension
        self.output_dim        = self.Decoderdimensions[-1]

        assert self.Nencoder >=1  , "Number of layers must be larger or equal to 1 as it consist of (intput_dim, dim1,..)"
        assert self.Ndecoder >=1  , "Number of layers must be larger or equal to 1 as it consist of (dim1, dim2, ... ouput_dim"

        # Initialize encoder and decoder 
        self.encoderMean, self.encoderLogVar = self._encoder_init()
        self.decoder = self._decoder_init()
        self._representativeInputs_init()

    def _encoder_init(self):
        """
        Initialization of the encoder 
        """
        Meanlayers = []
        logVarlayers = []
        for i in range(self.Nencoder-1):
            Meanlayers.append(NonLinear(self.Encoderdimensions[i], self.Encoderdimensions[i+1], bias=True, activation=self.activation))
            logVarlayers.append(NonLinear(self.Encoderdimensions[i], self.Encoderdimensions[i+1], bias=True, activation=self.activation))
        Meanlayers.append(NonLinear(self.Encoderdimensions[-1], self.hidden_dim, bias=True, activation=None))
        logVarlayers.append(NonLinear(self.Encoderdimensions[-1], self.hidden_dim, bias=True, activation=None))

        Meanlayers = nn.Sequential(*Meanlayers)
        logVarlayers = nn.Sequential(*logVarlayers)
        
        return Meanlayers, logVarlayers

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
    
    def _representativeInputs_init(self):
        """
        Function that initializes the representative inputs for the VampPrior
        """
        self.__weights_input = torch.eye((self._representativeD), requires_grad=False)
        self.__pseudo_input  = torch.eye((self._representativeD), requires_grad=False)

        # Softmax is used here to make sure that \sum_{i} w_{i} = 1
        self.__weight_layer  = nn.Sequential(nn.Linear(self._representativeD, 1, bias=False), nn.Softmax(dim=0))
        self.__pseudo_input_layer = NonLinear(self._representativeD, self.input_dim, bias=False,\
             activation=nn.Hardtanh(min_val=0.0, max_val=1.0))
    
    def reparametrize(self, mean, logvar):
        """
        Performs the reparameterization trick where 
            z^{n} = \mu(X^{n}) + \sigma(X^{n})  * \epsilon , where \epsilon ~ N(0,I)
        """
        # Var^{1/2}
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)

        return mean + std * eps, eps
    
    def forward(self,X):
        """
        Args:
            X (torch.tensor)    : Tensor with dimension (N, input_dim)
        
        Returns:
            decoder_output      : The output of the VAE
            mean                : The mean output from the encoder 
            logvar              : The logvar output from the encoder
            z_sample            : The randomly sampled Z from N(0,I)
        """
        mean = self.encoderMean(X)
        logvar = self.encoderLogVar(X)
        decoder_input, z_sample = self.reparametrize(mean, logvar)
        decoder_output = self.decoder(decoder_input)

        return decoder_output, mean, logvar, z_sample
    
    def evaluate(self, X):
        """
        """
        with torch.no_grad():
            decoder_output, mean, logVar, _ = self.forward(X)

            index = torch.argmax(decoder_output, axis=1, keepdim=True)

        return index, mean, logVar
    
    def log_pz(self,z, mean, logvar):
        """
        Function that calculates the log of the encoder output p(z|X) ~ N(mu(X), std(X))
        where mu(X) and std(X) are the encoder NN
        """
        # input z (N, hidden_dim)
        # output log_p (N,1)
        log_p = log_Normal_diag(z,mean, logvar)

        return log_p

    def log_rz(self,z):
        """
        Function that calculates the log of the prior distribution from VampPrior

        Args:
            z (torch.tensor)    : Tensor passed in with shape (N, hidden_dim)
        """
        # U should be of shape (Reprensetative_D, input_dim)
        # U notation follows the paper 
        U = self.__pseudo_input_layer(self.__pseudo_input)

        # shape (Representative_D, hidden_dim)
        representative_Z_mean = self.encoderMean(U)
        representative_Z_logvar = self.encoderLogVar(U)

        # Shape (1, Representative_D, hidden_dim)
        representative_Z_mean = representative_Z_mean.unsqueeze(0)
        representative_Z_logvar = representative_Z_logvar.unsqueeze(0)

        # expand z  (N,1, hidden_dim)
        z_expand = z.unsqueeze(1)

        # Find the log likelihood --> (N, Representative_D)
        log_p  = log_Normal_diag(z_expand, representative_Z_mean, representative_Z_logvar, dim=2)

        # Obtain the weights w --> which we will then dot with log_p
        # shape (Representative_D, 1)
        w = self.__weight_layer(self.__weights_input)

        # Shape (N, 1)
        log_p  = torch.log(torch.exp(log_p) @ w + 1e-10)

        return log_p
    
    def update_Labels(self, X):
        """
        Update the labels for the following X
        """
        with torch.no_grad():
            # shape (N, d2)
            decoder_output, _, _, _ = self.forward(X)

            # argmax 
            index  = torch.argmax(decoder_output, dim=1).flatten()
        
        return index
