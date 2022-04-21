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
    def __init__(self, Encoderdimensions:list, hidden_dim:int, Decoderdimensions:list, 
         activation=nn.ReLU(), prior='VampPrior', device="cpu", representative_dim=None, restrictLogVar=True):
        super().__init__()
        self.Encoderdimensions = Encoderdimensions
        self.Decoderdimensions = Decoderdimensions
        self.hidden_dim        = hidden_dim
        self.Nencoder          = len(self.Encoderdimensions)
        self.Ndecoder          = len(self.Decoderdimensions)
        self.activation        = activation

        # Name of the prior --> either "VampPrior" or "Normal"
        self.prior             = prior

        # The first element of encoder dimension is the input dimension
        self.input_dim         = self.Encoderdimensions[0]

        # The last element of the decoder dimension is the output dimension
        self.output_dim        = self.Decoderdimensions[-1]

        # set the representative dimension, initially set to output_dim
        if representative_dim is None:
            self.representative_dim = self.output_dim
        else:
            self.representative_dim = representative_dim

        # whether or not we are restricting logVar to within range [-10,0] --> could add option to change this too
        self.restrictLogVar = restrictLogVar

        # device name 
        self.device = device

        assert self.Nencoder >=1  , "Number of layers must be larger or equal to 1 as it consist of (intput_dim, dim1,..)"
        assert self.Ndecoder >=1  , "Number of layers must be larger or equal to 1 as it consist of (dim1, dim2, ... ouput_dim"

        # Initialize encoder and decoder 
        self.encoder , self.encoderMean, self.encoderLogVar = self._encoder_init()
        self.decoder = self._decoder_init()
        self._representativeInputs_init()

    def _encoder_init(self):
        """
        Initialization of the encoder 
        """
        encoder = []
        Meanlayers = []
        logVarlayers = []
        for i in range(self.Nencoder-1):
            encoder.append(NonLinear(self.Encoderdimensions[i], self.Encoderdimensions[i+1], bias=True, activation=self.activation))

        Meanlayers = NonLinear(self.Encoderdimensions[-1], self.hidden_dim, bias=True, activation=None)

        if self.restrictLogVar:
            logVarlayers = nn.Sequential(NonLinear(self.Encoderdimensions[-1], self.hidden_dim, bias=True, activation=None), \
                nn.Sigmoid())
        else:
            logVarlayers = NonLinear(self.Encoderdimensions[-1], self.hidden_dim, bias=True, activation=None)
        
        return nn.Sequential(*encoder), Meanlayers, logVarlayers

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
        # Here, output_dim is the number of labels for the metastable states 

        # This is of shape (output_dim, output_dim)
        # By passing through a network of (output_dim, 1) --> (output_dim, 1) 
        # These corresponds to uk, where k = i, ..., output_dim
        self.__weights_input = torch.eye(self.representative_dim, m=self.representative_dim, requires_grad=False, device=self.device)

        # This is of shape (output_dim, input_dim), which then by passing into encoder NN will give 
        # (output_dim, z_dim) --> p(z|uk) where k = i, .., output_dim
        self.__pseudo_input  = torch.eye(self.representative_dim, m=self.input_dim, requires_grad=False, device=self.device)

        # Softmax is used here to make sure that \sum_{i} w_{i} = 1
        self.__weight_layer  = nn.Sequential(nn.Linear(self.output_dim, 1, bias=False), nn.Softmax(dim=0))
    
    def RepresentativeInputs_init(self, X:torch.tensor, labels:torch.tensor):
        """
        Function used by user to overwrite the bad initial guess for representative input

        X(torch.tensor) : Shape (N, input_dim)
        labels  : Shape (N,1)
        """
        assert X.shape[1] == self.input_dim, "dimension of X is incorrect, it needs to be {}".format(self.input_dim)
        self.__pseudo_input = torch.zeros(self.representative_dim, self.input_dim, requires_grad=False, device=self.device)

        for i in range(self.representative_dim):
            index = (labels==i)
            self.__pseudo_input[i] = X[index,:].mean(axis=0) 

    def reparametrize(self, mean, logvar):
        """
        Performs the reparameterization trick where 
            z^{n} = \mu(X^{n}) + \sigma(X^{n})  * \epsilon , where \epsilon ~ N(0,I)
        """
        # Var^{1/2}
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std, device=self.device)

        return mean + std * eps
    
    def _encode(self, X):
        """
        Function for encoder

        Return:
            mean(torch.tensor)
            logVar(torch.tensor)
        """
        X = self.encoder(X)

        mean = self.encoderMean(X)
        if self.restrictLogVar:
            logvar = -10*self.encoderLogVar(X)
        else:
            logvar = self.encoderLogVar(X)
        
        return mean, logvar
    
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
        mean, logvar = self._encode(X)

        z_sampled = self.reparametrize(mean, logvar)
        decoder_output = self.decoder(z_sampled)

        return decoder_output, mean, logvar, z_sampled
    
    def evaluate(self, X, to_numpy=True):
        """
        Function that evaluates the output for any data point

        Args:
            X(torch.tensor) : Input data with shape (N,input_dim)
        """
        assert len(X.shape) == 2 , "Input must be 2-dimensional torch.tensor but the user input is dim = {}".format(len(X.shape))
        assert X.shape[1] == self.input_dim , "Input data must have the same dimension as the required input dimension = {}".format(self.input_dim)

        with torch.no_grad():
            decoder_output, mean, logVar, _ = self.forward(X)

            index = torch.argmax(decoder_output, axis=1, keepdim=True)

        if to_numpy:
            index , mean, logVar, decoder_output = index.detach().numpy(), \
                                                    mean.detach().numpy(), \
                                                    logVar.detach().numpy(), \
                                                    decoder_output.detach().numpy()

        return index, mean, logVar, decoder_output
    
    def log_pz(self,z, mean, logvar):
        """
        Function that calculates the log of the encoder output p(z|X) ~ N(mu(X), std(X))
        where mu(X) and std(X) are the encoder NN

        Args:
            z(torch.tensor)     : Sampled z from distribution N(z;mean,var), shape (N,d)
            mean(torch.tensor)  : Mean is torch tensor of shape (d)
            logvar(torch.tensor): Log var is torch tensor of shape (d)
        
        Return :
            log_p(z)    : The log likelihood of the data points
        """
        # input z (N, hidden_dim)
        # output log_p (N,1)
        log_p = log_Normal_diag(z,mean, logvar,dim=1, keepdims=True)

        return log_p

    def log_rz(self,z):
        """
        Function that calculates the log of the prior distribution from VampPrior

        Args:
            z (torch.tensor)    : Tensor passed in with shape (N, hidden_dim)
        """
        # uk should be of shape (output_dim, input_dim)
        # where k = i, .., output_dim
        uk = self.__pseudo_input

        # shape (Representative_D, hidden_dim)
        representative_Z_mean, representative_Z_logvar = self._encode(uk)

        # Shape (1, Representative_D, hidden_dim)
        representative_mean = representative_Z_mean.unsqueeze(0)
        representative_logvar = representative_Z_logvar.unsqueeze(0)

        # expand z  (N,1, hidden_dim)
        z_expand = z.unsqueeze(1)

        # Find the log likelihood --> (N, Representative_D)
        log_p  = log_Normal_diag(z_expand, representative_mean, representative_logvar, dim=2)

        # Obtain the weights w --> which we will then dot with log_p
        # shape (Representative_D, 1)
        w = self.__weight_layer(self.__weights_input)

        # Shape (N, 1)
        log_r  = torch.log(torch.exp(log_p)@w + 1e-10)

        return log_r
    
    def get_Labels(self, X):
        """
        Update the labels as well as the state population for the input X

        Args:
            X(torch.tensor) : Torch tensor of the shape (N, input_dim)
        
        Outputs:
            index(torch.tensor) : Return the state number that each of the input data resides in
            state_population(torch.tensor)  : Return the number of data points in each of the states 
        """
        with torch.no_grad():
            # shape (N, d2)
            decoder_output, _, _, _ = self.forward(X)

            # argmax 
            index  = torch.argmax(decoder_output, dim=1).flatten()

            # then get the one hot vector
            one_hot = nn.functional.one_hot(index, num_classes=self.output_dim)

            # get number of data points per state
            state_population = one_hot.sum(axis=0)

            state_population = state_population/state_population.sum()
        
        return index, state_population

    def update_representative_inputs(self, X:torch.tensor):
        """
        Function that updates the representative inputs
        """
        with torch.no_grad():
            # Labels are of shape (N, output_dim)
            # mean is of shape (N, hidden_dim)
            # logVar is of shape (N, hidden_dim)
            labels, z, _ , _ = self.forward(X)

            # argmax along dim1, --> (N,)
            labels = torch.argmax(labels, dim=1)

            # convert labels to one hot vector (N, output_dim)
            one_hot = nn.functional.one_hot(labels,num_classes=self.output_dim)

            # sum along dimension 0 to obtain how many data points are in each of the metastable state
            #  [ [ 1 0 0 ] , [ 0 1 0 ] ... ]
            state_population = one_hot.sum(axis=0)

            # set the new representative inputs
            # append to representative inputs if state_population[i] is not 0
            # This means that representative_inputs which was shape (output_dim, input_dim)
            # could change to some other (Nchange, input_dim)
            representative_inputs = []

            for i in range(self.output_dim):
                if state_population[i] > 0:
                    # among the state i, find the mean z --> then find the x_k that gives a point 
                    # closest to mean_z, call that our representative input

                    # This is the index where of [0,1,1,...] whether or not the point is in state i
                    # shape (N, )
                    index = one_hot[:,i].bool()

                    # find the center z 
                    # shape (1, hidden_dim)
                    center_z = z[index].mean(axis=0).reshape(1,-1)

                    # find the point closest to point z
                    # shape (N,)
                    dist = torch.sqrt(((z - center_z)**2).sum(axis=1))
                    minIndex = torch.argmin(dist)

                    # append the point at minIndex to representative_input
                    representative_inputs.append(X[minIndex].reshape(1,-1))

        representative_inputs = torch.cat(representative_inputs, dim=0)

        # reset representative_dim
        self.representative_dim = representative_inputs.shape[0]

        # reset the representative weight network
        self.__weight_layer = nn.Sequential(nn.Linear(self.representative_dim, 1, bias=False), nn.Softmax(dim=0))

        # reset the weights to be identity mapping initially 
        self.__weight_layer[0].weights = torch.ones((self.representative_dim,1), requires_grad=True, device=self.device)

        self.__weights_input = torch.eye(self.representative_dim, device=self.device, requires_grad=False)

        # reset the pseudo input u_k
        self.__pseudo_input = representative_inputs
    
    def get_psuedo_input(self):
        return self.__pseudo_input
