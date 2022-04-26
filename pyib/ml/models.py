import torch.nn as nn
import torch

from pyib.ml.layers import NonLinear
from pyib.ml.distributions import log_normal_diag


class SPIB(nn.Module):
    """
    State predictive information bottleneck model.

    encoder_dim (list): List of encoder layer widths, beginning with input layer width.
    hidden_dim (int): Dimension of hidden (latent) variable.
    decoder_dim (list): List of decoder layer widths, ending with output layer width.
    activation: PyTorch activation function to use for each layer (default = nn.ReLU())
    prior: Type of prior to use (options = 'VampPrior' or 'Normal', default = 'VampPrior')
    device: Device to train model on (default = 'cpu')
    representative_dim: Representative dimension (default = None)
    restrictLogVar: If true, restricts logVar value (default = False).
    logVar_min: If restrictLogVar is true, restricts logVar value from [logVar_min, 0] (default = -10).
    """
    def __init__(self,
                 encoder_dim: list,
                 hidden_dim: int,
                 decoder_dim: list,
                 activation=nn.ReLU(),
                 prior='VampPrior',
                 device="cpu",
                 representative_dim=None,
                 restrictLogVar=False,
                 logVar_min=-10):

        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.hidden_dim = hidden_dim

        self.Nencoder = len(self.encoder_dim)
        self.Ndecoder = len(self.decoder_dim)

        # Check that encoder and decoder have at least one layer
        assert self.Nencoder >= 1, "Number of layers must be >= 1 as it contains the input layer."
        assert self.Ndecoder >= 1, "Number of layers must be >= 1 as it contains the output layer."

        self.activation = activation

        # Name of the prior: either "VampPrior" or "Normal"
        self.prior = prior

        # Get input and output layer dimensions from encoder and decoder dimensions
        self.input_dim = self.encoder_dim[0]
        self.output_dim = self.decoder_dim[-1]

        # Set the representative dimension, initially to output_dim
        if representative_dim is None:
            self.representative_dim = self.output_dim
        else:
            self.representative_dim = representative_dim

        # Restrict logVar
        self.restrictLogVar = restrictLogVar
        self.logVar_min = logVar_min

        # Device, for torch
        self.device = device

        # Initialize encoder and decoder
        self.encoder, self.encoderMean, self.encoderLogVar = self._encoder_init()
        self.decoder = self._decoder_init()
        self._representativeInputs_init()

    def _encoder_init(self):
        """
        Initializes encoder
        """
        encoder = []
        mean_layer = []
        logVar_layer = []

        # Construct encoder layers
        for i in range(self.Nencoder - 1):
            encoder.append(NonLinear(self.encoder_dim[i], self.encoder_dim[i + 1], bias=True, activation=self.activation))

        # Contruct mean and logvar layers
        mean_layer = NonLinear(self.encoder_dim[-1], self.hidden_dim, bias=True, activation=None)

        if self.restrictLogVar:
            logVar_layer = nn.Sequential(NonLinear(self.encoder_dim[-1], self.hidden_dim, bias=True, activation=None), nn.Sigmoid())
        else:
            logVar_layer = NonLinear(self.encoder_dim[-1], self.hidden_dim, bias=True, activation=None)

        return nn.Sequential(*encoder), mean_layer, logVar_layer

    def _decoder_init(self):
        """
        Initializes decoder
        """
        tempDim = [self.hidden_dim] + list(self.decoder_dim)
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

        Args:
            X(torch.tensor) : Shape (N, input_dim)
            labels  : Shape (N,1)
        """
        assert X.shape[1] == self.input_dim, "dimension of X is incorrect, it needs to be {}".format(self.input_dim)
        self.__pseudo_input = torch.zeros(self.representative_dim, self.input_dim, requires_grad=False, device=self.device)

        # For each of the initial guess label, we find the mean among the x to use as initial uk
        # e.g. if label == 1, we take all the points which we initial guessed as 1, {xi,yi} --> take the mean {xmean, ymean}
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

            index_one_hot = torch.nn.functional.one_hot(index.flatten(), num_classes=self.output_dim)

        if to_numpy:
            index , mean, logVar, decoder_output, index_one_hot = index.detach().numpy(), \
                                                    mean.detach().numpy(), \
                                                    logVar.detach().numpy(), \
                                                    decoder_output.detach().numpy(), \
                                                    index_one_hot.detach().numpy

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
        log_p = log_normal_diag(z,mean, logvar,dim=1, keepdims=True)

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
        log_p  = log_normal_diag(z_expand, representative_mean, representative_logvar, dim=2)

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
