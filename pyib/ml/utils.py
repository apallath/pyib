from tokenize import Binnumber
import torch
import numpy as np

from pyib.ml.models import SPIB
from scipy.stats import binned_statistic_dd
from scipy.special import logsumexp

def prep_data_PIB(file:str, dt:int):
    """
    Function that preps the data for PIB reading from a file, the file will have the following format 
    ' # time x1 x2 x3 ... '
    
    where the xi are the CVs collected from the simulation

    Input:
    -----
        file(str)       : The input .dat file as a str --> could be a npy file as well
        dt(int)         : The time step lag 
    """
    def parseNPY(file:str):
        data = np.load(file)

        return data

    def parseDAT(file:str):
        f = open(file, "r")
        lines = np.array([[float(f) for f in line.nstrip("\n").split()] for line in f.readlines()])

        return lines

    # register the functions into a dictionary
    loadDic_   = {
                "npy" : parseNPY,
                "dat" : parseDAT
    }

    # automatically read the type of files
    file_split = file.split(".")
    filetype   = file_split[1]

    # load the parsing function
    parseFunc  = loadDic_[filetype]
    parsedData = parseFunc(file)[:,1:]

    # Parsed data will be [time, x1, x2, x3, ... , ]
    # We will make an X that has [[x1(t),x2(t),x3(t), ..],...] and y that has [[x1(t+dt), x2(t+dt), x3(t+dt), ...],...]
    X = torch.tensor(parsedData[:,:-dt].astype(np.float32))
    y = torch.tensor(parsedData[:,dt:].astype(np.float32))

    assert len(X) == len(y) , "The length of X and y do not agree, something is wrong with the code."

    return X, y

def normalize_data(X:torch.tensor, axis=None):
    """
    Normalize the data along dimension dim
    """
    if axis is not None:
        X_norm = (X - X.mean(axis=axis))/X.std(axis=axis)
    else:
        X_norm = (X - X.mean())/X.std()
    
    return X_norm


def bin_FE(inputs:np.ndarray, bins:int):
    """
    Function that bins the d-dimensional input and finds its free energy by performing -ln(Pi). This function also returns a weight for each of the data points 
    of ln(Pi)
    Args:
    ----
        inputs(np.ndarray)    : Torch.tensor of the shape (N,d)
    """
    # Obtain the number N, d
    d = inputs.shape[1]
    N = inputs.shape[0]

    # find the histogram as well as pi
    histogram, _, binnumber= binned_statistic_dd(inputs, np.arange(N), statistic="count", bins=bins, expand_binnumbers=True)
    linearIndex = np.ravel_multi_index(binnumber-1, histogram.shape)

    # find the probability Pi
    Pi   = histogram / N

    with np.errstate(divide='ignore'):
        FreeEnergy = - np.log(Pi) 
        minFE = FreeEnergy[~np.isnan(FreeEnergy)].min()
        FreeEnergy -= minFE

    # assign a weight to each of the points that corresponds to ln(pi)
    weights = Pi.flatten()[linearIndex]
    
    return FreeEnergy, weights


def projectFEToRC(model: SPIB, inputs: torch.tensor, bins=50):
    """
    Function that projects the free energy onto the reaction coordinate

    Args:
    ----
        model(SPIB)     : An SPIB model, can be called by using model.evaluate(X)
        inputs(torch.tensor)    : The inputs passed in shape (N,d) 
        potential(Potential2D)  : Potential2D object that can calculate the potential 
    """
    N = inputs.shape[0]

    # Obtain the binned Free Energy and weights(log-likelihood) of each of the points 
    FE, weights = bin_FE(inputs.detach().numpy(), bins)

    # mu is of shape (N, d_hidden)
    _, mu, _, _ = model.evaluate(inputs, to_numpy=True)
    d_hidden = mu.shape[1]

    # bin along mu 
    hist,mu_hist_x,binnumber = binned_statistic_dd(mu, np.arange(len(mu)), statistic="count", bins=bins, expand_binnumbers=True)
    if d_hidden==1:
        binnumber = binnumber.reshape(d_hidden,-1)
    linearIndex = np.ravel_multi_index(binnumber-1, hist.shape)
    FE_mu = np.zeros((np.prod(hist.shape),))

    for i in range(np.prod(hist.shape)):
        index = linearIndex == i
        w = weights[index]

        if len(w) > 0:
            FE_mu[i] = -logsumexp(w)
    nonLinearIndex = np.unravel_index(linearIndex, hist.shape)

    return mu_hist_x[0], FE_mu, nonLinearIndex
