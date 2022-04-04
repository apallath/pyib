import torch 
import numpy as np
import torch.nn as nn

def log_Normal_diag(x, log_var, mean, dim=None):
    """
    Function that calculates the log-likelihood for a diagonal multivariate gaussian distribution
    """
    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2)/ torch.exp(log_var))

    return torch.sum(log_normal, dim=dim)
