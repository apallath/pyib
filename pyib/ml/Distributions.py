import torch 
import numpy as np
import torch.nn as nn

def log_Normal_diag(x, mean, log_var, dim=None):
    """
    Function that calculates the log-likelihood for a diagonal multivariate gaussian distribution
    """
    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2)/ torch.exp(log_var))

    if dim is not None:
        sum_ = torch.sum(log_normal, dim=dim)
    else:
        sum_ = log_normal

    return sum_
