import torch


def log_normal_diag(x, mean, log_var, dim=None, keepdims=False):
    """
    Calculates the log-likelihood for a diagonal multivariate Gaussian distribution.
    """
    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))

    if dim is not None:
        sum_ = torch.sum(log_normal, dim=dim, keepdims=keepdims)
    else:
        sum_ = log_normal

    return sum_
