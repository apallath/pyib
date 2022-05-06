import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import legendre
import torch

from pyib.ml.biases import LegendreBias1D

################################################################################
# Legendre polynomial tests
################################################################################


def test_legendre_polynomials():
    """
    Benchmarks ves.basis.LegendreBasis_x against scipy.special.legendre.
    """
    fig, ax = plt.subplots(dpi=200)

    for degree in range(5):
        x = torch.linspace(-1, 1, 1000)
        lbias = LegendreBias1D.legendre_polynomial(x, degree).detach().cpu().numpy()
        lsci = legendre(degree)(x.cpu().numpy())

        assert(np.allclose(lbias, lsci))
        ax.plot(x.cpu().numpy(), lsci)
        ax.plot(x.cpu().numpy(), lbias, '--')

    if not os.path.exists("./tmp/legendre/"):
        os.makedirs("./tmp/legendre/")
    plt.savefig("./tmp/legendre/legendre_compare.png")


def test_grad_legendre_polynomials():
    """
    Benchmarks gradients (derivatives) of ves.basis.LegendreBasis_x against those of scipy.special.legendre.
    """
    fig, ax = plt.subplots(dpi=200)

    for degree in range(1, 5):
        x = torch.linspace(-1, 1, 1000, requires_grad=True)

        lbias = LegendreBias1D.legendre_polynomial(x, degree)
        lbias_grad = torch.autograd.grad(lbias, x, grad_outputs=torch.ones_like(lbias), create_graph=True, allow_unused=True)[0].detach().cpu().numpy()

        lsci = legendre(degree)
        lsci_grad = lsci.deriv()(x.detach().cpu().numpy())

        assert(np.allclose(lbias_grad, lsci_grad))

        ax.plot(x.detach().cpu().numpy(), lsci_grad)
        ax.plot(x.detach().cpu().numpy(), lbias_grad, '--')

    if not os.path.exists("./tmp/legendre/"):
        os.makedirs("./tmp/legendre/")
    plt.savefig("./tmp/legendre/grad_legendre_compare.png")
