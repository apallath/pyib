import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################################
# Biases that act directly on the (x, y) coordinates of a particle
################################################################################


class HarmonicBias(torch.nn.Module):
    """
    A harmonic potential k_x / 2 (x - x_0)^2 + k_y / 2 (y - y_0)^2 as a static compute graph.
    The potential is only applied to the x-coordinate of the particle.

    Attributes:
        k_x (float)
        x_0 (float)
        k_y (float)
        y_0 (float)
    """
    def __init__(self, k_x, x_0, k_y, y_0):
        self.k_x = k_x
        self.x_0 = x_0
        self.k_y = k_y
        self.y_0 = y_0
        super().__init__()

    def forward(self, positions):
        """The forward method returns the energy computed from positions.
        Args:
            positions : torch.Tensor with shape (1, 3)
                positions[0, k] is the position (in nanometers) of spatial dimension k of particle 0
        Returns:
            potential : torch.Scalar
                The potential energy (in kJ/mol)
        """
        return self.k_x / 2 * torch.sum((positions[:, 0] - self.x_0) ** 2) + \
            self.k_y / 2 * torch.sum((positions[:, 1] - self.y_0) ** 2)


class LegendreBias1D(torch.nn.Module):
    """
    Legendre polynomial bias along x- or y- coordinate, with user-defined degree.

    Attributes:
        degree (int)
        axis (str): 'x' or 'y' (default='x')
    """
    def __init__(self, degree, axis='x'):
        self.degree = degree
        self.weights = nn.parameter.Parameter(torch.randn(degree + 1))
        self.axis = axis

    @classmethod
    def legendre_polynomial(cls, x, degree):
        r"""
        Computes a legendre polynomial of degree $n$ using dynamic programming
        and the Bonnet's recursion formula:

        $$(n + 1) P_{n+1}(x) = (2n + 1) x P_n(x) - nP_{n-1}(x)$$
        """
        if degree == 0:
            return torch.ones(x.size(0), requires_grad=True).type(x.type())

        elif degree == 1:
            return x

        elif degree > 1:
            P_n_minus = torch.ones(x.size(0), requires_grad=True).type(x.type())
            P_n = x

            for n in range(1, degree):
                P_n_plus = ((2 * n + 1) * x * P_n - n * P_n_minus) / (n + 1)

                # Replace
                P_n_minus = P_n
                P_n = P_n_plus

        return P_n

    def forward(self, positions):
        """The forward method returns the energy computed from positions.

        Args:
            positions : torch.Tensor with shape (1, 3)
                positions[0, k] is the position (in nanometers) of spatial dimension k of particle 0

        Returns:
            potential : torch.Scalar
                The potential energy (in kJ/mol)
        """
        # Extract coordinate
        if self.axis == 'x':
            x = positions[:, 0]
        elif self.axis == 'y':
            x = positions[:, 1]
        else:
            raise ValueError("Invalid axis")

        # Apply legendre expansion bias
        bias = torch.zeros_like(x)
        for i in range(self.degree):
            bias += self.weights[i] * self.legendre_polynomial(x, i)

        return bias
