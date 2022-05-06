import torch


class HarmonicBias(torch.nn.Module):
    """
    A harmonic potential k_x / 2 (x - x_0)^2 + k_y / 2 (y - y_0)^2 as a static compute graph.
    The potential is only applied to the x-coordinate of the particle.
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
