"""
Classes defining potential energy surfaces.
"""
import numpy as np

from openmm import openmm


class Potential2D(openmm.CustomExternalForce):
    """
    Abstract class defining basic 2D potential behavior.

    A harmonic restraining potential of magnitude 1000 kJ/mol is applied on the
    z coordinates about z=0.

    Note:
        Child classes must call super.__init__() only after initializing the force
        attribute in x and y variables.

    Attributes:
        force (str): `OpenMM-compatible custom force expression`_.

    .. _OpenMM-compatible custom force expression:
       http://docs.openmm.org/latest/userguide/theory/03_custom_forces.html#writing-custom-expressions
    """
    def __init__(self):
        # Apply restraining potential along z direction
        # Child classes will add terms for x and y and initialize this force expression
        self.force += " + 1000 * z^2"

        # Print force expression
        print("[Potential] Initializing potential with expression:\n" + self.force)

        # Initialize force expression
        super().__init__(self.force)

    def potential(self, x: float, y: float):
        """
        Computes the potential at a given point (x, y).

        Args:
            x (float): x-coordinate of the point to compute potential at.
            y (float): y-coordinate of the point to compute potential at.

        Returns:
            V (float): Value of potential at (x, y).
        """
        # Child classes will implement this method.
        raise NotImplementedError()


class SlipBondPotential2D(Potential2D):
    pass


class CatchBondPotential2D(Potential2D):
    pass


class SzaboBerezhkovskiiPotential(Potential2D):
    """
    2D Szabo-Berezhkovskii potential.
    """

    x0 = 2.2
    omega2 = 4.0
    Omega2 = 1.01 * omega2
    Delta = omega2 * x0 ** 2 / 4.0

    def __init__(self, x0 = 2.2, omega2 = 4.0):
        # Look up Szabo-Berezhkovskii potential formula for details
        constvals = {"x0": self.x0,
                     "omega2": self.omega2,
                     "Omega2": self.Omega2,
                     "Delta": self.Delta}

        self.force = '''{Omega2} * 0.5 * (x - y)^2'''.format(**constvals)
        self.force += ''' + (select(step(x + 0.5 * {x0}), select(step(x - 0.5 * {x0}), -{Delta} + {omega2} * 0.5 * (x - {x0})^2, -{omega2} * 0.5 * x^2), -{Delta} + {omega2} * 0.5 * (x + {x0})^2))'''.format(**constvals)

        super().__init__()

    def potential(self, x, y):
        """Computes the Szabo-Berezhkovskii potential at a given point (x, y)."""
        Ux = np.piecewise(x,
                          [x <= -self.x0 / 2,
                           np.logical_and(x > -self.x0 / 2, x < self.x0 / 2),
                           x >= self.x0 / 2],
                          [lambda x: -self.Delta + self.omega2 * (x + self.x0) ** 2 / 2.0,
                           lambda x: -self.omega2 * x ** 2 / 2.0,
                           lambda x: -self.Delta + self.omega2 * (x - self.x0) ** 2 / 2.0])
        return (Ux + self.Omega2 * (x - y) ** 2 / 2.0)


class MullerBrownPotential(Potential2D):
    """
    2D Muller-Brown potential.
    """

    a = [-1, -1, -6.5, 0.7]
    b = [0, 0, 11, 0.6]
    c = [-10, -10, -6.5, 0.7]
    A = [-200, -100, -170, 15]
    x_bar = [1, 0, -0.5, -1]
    y_bar = [0, 0.5, 1.5, 1]

    def __init__(self):
        for i in range(4):
            fmt = dict(a = self.a[i], b = self.b[i], c = self.c[i], A = self.A[i], x_bar = self.x_bar[i], y_bar = self.y_bar[i])
            if i == 0:
                self.force = '''{A} * exp({a} * (x - {x_bar})^2 + {b} * (x - {x_bar}) * (y - {y_bar}) + {c} * (y - {y_bar})^2)'''.format(**fmt)
            else:
                self.force += ''' + {A} * exp({a} * (x - {x_bar})^2 + {b} * (x - {x_bar}) * (y - {y_bar}) + {c} * (y - {y_bar})^2)'''.format(**fmt)

        super().__init__()

    def potential(self, x, y):
        """Compute the potential at a given point (x, y)."""
        value = 0
        for i in range(4):
            value += self.A[i] * np.exp(self.a[i] * (x - self.x_bar[i])**2 + \
                self.b[i] * (x - self.x_bar[i]) * (y - self.y_bar[i]) + self.c[i] * (y - self.y_bar[i])**2)
        return value
