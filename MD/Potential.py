from openmm.app import *
from openmm import *
import numpy as np

class DoubleWellPotential(CustomExternalForce):
    def __init__(self,*kwargs):
        force = "(x^2-1)^2 + y^2"
        super().__init__(force)

    @staticmethod 
    def Potential(pos:np.ndarray):
        """
        This is a class method that evaluates the potential at position provided by the user (pos)

        Args:
            pos(np.ndarray)     : (N,2) array that contains information about the positions at where the user wants potential calculated
        """
        if len(pos.shape) == 1:
            assert pos.shape[0] == 2, "Dimension must be 2 while it is {}".format(pos.shape[0])

        if len(pos.shape) == 2:
            assert pos.shape[1] == 2, "Dimension must be 2 while it is {}".format(pos.shape[1])

        return (pos[:,0] ** 2- 1) **2 + pos[:,1] ** 2


        