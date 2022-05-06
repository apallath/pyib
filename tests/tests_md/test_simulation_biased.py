"""
Test TorchForce biases on SlipBondPotential2D landscape.

Note:
    Tests are self-explanatory and are hence not documented.
"""
import inspect
import os
import re
import sys

import numpy as np
import torch
from openmmtorch import TorchForce

from pyib.md.potentials import SlipBondPotential2D
from pyib.md.simulation import SingleParticleSimulation
from pyib.md.visualization import VisualizePotential2D
from pyib.ml.biases import HarmonicBias


def test_sim_SlipBondPotential2D():
    # Initialize potential
    pot = SlipBondPotential2D()

    # Set temperature
    temp = 300

    # Bias
    harmonic_bias_model = HarmonicBias(50, 1, 50, 1)
    torch.jit.script(harmonic_bias_model).save("./tmp/harmonic_bias.pt")
    harmonic_bias = TorchForce("./tmp/harmonic_bias.pt")

    # Initialize simulation
    sim = SingleParticleSimulation(pot,
                                   init_coord=np.array([[-5, -4, 0]]),
                                   traj_in_mem=True,
                                   bias=harmonic_bias)

    if not os.path.exists("./tmp/slip_bond_biased/"):
        os.makedirs("./tmp/slip_bond_biased/")

    # Run simulation
    sim(nsteps=20000,
        chkevery=5000,
        trajevery=20,
        energyevery=20,
        chkfile="./tmp/slip_bond_biased/chk_state.pkl",
        trajfile="./tmp/slip_bond_biased/traj.dat",
        energyfile="./tmp/slip_bond_biased/energies.dat")

    # Visualize
    vis = VisualizePotential2D(pot, temp=temp,
                               xrange=[-12, 15], yrange=[-6, 8],
                               contourvals=10, clip=15)

    vis.scatter_traj(sim.traj, "./tmp/slip_bond_biased/traj.png")
    vis.scatter_traj_projection_x(sim.traj, "./tmp/slip_bond_biased/traj_x.png")
    vis.animate_traj(sim.traj, "./tmp/slip_bond_biased/traj_movie", every=50)
    vis.animate_traj_projection_x(sim.traj, "./tmp/slip_bond_biased/traj_movie", every=50)
