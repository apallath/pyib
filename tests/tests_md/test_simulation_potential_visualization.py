"""
Tests for simulation with different potentials.

Note:
    Tests are self-explanatory and are hence not documented.
"""
import inspect
import os
import re
import sys

import numpy as np

from pyib.md.potentials import SlipBondPotential2D, CatchBondPotential2D, SzaboBerezhkovskiiPotential as SBPotential, MullerBrownPotential
from pyib.md.simulation import SingleParticleSimulation
from pyib.md.visualization import VisualizePotential2D


def test_sim_SlipBondPotential2D():
    # Initialize potential
    pot = SlipBondPotential2D()

    # Set temperature
    temp = 300

    # Initialize simulation
    sim = SingleParticleSimulation(pot,
                                   init_coord=np.array([[-5, -4, 0]]),
                                   traj_in_mem=True)

    if not os.path.exists("./tmp/slip_bond/"):
        os.makedirs("./tmp/slip_bond/")

    # Run simulation
    sim(nsteps=20000,
        chkevery=5000,
        trajevery=20,
        energyevery=20,
        chkfile="./tmp/slip_bond/chk_state.pkl",
        trajfile="./tmp/slip_bond/traj.dat",
        energyfile="./tmp/slip_bond/energies.dat")

    # Visualize
    vis = VisualizePotential2D(pot, temp=temp,
                               xrange=[-12, 15], yrange=[-6, 8],
                               contourvals=10, clip=15)

    vis.scatter_traj(sim.traj, "./tmp/slip_bond/traj.png")
    vis.scatter_traj_projection_x(sim.traj, "./tmp/slip_bond/traj_x.png")
    vis.animate_traj(sim.traj, "./tmp/slip_bond/traj_movie", every=50)
    vis.animate_traj_projection_x(sim.traj, "./tmp/slip_bond/traj_movie", every=50)


def test_sim_CatchBondPotential2D():
    # Initialize potential
    pot = CatchBondPotential2D()

    # Set temperature
    temp = 300

    # Initialize simulation
    sim = SingleParticleSimulation(pot,
                                   init_coord=np.array([[-5, -4, 0]]),
                                   traj_in_mem=True)

    if not os.path.exists("./tmp/catch_bond/"):
        os.makedirs("./tmp/catch_bond/")

    # Run simulation
    sim(nsteps=20000,
        chkevery=5000,
        trajevery=20,
        energyevery=20,
        chkfile="./tmp/catch_bond/chk_state.pkl",
        trajfile="./tmp/catch_bond/traj.dat",
        energyfile="./tmp/catch_bond/energies.dat")

    # Visualize
    vis = VisualizePotential2D(pot, temp=temp,
                               xrange=[-12, 15], yrange=[-6, 8],
                               contourvals=10, clip=15)

    vis.scatter_traj(sim.traj, "./tmp/catch_bond/traj.png")
    vis.scatter_traj_projection_x(sim.traj, "./tmp/catch_bond/traj_x.png")
    vis.animate_traj(sim.traj, "./tmp/catch_bond/traj_movie", every=50)
    vis.animate_traj_projection_x(sim.traj, "./tmp/catch_bond/traj_movie", every=50)


def test_vis_SBPotential():
    # Initialize potential
    pot = SBPotential()

    # Set temperature
    temp = 300

    # Initialize simulation
    sim = SingleParticleSimulation(pot,
                                   init_coord=np.array([[-2, -2, 0]]),
                                   traj_in_mem=True)

    if not os.path.exists("./tmp/SB/"):
        os.makedirs("./tmp/SB/")

    # Run simulation
    sim(nsteps=20000,
        chkevery=5000,
        trajevery=20,
        energyevery=20,
        chkfile="./tmp/SB/chk_state.pkl",
        trajfile="./tmp/SB/traj.dat",
        energyfile="./tmp/SB/energies.dat")

    # Visualize
    vis = VisualizePotential2D(pot, temp=temp,
                               xrange=[-7.5, 7.5], yrange=[-7.5, 7.5],
                               contourvals=[-2, -1, 0, 1, 2, 5, 8, 10])

    vis.scatter_traj(sim.traj, "./tmp/SB/traj.png")
    vis.scatter_traj_projection_x(sim.traj, "./tmp/SB/traj_x.png")
    vis.animate_traj(sim.traj, "./tmp/SB/traj_movie", every=50)
    vis.animate_traj_projection_x(sim.traj, "./tmp/SB/traj_movie", every=50)


def test_vis_MullerBrownPotential():
    # Initialize potential
    pot = MullerBrownPotential()

    # Set temperature
    temp = 300

    # Initialize simulation
    sim = SingleParticleSimulation(pot,
                                   init_coord=np.array([[0.6, 0, 0]]),
                                   traj_in_mem=True)

    if not os.path.exists("./tmp/MB/"):
        os.makedirs("./tmp/MB/")

    # Run simulation
    sim(nsteps=20000,
        chkevery=5000,
        trajevery=20,
        energyevery=20,
        chkfile="./tmp/MB/chk_state.pkl",
        trajfile="./tmp/MB/traj.dat",
        energyfile="./tmp/MB/energies.dat")

    # Visualize
    vis = VisualizePotential2D(pot, temp=temp,
                               xrange=[-1.5, 1.2], yrange=[-0.2, 2],
                               contourvals=50, clip=100)

    vis.scatter_traj(sim.traj, "./tmp/MB/traj.png")
    vis.scatter_traj_projection_x(sim.traj, "./tmp/MB/traj_x.png")
    vis.animate_traj(sim.traj, "./tmp/MB/traj_movie", every=50)
    vis.animate_traj_projection_x(sim.traj, "./tmp/MB/traj_movie", every=50)


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print("Running " + obj[0] + " ...")
            obj[1]()
            print("Done")
