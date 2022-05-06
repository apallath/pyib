import os

import matplotlib.pyplot as plt
import numpy as np

from pyib.md.potentials import SlipBondPotential2D
from pyib.md.simulation import SingleParticleSimulation
from pyib.md.visualization import VisualizePotential2D
from pyib.md.utils import TrajectoryReader


##########################################################
# Experiment with these parameters:

# Constants
temp = 300
force = 0

init_coord = np.array([[-5, -4, 0]])
##########################################################

# Initialize potential
pot = SlipBondPotential2D(force_x=force)

# Initialize simulation
# For this example, we won't save trajectory data in memory
sim = SingleParticleSimulation(pot,
                               temp=temp,
                               init_coord=init_coord,
                               traj_in_mem=False,
                               cpu_threads=1)

# Visualize
vis = VisualizePotential2D(pot, temp=temp,
                           xrange=[-15, 20], yrange=[-8, 10],
                           contourvals=20, clip=10)

if not os.path.exists("./tmp/slip_bond_read_traj/"):
    os.makedirs("./tmp/slip_bond_read_traj/")

# 2D surface
fig, ax = vis.plot_potential()
plt.savefig("./tmp/slip_bond_read_traj/pot.png")

# 1D projection
fig, ax, _, _ = vis.plot_projection_x()
plt.savefig("./tmp/slip_bond_read_traj/pot_x.png")

# Run simulation
sim(nsteps=200001,
    chkevery=5000,
    trajevery=50,
    energyevery=50,
    chkfile="./tmp/slip_bond_read_traj/chk_state.pkl",
    trajfile="./tmp/slip_bond_read_traj/traj.dat",
    energyfile="./tmp/slip_bond_read_traj/energies.dat")

# Read trajectory
reader = TrajectoryReader("./tmp/slip_bond_read_traj/traj.dat")
_, traj = reader.read_traj()

# Trajectories (already in memory)
vis.scatter_traj(traj, "./tmp/slip_bond_read_traj/traj.png", every=50)
vis.scatter_traj_projection_x(traj, "./tmp/slip_bond_read_traj/traj_x.png", every=50)
vis.animate_traj(traj, "./tmp/slip_bond_read_traj/traj_movie", every=500, ffmpeg_rate=50)
vis.animate_traj_projection_x(traj, "./tmp/slip_bond_read_traj/traj_movie", every=500, ffmpeg_rate=50)
