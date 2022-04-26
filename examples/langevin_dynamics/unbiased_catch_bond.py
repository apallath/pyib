import os

import matplotlib.pyplot as plt
import numpy as np

from pyib.md.potentials import CatchBondPotential2D
from pyib.md.simulation import SingleParticleSimulation
from pyib.md.visualization import VisualizePotential2D

# Constants
temp = 300
force = 0
run = 0

init_coord = np.array([[-5, -4, 0]])

# Initialize potential
pot = CatchBondPotential2D(force_x=force)

# Initialize simulation
sim = SingleParticleSimulation(pot,
                               init_coord=init_coord,
                               traj_in_mem=True,
                               cpu_threads=1)

if not os.path.exists("./tmp/catch_bond_f{}/{}/".format(force, run)):
    os.makedirs("./tmp/catch_bond_f{}/{}/".format(force, run))

# Visualize
vis = VisualizePotential2D(pot, temp=temp,
                           xrange=[-12, 15], yrange=[-6, 8],
                           contourvals=10, clip=15)

# 2D surface
fig, ax = vis.plot_potential()
plt.savefig("./tmp/catch_bond_f{}/{}/pot.png".format(force, run))

# 1D projection
fig, ax, _, _ = vis.plot_projection_x()
plt.savefig("./tmp/catch_bond_f{}/{}/pot_x.png".format(force, run))

# Run simulation
sim(nsteps=200001,
    chkevery=5000,
    trajevery=50,
    energyevery=50,
    chkfile="./tmp/catch_bond_f{}/{}/chk_state.pkl".format(force, run),
    trajfile="./tmp/catch_bond_f{}/{}/traj.dat".format(force, run),
    energyfile="./tmp/catch_bond_f{}/{}/energies.dat".format(force, run))

# Trajectories
vis.scatter_traj(sim.traj, "./tmp/catch_bond_f{}/{}/traj.png".format(force, run), every=50)
vis.scatter_traj_projection_x(sim.traj, "./tmp/catch_bond_f{}/{}/traj_x.png".format(force, run), every=50)
vis.animate_traj(sim.traj, "./tmp/catch_bond_f{}/{}/traj_movie".format(force, run), every=50)
vis.animate_traj_projection_x(sim.traj, "./tmp/catch_bond_f{}/{}/traj_movie".format(force, run), every=50)
