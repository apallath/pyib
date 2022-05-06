import os

import matplotlib.pyplot as plt
import numpy as np
from openmmtorch import TorchForce
import torch

from pyib.md.potentials import SlipBondPotential2D
from pyib.md.simulation import SingleParticleSimulation
from pyib.md.utils import legendreFit, getLegendreValues
from pyib.md.visualization import VisualizePotential2D
from pyib.ml.biases import LegendreBias1D

if not os.path.exists("./tmp/slip_bond_legendre_biased/"):
    os.makedirs("./tmp/slip_bond_legendre_biased/")

##########################################################
# Experiment with these parameters:

# Constants
temp = 300
force = 0
run = 0

# Legendre polynomial degree
deg = 6

init_coord = np.array([[-5, -4, 0]])
##########################################################

# Initialize potential
pot = SlipBondPotential2D(force_x=force)

# Visualize
vis = VisualizePotential2D(pot, temp=temp,
                           xrange=[-12, 15], yrange=[-6, 8],
                           contourvals=10, clip=15)

# 2D surface
fig, ax = vis.plot_potential()
plt.savefig("./tmp/slip_bond_legendre_biased/pot.png")

# 1D projection - x
fig, ax, x, Fx = vis.plot_projection_x()
c_x = legendreFit(x, Fx, -12, 15, deg=deg)
Fx_fit = getLegendreValues(x, -12, 15, c_x)
ax.plot(x, Fx_fit, "--")
plt.savefig("./tmp/slip_bond_legendre_biased/pot_x_fit.png")

# 1D projection - y
fig, ax, y, Fy = vis.plot_projection_y()
c_y = legendreFit(y, Fy, -6, 8, deg=deg)
Fy_fit = getLegendreValues(y, -6, 8, c_y)
ax.plot(y, Fy_fit, "--")
plt.savefig("./tmp/slip_bond_legendre_biased/pot_y_fit.png")

"""
Baseline - unbiased simulation
"""

# Initialize simulation
sim = SingleParticleSimulation(pot,
                               init_coord=init_coord,
                               traj_in_mem=True,
                               cpu_threads=1)

# Run simulation
sim(nsteps=1000001,
    chkevery=50000,
    trajevery=100,
    energyevery=100,
    chkfile="./tmp/slip_bond_f{}/{}/chk_state.pkl".format(force, run),
    trajfile="./tmp/slip_bond_f{}/{}/traj.dat".format(force, run),
    energyfile="./tmp/slip_bond_f{}/{}/energies.dat".format(force, run))

# Trajectories (already in memory)
vis.scatter_traj(sim.traj, "./tmp/slip_bond_legendre_biased/traj.png", every=50)
vis.scatter_traj_projection_x(sim.traj, "./tmp/slip_bond_legendre_biased/traj_x.png", every=500)
vis.animate_traj(sim.traj, "./tmp/slip_bond_legendre_biased/traj_movie", every=50)
vis.animate_traj_projection_x(sim.traj, "./tmp/slip_bond_legendre_biased/traj_movie", every=500)

# Timeseries
fig, ax = plt.subplots(dpi=300)
ax.plot(sim.traj[:, 0], label="x")
ax.plot(sim.traj[:, 1], label="y")
ax.legend()
ax.set_xlabel("t")
ax.set_ylabel("pos (nm)")
fig.savefig("./tmp/slip_bond_legendre_biased/timeseries.png")

"""
Flattened along x using TorchForce.
Warning: this is incredibly expensive!
"""

bias_model_x = LegendreBias1D(deg, -c_x, -12, 15, axis='x')
torch.jit.script(bias_model_x).save("./tmp/slip_bond_legendre_biased/bias_x.pt")
bias_x = TorchForce("./tmp/slip_bond_legendre_biased/bias_x.pt")

sim = SingleParticleSimulation(pot,
                               init_coord=init_coord,
                               traj_in_mem=True,
                               cpu_threads=1,
                               bias=bias_x)

# Run simulation
sim(nsteps=1000001,
    chkevery=50000,
    trajevery=100,
    energyevery=100,
    chkfile="./tmp/slip_bond_legendre_biased/chk_state_bias_x.pkl",
    trajfile="./tmp/slip_bond_legendre_biased/traj_bias_x.dat",
    energyfile="./tmp/slip_bond_legendre_biased/energies_bias_x.dat")

# Trajectories (already in memory)
vis.scatter_traj(sim.traj, "./tmp/slip_bond_legendre_biased/traj_bias_x.png", every=50)
vis.scatter_traj_projection_x(sim.traj, "./tmp/slip_bond_legendre_biased/traj_x_bias_x.png", every=500)
vis.animate_traj(sim.traj, "./tmp/slip_bond_legendre_biased/traj_movie_bias_x", every=50)
vis.animate_traj_projection_x(sim.traj, "./tmp/slip_bond_legendre_biased/traj_movie_bias_x", every=500)

# Timeseries
fig, ax = plt.subplots(dpi=300)
ax.plot(sim.traj[:, 0], label="x")
ax.plot(sim.traj[:, 1], label="y")
ax.legend()
ax.set_xlabel("t")
ax.set_ylabel("pos (nm)")
fig.savefig("./tmp/slip_bond_legendre_biased/timeseries_bias_x.png")

"""
Flattened along y using TorchForce.
Warning: this is incredibly expensive!
"""
bias_model_y = LegendreBias1D(deg, -c_y, -6, 8, axis='y')
torch.jit.script(bias_model_y).save("./tmp/slip_bond_legendre_biased/bias_y.pt")
bias_y = TorchForce("./tmp/slip_bond_legendre_biased/bias_y.pt")

sim = SingleParticleSimulation(pot,
                               init_coord=init_coord,
                               traj_in_mem=True,
                               cpu_threads=1,
                               bias=bias_y)

# Run simulation
sim(nsteps=1000001,
    chkevery=50000,
    trajevery=100,
    energyevery=100,
    chkfile="./tmp/slip_bond_legendre_biased/chk_state_bias_y.pkl",
    trajfile="./tmp/slip_bond_legendre_biased/traj_bias_y.dat",
    energyfile="./tmp/slip_bond_legendre_biased/energies_bias_y.dat")

# Trajectories (already in memory)
vis.scatter_traj(sim.traj, "./tmp/slip_bond_legendre_biased/traj_bias_y.png", every=50)
vis.scatter_traj_projection_x(sim.traj, "./tmp/slip_bond_legendre_biased/traj_x_bias_y.png", every=500)
vis.animate_traj(sim.traj, "./tmp/slip_bond_legendre_biased/traj_movie_bias_y", every=50)
vis.animate_traj_projection_x(sim.traj, "./tmp/slip_bond_legendre_biased/traj_movie_bias_y", every=500)

# Timeseries
fig, ax = plt.subplots(dpi=300)
ax.plot(sim.traj[:, 0], label="x")
ax.plot(sim.traj[:, 1], label="y")
ax.legend()
ax.set_xlabel("t")
ax.set_ylabel("pos (nm)")
fig.savefig("./tmp/slip_bond_legendre_biased/timeseries_bias_y.png")
