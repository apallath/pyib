"""
Classes for performing simulations.
"""
import multiprocessing
import pickle

import numpy as np
from openmm import unit
from openmm import openmm
from tqdm import tqdm


class SingleParticleSimulation:
    """
    Performs langevin dynamics simulation of a particle on a potential energy surface.

    Attributes:
        potential (openmm.CustomExternalForce): Underlying potential energy surface.
        mass (int): Mass of particle on the surface, in dalton (default = 1).
        temp (float): Temperature, in Kelvin (default = 300).
        friction (float): Friction factor, in ps^-1 (default = 100).
        timestep (float): Timestep, in fs (default = 10).
        init_state (openmm.State): Initial state for re-runs (default = None).
        init_coord (np.ndarray): Initial coordinates of the particle on the surface (default = [[0, 0, 0]]).
        gpu (bool): If True, uses GPU for simulation (default = False).
        cpu_threads (int): If gpu is False, number of CPU threads to use for simulation. If None, the max cpu count is used. (default = None).
        seed (int): Seed for reproducibility (default = None).
        traj_in_mem (bool): If True, stores trajectory in memory. The trajectory can be accessed by the object's `traj` attribute (default=False).

    Arguments:
        nsteps (int): Number of steps to run simulation for (default = 1000)
        chkevery (int): Checkpoint interval (default = 500).
        trajevery (int): Trajectory output interval (default = 1).
        energyevery (int): Energy output interval (default = 1).
        chkfile (str): File to write checkpoints to (default = "./chk_state.pkl"). If this file already exists, it will be overwritten.
        trajfile (str): File to write trajectory data to (default = "./traj.dat"). If this file already exists, it will be overwritten.
        energyfile (str): File to write energies to (default = "./energies.dat"). If this file already exists, it will be overwritten.
    """
    def __init__(self,
                 potential: openmm.CustomExternalForce,
                 mass: int = 1,
                 temp: float = 300,
                 friction: float = 100,
                 timestep: float = 10,
                 init_state: openmm.State = None,
                 init_coord: np.ndarray = np.array([0, 0, 0]).reshape((1, 3)),
                 gpu: bool = False,
                 cpu_threads: int = None,
                 seed: int = None,
                 traj_in_mem: bool = False):
        # Properties
        self.mass = mass * unit.dalton  # mass of particles
        self.temp = temp * unit.kelvin  # temperature
        self.friction = friction / unit.picosecond  # LD friction factor
        self.timestep = timestep * unit.femtosecond   # LD timestep

        self.init_state = init_state
        self.gpu = gpu
        self.traj_in_mem = traj_in_mem

        # Init simulation objects
        self.system = openmm.System()
        self.potential = potential
        self.system.addParticle(self.mass)
        self.potential.addParticle(0, [])  # no parameters associated with each particle
        self.system.addForce(potential)

        self.integrator = openmm.LangevinIntegrator(self.temp,
                                                    self.friction,
                                                    self.timestep)

        if seed is not None:
            self.integrator.setRandomNumberSeed(seed)

        if self.gpu:
            platform = openmm.Platform.getPlatformByName('CUDA')
            properties = {'CudaPrecision': 'mixed'}
            print("Running simulation on GPU.")
        else:
            platform = openmm.Platform.getPlatformByName('CPU')
            if cpu_threads is None:
                cpu_threads = multiprocessing.cpu_count()
            properties = {'Threads': str(cpu_threads)}
            print("Running simulation on {} CPU threads.".format(cpu_threads))

        self.context = openmm.Context(self.system, self.integrator, platform, properties)

        # Init state
        if init_state is None:
            self.context.setPositions(init_coord)
            if seed is not None:
                self.context.setVelocitiesToTemperature(self.temp, randomSeed=seed)
            else:
                self.context.setVelocitiesToTemperature(self.temp)
        else:
            self.context.setState(init_state)

    def __call__(self,
                 nsteps: int = 1000,
                 chkevery: int = 500,
                 trajevery: int = 1,
                 energyevery: int = 1,
                 chkfile="./chk_state.pkl",
                 trajfile="./traj.dat",
                 energyfile="./energies.dat"):

        if self.traj_in_mem:
            self.traj = None

        for i in tqdm(range(nsteps)):
            # Checkpoint
            if i > 0 and i % chkevery == 0:
                self._dump_state(chkfile, i)

            # Store positions
            if i % trajevery == 0:
                self._write_trajectory(trajfile, i)

            # Store energy
            if i % energyevery == 0:
                self._write_energies(energyfile, i)

            # Integrator step
            self.integrator.step(1)

        # Finalize
        self._dump_state(chkfile, i)

    def _dump_state(self, ofilename, i):
        t = i * self.timestep / unit.picosecond
        print("Checkpoint at {:10.7f} ps".format(t))

        state = self.context.getState(getPositions=True, getVelocities=True)

        with open(ofilename, "wb") as fh:
            pickle.dump(state, fh)

    def _write_trajectory(self, ofilename, i):
        t = i * self.timestep / unit.picosecond
        pos = self.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        # Store trajectory in memory
        if self.traj_in_mem:
            if i == 0:
                self.traj = pos
            else:
                self.traj = np.vstack((self.traj, pos))

        # Write trajectory to disk
        if i == 0:
            with open(ofilename, "w") as of:
                of.write("# t[ps]    x [nm]    y [nm]    z[nm]\n")

        with open(ofilename, "a") as of:
            of.write("{:10.5f}\t{:10.7f}\t{:10.7f}\t{:10.7f}\n".format(t, pos[0, 0], pos[0, 1], pos[0, 2]))

    def _write_energies(self, ofilename, i):
        t = i * self.timestep / unit.picosecond
        PE = self.context.getState(getEnergy=True).getPotentialEnergy() / unit.kilojoule_per_mole
        KE = self.context.getState(getEnergy=True).getKineticEnergy() / unit.kilojoule_per_mole

        if i == 0:
            with open(ofilename, "w") as of:
                of.write("# t[ps]    PE [kJ/mol]    KE [kJ/mol]\n")

        with open(ofilename, "a") as of:
            of.write("{:10.5f}\t{:10.7f}\t{:10.7f}\n".format(t, PE, KE))
