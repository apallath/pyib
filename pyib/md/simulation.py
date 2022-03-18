from os import times
from openmm.app import *
import openmm.unit as unit
from openmm import *

import numpy as np
import multiprocessing as mp


class MDSimulation:
    def __init__(self, Potential:CustomExternalForce, 
                init_coords : np.ndarray, 
                friction= 10/unit.picosecond,
                temperature = 300 * unit.kelvin, 
                timestep = 2.0*unit.femtosecond, 
                mass     = None,
                platform = "CPU"):
        self.__potential = Potential
        self.timestep    = timestep
        self.friction    = friction
        self.temperature = temperature

        self.integrator  = LangevinIntegrator(temperature, friction, timestep)

        # Initialize the initial coordinates
        self.coords = init_coords
        self.N      = self.coords.shape[0]

        self.system = System()

        if mass is not None:
            self.mass = mass
        else:
            self.mass = np.ones((self.N,)) * 1.0 * unit.dalton

        for i in range(self.N):
            self.system.addParticle(self.mass[i])
            self.__potential.addParticle(i,[])

        self.platform = Platform.getPlatformByName(platform)
        self.system.addForce(self.__potential)
        self.num_threads = str(mp.cpu_count())
        self.properties = {"Threads" : self.num_threads}
        self.context = Context(self.system, self.integrator, self.platform, self.properties)
        self.context.setPositions(init_coords)
        self.context.setVelocitiesToTemperature(self.temperature)
        
    
    def __call__(self, iterations:int, printevery=-1, outputfile=None, PositionTrackFreq=-1):
        if PositionTrackFreq != -1:
            TotalPos = []

        for i in range(iterations):
            # Step in integrator
            self.integrator.step(1)

            # record position
            pos = self.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)

            # get energy information
            force = self.context.getState(getForces=True).getForces()
            PE  = self.context.getState(getEnergy=True).getPotentialEnergy()
            KE  = self.context.getState(getEnergy=True).getKineticEnergy()
            TE  = PE + KE

            if printevery != -1:
                if i % printevery == 0:
                    print("Step : {} | PE = {} , KE = {}, TE = {}".format(i+1, PE, KE, TE))

            if PositionTrackFreq != -1:
                if i % PositionTrackFreq == 0:
                    TotalPos.append(pos)
        
        if PositionTrackFreq != -1:
            TotalPos = np.concatenate(TotalPos)
            return TotalPos
        else:
            return 1
    