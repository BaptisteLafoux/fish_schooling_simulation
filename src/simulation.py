"""
Created on Wed Jan 31 13:57:22 2023

Defines the 'Simulation' class, to initialize positions and enforce the integration scheme by creating a `Step`object at each timestep.

The script can be directly executed to run a single simulation. 
Usage: python simulation.py (for instance)

@author: BaptisteLafoux
"""

import numpy as np
import tqdm
np.seterr(all='ignore')

#homemade modules 
from constants import *
from trajectory import Trajectory
from step import Step
from utils import animate

class ErrorSimulationExploded(Exception):
    pass

class Simulation:
    def __init__(self, tau, J, N, eps, v0, a, Ra, phi, R, b) -> None:

        self.v0_avg = v0 
        self.params = dict(
            
            tau = tau,
            J = J,
            N = N,
            eps = eps,
            v0 = v0 * (1 + np.random.uniform(-b, b, N)),
            a = a,
            Ra = Ra,
            phi = phi,
            R = R,
            b = b,
        )

        self.N = N
        self.pos = np.zeros((2, self.N, T))
        self.vel = np.zeros((2, self.N, T))

    def _init_traj(self):

        self.pos[0, :, 0] = np.random.uniform(0, W, self.N)
        self.pos[1, :, 0] = np.random.uniform(0, H, self.N)
        
        self.vel[..., 0]  = np.random.normal(0, v_ini, (2, self.N))

    def run(self, savedata=True):
        """compute positions ond velocities at each timestep. 

        Raises:
            ErrorSimulationExploded: if at least one position is outside the boundary of the tank, stop the time loop

        Returns:
            traj (Trajectory): a trajectory object containing all the position and velocity data, and other physical quantities computed in post-processing (polarization, milling, distances)
        """
        self._init_traj()

        for t in tqdm.tqdm(range(T - 1)):
            try:
                if not (self.pos[..., t] <= L).all():
                    raise ErrorSimulationExploded
                
                step = Step(self.pos[..., t], self.vel[..., t], self.params)
                
                self.move(t, step)

            except ErrorSimulationExploded:
                print('Fatal error : \tone of the positions was out-of-bound (a fish escaped the tank...)\n\tSTOP this simulation: Ciao and next >>>>')
                break
        
        self.params['v0'] = self.v0_avg
        
        traj = Trajectory(self.pos, self.vel, self.params, self)
        
        if savedata: 
            traj.to_netcdf() #saves the Trajectory object into an xarray dataset (.nc file) 

        return traj

    def move(self, t, step):
        """Integration of the equaltions of motion: 
        - gets the social forces from a `Step` object 
        - updates the position and velocity arrays  
        """

        fnoise  = step.get_noise_force()
        fp      = step.get_propulsion_force()
        fal     = step.get_alignment_force()
        fat     = step.get_attraction_force()
        fwall   = step.get_wall_force()

        self.vel[..., t + 1] = self.vel[..., t] + (fat + fal + fp + fwall) * dt + fnoise * np.sqrt(dt)
        self.pos[..., t + 1] = self.pos[..., t] + self.vel[..., t + 1] * dt
        
if __name__ == '__main__': 

    sim = Simulation(N=20, 
                     tau=0.5, 
                     Ra=5, 
                     phi=360, #in degrees
                     v0=3, 
                     R=50, 
                     J=2, 
                     eps=1, 
                     a=0.5, 
                     b=0.4)
    
    traj = sim.run(savedata=False)
    anim = animate(traj, acc=5, draw_quiv=False, fov=False, tailsec=1, save=False)




