"""
Created on Wed Jan 31 09:57:22 2023

Generates a bunch of simulation with multuiple parameters defined in 'constants' and runs them in parallel

The script is intended to be directly executed.
Usage: python main.py (for instance)

@author: BaptisteLafoux
"""

from tqdm.contrib.concurrent import process_map

from utils import merge_ds
from simulation import Simulation
from constants import taus, Js, Ns, bs, Rs, epss, v0s, As, Ras, phis, nrep, ncores

def launch_simu(simu):
    """just a wrapper to run a simulation with given parameters in the multiprocessing map 

    Args:
        simu (Simulation): a Simulation 
    """
    simu.run()

if __name__ == "__main__":
    
    ## an array of simulations 
    simulations = [Simulation(tau=tau, J=J, N=N, eps=eps, v0=v0, a=a, Ra=Ra, phi=phi, b=b, R=R) 
                              
                              for tau in taus 
                              for J in Js 
                              for N in Ns
                              for b in bs
                              for R in Rs
                              
                              for eps in epss 
                              for v0 in v0s
                              for a in As
                              for Ra in Ras
                              for phi in phis
                              for _ in range(nrep)
                              ]

    ## runs all the different simulations created from the arrays of parameters in "constants" on "ncores" number of parallel processors 
    
    _ = process_map(launch_simu, simulations, max_workers=ncores)
    
    ## 
    merge_ds(light=True, 
             dropped_vars=['X', 'V'], 
             dropped_dims=['fish'], 
             delete_files=False)

