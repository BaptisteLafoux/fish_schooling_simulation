import multiprocessing
import numpy as np
import xarray as xr 
from tqdm.contrib.concurrent import process_map
from glob import glob

from utils import merge_ds
from simulation import Simulation
from constants import *
from multiprocessing import Pool

def launch_simu(simu):
    simu.run()

if __name__ == "__main__":

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

    # # with multiprocessing.Pool(ncores) as pool:

    # with Pool(processes=6) as pool:
    #     pool.map(launch_simu, simulations)
    _ = process_map(launch_simu, simulations, max_workers=ncores)
    merge_ds()

             


