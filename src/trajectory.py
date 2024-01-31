"""
Created on Wed Jan 31 16:57:22 2023

Defines the 'Trajectory' class, used to compute physical values after a simulation, and to save the data as xarray datasets

The script is not intended to be directly executed; it a module that is used by other files
@author: BaptisteLafoux
"""

from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

import numpy as np
import xarray as xr
import os 

from utils import get_rep_number, get_filename, get_coord_dict
from constants import *

class Trajectory:
    def __init__(self, pos, vel, params) -> None:
        """ Initialization function of a Trajectory (called when a Trajectory object is created)
        
        Args:
            - pos (`np.ndarray`): Positions of the individuals in the 2D plan over time. numpy array of size (2, N, T)
            - vel (`np.ndarray`): Velocities of the individuals in the 2D plan over time. numpy array of size (2, N, T)
            - params (dict): Dictionnary containing the parameters of a simulation
        """
        self.pos = pos
        self.vel = vel 

        # postion with respect to the center of mass -- shape (2, N, T)
        self.R = self.pos - self.pos.mean(axis=1)[:, None, :]

        # norm of the velocity -- shape (N, T)
        self.velnorm = np.linalg.norm(self.vel, axis=0)
        # distance to the center of mass -- shape (N, T)
        self.Rnorm = np.linalg.norm(self.R, axis=0)
        
        # rij = ||Xi - Xj||
        self.rij = np.linalg.norm(self.pos[:, None, :, :] - self.pos[:, :, None, :], axis=0) 

        # futur attributs of the dataset (constants)
        self.attrs = dict(
            v_ini = v_ini,
            T = T,
            H = H,
            W = W,
            dt = dt,
            gammawall = gammawall,
            delta = delta,
        )

        self.filename = get_filename(params)

        params.update({'rep': get_rep_number(params)})
        self.params = params

    def get_m(self):
        """Computes milling parameter -- shape (T,)"""
        return ((self.R[0] * self.vel[1] - self.R[1] * self.vel[0]) / (self.Rnorm * self.velnorm)).mean(axis=0)
    
    def get_p(self):
        """Computes polarization parameter -- shape (T,)"""
        return np.linalg.norm((self.vel / self.velnorm).mean(axis=1), axis=0)
    
    def get_v(self):
        """Computes norm of velocity averaged over all individuals -- shape (T,)"""
        return np.linalg.norm(self.vel, axis=0).mean(axis=0)
    
    def get_nnd(self):
        """Computes distance to nearest neighbour, averaged over all individuals -- shape (T,)"""
        ## if there is only 1 fish, nnd = 0 
        if self.params['N'] > 1: return np.sort(self.rij, axis=-1)[..., 1].mean(axis=-1)
        else: return np.zeros(T)

    def get_iid(self):
        """Computes average distance between every indiv, averaged over all individuals -- shape (T,)"""
        ## if there is only 1 fish, iid = 0 
        if self.params['N'] > 1: return self.rij.mean(axis=-1).mean(axis=-1)
        else: return np.zeros(T)

    def to_netcdf(self): 
        """Saving function: saves a .nc dataset and return the name of the saved file"""

        ds = xr.Dataset(data_vars=self.get_data_dict(), coords=get_coord_dict(self.params), attrs=self.attrs)
        
        if SAVEMETHOD=='timeseries': 
            ds = ds.coarsen(time=subsamp).mean() ## reduce the frame rate of saved data if saving multiple timepoints
            
        ds.to_netcdf(os.path.join(TEMPSAVEPATH, self.filename), engine='scipy', mode='w')
        ds.close()

        return self.filename

    def get_data_dict(self):
        """See https://docs.xarray.dev/en/stable/getting-started-guide/quick-overview.html if needed"""
        param_keys = [key for key in self.params.keys()]
        n_params = len(param_keys)

        npts = int((T * percentage_of_pts_tosave))
        
        if SAVEMETHOD=='timeseries': 
            
            data = dict(
                X = (['space', 'time', 'fish'] + param_keys, self.pos[..., -npts:].transpose(0, 2, 1)[(..., slice(None)) + (None, ) * n_params]),
                V = (['space', 'time', 'fish'] + param_keys, self.vel[..., -npts:].transpose(0, 2, 1)[(..., slice(None)) + (None, ) * n_params]),

                m = (['time'] + param_keys, self.get_m()[-npts:][(..., slice(None)) + (None, ) * n_params]),
                p = (['time'] + param_keys, self.get_p()[-npts:][(..., slice(None)) + (None, ) * n_params]),
                v = (['time'] + param_keys, self.get_v()[-npts:][(..., slice(None)) + (None, ) * n_params]),

                nnd = (['time'] + param_keys, self.get_nnd()[-npts:][(..., slice(None)) + (None, ) * n_params]),
                iid = (['time'] + param_keys, self.get_iid()[-npts:][(..., slice(None)) + (None, ) * n_params]),

            )

        if SAVEMETHOD=='lastpoint':
            data = dict(
                m = (param_keys, np.array([self.get_m()[-npts:].mean()])[(..., slice(None)) + (None, ) * (n_params-1)]),
                p = (param_keys, np.array([self.get_p()[-npts:].mean()])[(..., slice(None)) + (None, ) * (n_params-1)]),
                v = (param_keys, np.array([self.get_v()[-npts:].mean()])[(..., slice(None)) + (None, ) * (n_params-1)]),

                nnd = (param_keys, np.array([self.get_nnd()[-npts:].mean()])[(..., slice(None)) + (None, ) * (n_params-1)]),
                iid = (param_keys, np.array([self.get_iid()[-npts:].mean()])[(..., slice(None)) + (None, ) * (n_params-1)]),
                
            )

        return data