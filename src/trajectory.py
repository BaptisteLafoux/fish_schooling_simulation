from matplotlib import pyplot as plt

import numpy as np
import xarray as xr
from constants import *
import os 
from scipy.spatial.distance import cdist

from utils import get_rep_number, get_filename, get_coord_dict

class Trajectory:
    def __init__(self, pos, vel, params, simu) -> None:
        self.pos = pos
        self.vel = vel 

        self.r = self.pos - self.pos.mean(axis=1)[:, None, :]

        self.velnorm = np.linalg.norm(self.vel, axis=0)
        self.rnorm = np.linalg.norm(self.r, axis=0)

        self.attrs = dict(
            v_ini = v_ini,
            T = T,
            H = H,
            W = W,
            dt = dt,
        )

        self.filename = get_filename(params)

        params.update({'rep': get_rep_number(params)})
        self.params = params

    def get_m(self):
        return ((self.r[0] * self.vel[1] - self.r[1] * self.vel[0]) / (self.rnorm * self.velnorm)).mean(axis=0)
    
    def get_p(self):
        return np.linalg.norm((self.vel / self.velnorm).mean(axis=1), axis=0)
    
    def get_v(self):
        return np.linalg.norm(self.vel, axis=0).mean(axis=0)
    
    def get_nnd(self):
        rij = np.array([cdist(self.pos[..., t].T, self.pos[..., t].T)
                       for t in range(self.pos.shape[-1])])
        
        if self.params['N'] > 1: return np.sort(rij, axis=-1)[..., 1].mean(axis=-1)
        else: return rij.mean(axis=-1).mean(axis=-1)

    def get_iid(self):
        rij = np.array([cdist(self.pos[..., t].T, self.pos[..., t].T)
                       for t in range(self.pos.shape[-1])])

        if self.params['N'] > 1: return rij.mean(axis=-1).mean(axis=-1)
        else: return rij.mean(axis=-1).mean(axis=-1)

    def to_netcdf(self): 

        ds = xr.Dataset(data_vars=self.get_data_dict(), coords=get_coord_dict(self.params), attrs=self.attrs)
        
        if SAVEMETHOD=='timeseries': ds = ds.coarsen(time=subsamp).mean() ## reduce the frame rate of saved data
        ds.to_netcdf(os.path.join(TEMPSAVEPATH, self.filename), engine='scipy', mode='w')
        ds.close()

        return self.filename

    def get_data_dict(self):
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