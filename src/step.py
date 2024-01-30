import numpy as np
# from scipy.spatial.distance import cdist
# from scipy.stats import vonmises
from constants import H, W, gammawall, delta


class Step:
    def __init__(self, X, V, params) -> None:
        self.X = X # shape (2, N)
        self.V = V # shape (2, N)

        self.params = params

        self.Xij = self.X[:, None, :] - self.X[:, :, None]
        self.rij = (self.Xij[0] ** 2 + self.Xij[1] ** 2) ** 0.5 + 1e-5*np.identity(self.params['N'])
        self.Vnorm = np.linalg.norm(self.V, axis=0)

        self.thetaij = np.abs(np.arccos(np.einsum("im,inm->mn", 
                                                  self.V / self.Vnorm, -self.Xij / self.rij)))
        np.fill_diagonal(self.thetaij, 0) 
        
        self.fov = (self.thetaij < (self.params['phi'] / 2 * np.pi / 180)).astype(int) * (self.rij < self.params['R']).astype(int) 
        self.nfov = self.fov.sum(axis=1)
        self.fov = self.fov / self.fov.sum(axis=1)
        np.fill_diagonal(self.fov, 0) 
        
        

    def get_noise_force(self):
        return np.random.normal(0, self.params['eps'], (2, self.params['N']))

    def get_propulsion_force(self):
        return (1 / self.params['tau']) * (1 - self.Vnorm**2 / self.params['v0']**2) * self.V

    def get_alignment_force(self):
        return self.params['J'] * (((self.V[:, None, :] - self.V[:, :, None])) * self.fov).sum(axis=2)

    def get_attraction_force(self):
        return self.params['a'] * ((self.Xij / self.rij -  (self.params['Ra']**2) * self.Xij / (self.rij**3)) * self.fov).sum(axis=2)
    
    def get_wall_force(self):

        dN = H - self.X[1] 
        dS = self.X[1] 
        dE = W - self.X[0] 
        dW = self.X[0]


        nN = np.r_[0,  1]
        nS = np.r_[0, -1]
        nE = np.r_[ 1, 0]
        nW = np.r_[-1, 0]


        FN = - (np.maximum((1 / dN - 1 / delta), 0) * nN[..., None]) * np.maximum(np.dot((self.V / self.Vnorm).T, nN[..., None]), 0).T

        FS = - (np.maximum((1 / dS - 1 / delta), 0) * nS[..., None]) * np.maximum(np.dot((self.V / self.Vnorm).T, nS[..., None]), 0).T

        FE = - (np.maximum((1 / dE - 1 / delta), 0) * nE[..., None]) * np.maximum(np.dot((self.V / self.Vnorm).T, nE[..., None]), 0).T

        FW = - (np.maximum((1 / dW - 1 / delta), 0) * nW[..., None]) * np.maximum(np.dot((self.V / self.Vnorm).T, nW[..., None]), 0).T

        return  gammawall * (FN + FS + FE + FW)


