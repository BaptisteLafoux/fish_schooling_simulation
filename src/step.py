"""
Created on Wed Jan 31 13:57:22 2023

Defines the 'Step' class, used to compute social forces at each timestep

The script is not intended to be directly executed; it a module that is used by other files
@author: BaptisteLafoux
"""

import numpy as np
from constants import H, W, gammawall, delta


class Step:
    def __init__(self, X, V, params) -> None:
        """Initialization function of a `Step` Object. Generates and attaches once and for all useful data to avoid computing them several times at each timestep. 

        Args:
            - X (`np.ndarray`): Positions of the individuals in the 2D plan. numpy array  of size (2, N), N being the number of fish. x-positions: X[0], y-position: X[1]
            - V (`np.ndarray`): Velocities in the 2D plan, of size (2, N)
            - params (dict): a dictionary containing the simulation parameters
        """
        self.X = X
        self.V = V
        
        self.params = params

        # Xij = Xi - Xj -- shape = (2, N, N)
        self.Xij = self.X[:, None, :] - self.X[:, :, None]
        
        # rij = ||Xi - Xj|| (an epsilon value is added on the diagonal to avoid division by 0 later) -- shape = (N, N) 
        self.rij = (self.Xij[0] ** 2 + self.Xij[1] ** 2) ** 0.5 + 1e-9*np.identity(self.params['N'])
        
        # norm of the velocities -- shape = (N, )
        self.Vnorm = np.linalg.norm(self.V, axis=0) 

        # angle between direction and motion of individual i and the unit vector from i to j -- shape (N, N)
        self.thetaij = np.abs(np.arccos(np.einsum("im,inm->mn", self.V / self.Vnorm, -self.Xij / self.rij)))
        # make sure that thetaij[i, i] = 0 
        np.fill_diagonal(self.thetaij, 0) 
        
        # Field of view mask matrix. For each i, if j is outside the angle of vision, fov[i, j] = 0, otherwise 1 (similarly, if rij[i, j] > R, fov[i, j] = 0) -- shape (N, N)
        self.fov = (self.thetaij < (self.params['phi'] / 2 * np.pi / 180)).astype(int) * (self.rij < self.params['R']).astype(int) 
        
        # Number of individual in the field of view of each individual 
        self.nfov = self.fov.sum(axis=1)
        
        # We normalize the field of view by the number of indiv in the field of view of each indiv i  
        self.fov = self.fov / self.fov.sum(axis=1)
        np.fill_diagonal(self.fov, 0) 
        
    
    def get_noise_force(self):
        """Generate a random noise force, according to parameter epsilon
        
        Returns:
            `np.array`: random noise force -- shape (2, N)
        """
        return np.random.normal(0, self.params['eps'], (2, self.params['N']))

    def get_propulsion_force(self):
        """This function will calculate the propulsion force for each individual:

            F_prop = 1/tau * (1 - |V|^2 / v_0^2) * V 
            
        Returns:
            `np.array`: Propulsion force -- shape (2, N)
        """
        return (1 / self.params['tau']) * (1 - self.Vnorm**2 / self.params['v0']**2) * self.V

    def get_alignment_force(self):
        """This function will calculate the alignment force for each individual:

            F_ali = J * \sum_{j=1}^N (V_i - V_j) * Z_ali/N_i,al
            
            Here Z_ali/N_i,al is encompassed in the `fov` matrix 
            
        Returns:
            `np.array`: Alignment force -- shape (2, N)
        """
        Vij = self.V[:, None, :] - self.V[:, :, None]
        return self.params['J'] * (Vij * self.fov).sum(axis=2) # axis=2 sums over j 

    def get_attraction_force(self):
        """This function will calculate the attraction-repulsion force for each individual:

            F_att = a * \sum_{j=1}^N [ xij / ||xij|| - Ra^2 * xij / ||xij||^3 ] * Z_att/N_i,att
            
            Here Z_att/N_i,att is encompassed in the `fov` matrix (/!\ here the fov is the same for alignment and attraction)
            
        Returns:
            `np.array`: Attraction force -- shape (2, N)
        """
        
        return self.params['a'] * ((self.Xij / self.rij -  (self.params['Ra']**2) * self.Xij / (self.rij**3)) * self.fov).sum(axis=2)
    
    def get_wall_force(self):

        # distances of each indiv to North, South, East, West walls 
        dN = H - self.X[1] 
        dS = self.X[1] 
        dE = W - self.X[0] 
        dW = self.X[0]

        # normal vector to North, South, East, West walls (directed from inside to outside)
        nN = np.r_[0,  1]
        nS = np.r_[0, -1]
        nE = np.r_[ 1, 0]
        nW = np.r_[-1, 0]

        # Compute walls forces 
        # if distance to wall is larger than delta, force is 0 
        # if dot product between normal to wall and velocity is negative, force is 0 
        FN = - (np.maximum((1 / dN - 1 / delta), 0) * nN[..., None]) * np.maximum(np.dot((self.V).T, nN[..., None]), 0).T

        FS = - (np.maximum((1 / dS - 1 / delta), 0) * nS[..., None]) * np.maximum(np.dot((self.V).T, nS[..., None]), 0).T

        FE = - (np.maximum((1 / dE - 1 / delta), 0) * nE[..., None]) * np.maximum(np.dot((self.V).T, nE[..., None]), 0).T

        FW = - (np.maximum((1 / dW - 1 / delta), 0) * nW[..., None]) * np.maximum(np.dot((self.V).T, nW[..., None]), 0).T

        # apply the gamma_wall intensity coefficient 
        return  gammawall * (FN + FS + FE + FW)


