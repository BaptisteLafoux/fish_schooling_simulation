import numpy as np

from constants import *
import tqdm
from trajectory import Trajectory
from step import Step


from utils import animate
np.seterr(all='ignore')

class ErrorSimulationExploded(Exception):
    'Error in calculation, skipping this simulation ! Ciao >>>'
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

    def _init_traj(self):
        pos = np.zeros((2, self.N, T))
        vel = np.zeros((2, self.N, T))

        pos[0, :, 0] = np.random.uniform(0, W, self.N)
        pos[1, :, 0] = np.random.uniform(0, H, self.N)
        vel[..., 0]  = np.random.normal(0, v_ini, (2, self.N))
        return pos, vel

    def run(self):
        pos, vel = self._init_traj()

        self.fnoise, self.fp, self.fal, self.fat, self.fwall = np.zeros((2, self.N, T)), np.zeros((2, self.N, T)), np.zeros((2, self.N, T)), np.zeros((2, self.N, T)), np.zeros((2, self.N, T))

        for t in tqdm.tqdm(range(T - 1)):
            try:
                if not np.isfinite(pos[..., t]).all():
                    raise ErrorSimulationExploded
                step = Step(pos[..., t], vel[..., t], self.params)
                self.time_step(t, step, pos, vel)

            except ErrorSimulationExploded:
                print('Error : STOP')
                break
        
        self.params['v0'] = self.v0_avg
        traj = Trajectory(pos, vel, self.params, self)

        traj.to_netcdf()

        return pos, vel

    def time_step(self, t, step, pos, vel):

        fnoise  = step.get_noise_force()
        fp      = step.get_propulsion_force()
        fal     = step.get_alignment_force()
        fat     = step.get_attraction_force()
        fwall   = step.get_wall_force()

        vel[..., t + 1] = vel[..., t] + (fat + fal + fp + fwall) * dt + fnoise * np.sqrt(dt)
        pos[..., t + 1] = pos[..., t] + vel[..., t + 1] * dt

        self.fnoise[..., t + 1] = fnoise
        self.fp[..., t + 1] = fp
        self.fal[..., t + 1] = fal
        self.fat[..., t + 1] = fat
        self.fwall[..., t + 1] = fwall

        
if __name__ == '__main__': 

    N = 50
    sim = Simulation(N=N, tau=0.5, Ra=1.5, phi=250, v0=3, R=5, J=2, eps=1, a=0.5, b=0)

    pos, vel = sim.run()
    traj = Trajectory(pos, vel, sim.params, sim)

    anim = animate(traj, acc=5, draw_quiv=False, fov=False, tailsec=0.5)
    #anim.save(f'video/temp.mp4', dpi=100, fps=1/dt / 5 / 5)




