"""
Created on Wed Jan 31 13:57:22 2023

Utility functions. 

The script is not intended to be directly executed; it a module that is used by other files
@author: BaptisteLafoux
"""

from constants import TEMPSAVEPATH, SAVEPATH, SAVEMETHOD, VIDEOPATH, batch_filename, T, percentage_of_pts_tosave, dt, W, H

import numpy as np
import tqdm
import glob
import os 
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

########################
## Utils for merging a stack of .nc file after a batch run 

def merge_ds(light: bool=False, dropped_vars=['X', 'V'], dropped_dims=['fish'], delete_files: bool=False):
    """ Merge multiple xarray datasets (.nc files) present in the temporary folder TEMPSAVEPATH into a larger dataset. Each dataset corresponds to a single simulation, the merged dataset regroups them all into a larger file for post-processing and handling purposes. 

    Args:
        - light (bool, optional): if True, the merged dataset is saved in a "lighter" mode. Same variables and dimensions are dropped, leading to a file that is much smaller (see dropped_vars and dropped_dims argument). Defaults to False.
        - dropped_vars (list, optional): List of variables to drop from the merged dataset. Ignored if light is False. Defaults to ['X', 'V'].
        - dropped_dims (list, optional): List of dimensions to drop from the merged dataset. Ignored if light is False. Defaults to ['fish'].
        - delete_files (bool, optional): if True, deletes all .nc file in the TEMPSAVEPATH after merging. Defaults to False.
    """

    dspaths = glob.glob(f'{TEMPSAVEPATH}/*.nc')
    dslist = []
    print(f'\n##############\n\t> Saving a merged file in {SAVEPATH} from {len(dspaths)} file located in {TEMPSAVEPATH}\n')

    for file in tqdm.tqdm(dspaths):
        ds_temp = xr.load_dataset(file)
        if light: 
            ds_temp = ds_temp.drop(dropped_vars)
            ds_temp = ds_temp.drop_dims(dropped_dims)
        dslist.append(ds_temp)
        
        ds_temp.close()

    ds = xr.combine_by_coords(dslist, combine_attrs='drop_conflicts') 
    ds.to_netcdf(os.path.join(SAVEPATH, f'{batch_filename}.nc'))

    if delete_files: 
        for file in os.scandir(TEMPSAVEPATH):
            os.remove(file.path)
        
########################
## Utils for saving trajectories 

def get_rawname(params)->str:
    """Generate the name under which a simulation will be saved into an xarray dataset (.nc file), witout the suffix that corresponds to the replicate number

    Args:
        params (dict): Dictionnary containing the parameters of the simulation

    Returns:
        str: a file name 
    """
    return '_'.join([f'{key}={params[key]:06.3f}' for key in params.keys()])

def get_rep_number(params)->int: 
    """ Provides the replicate number (number of replications of simulations with the exact same parameters)

    Args:
        params (dict): Dictionnary containing the parameters of the simulation

    Returns:
        int: Number of replicates
    """
    rep = 0 

    while os.path.exists(os.path.join(TEMPSAVEPATH, f'{get_rawname(params)}_{rep}.nc')):
        rep +=1

    return rep 

def get_filename(params):
    """ Generate a file name for an xarray dataset, of the form rawfilename_number-of-replicates.nc

    Args:
        params (dict): Dictionnary containing the parameters of the simulation

    Returns:
        str: Complete dataset filename
    """
    rep = get_rep_number(params)
    return f'{get_rawname(params)}_{rep}.nc'

def get_coord_dict(params): 
    """ Genereate a dictonary containing the coordinates of a simulation (used to create an xarray dataset)

    Args:
        params (dict): Dictionnary containing the parameters of the simulation

    Returns:
        dict: The coordinate dictionary  
    """

    coords = {key: ([key], [params[key]]) for key in params.keys()}

    if SAVEMETHOD == 'lastpoint':
        pass
    
    elif SAVEMETHOD == 'timeseries':
        coords.update(dict(
            time = (['time'], np.arange(int(T * percentage_of_pts_tosave)) * dt ),
            space = (['space'], ['x', 'y']),
            fish = (['fish'], np.arange(params['N'])),
        ))
    
    return coords


########################
## utils for animation


def animate(traj, acc: int=5, save=False, draw_quiv=False, fov=False, tailsec=0.5):
    """Generates an animation from a Trajectory (result of a Simulation)

    Args:
        traj (Trajectory): a Trajectory object created from a Simulation
        acc (int, optional): Acceleration factor of the video. Defaults to 5.
        save (bool, optional): if True, saves the video in the VIDEOPATH folder. Defaults to False.
        draw_quiv (bool, optional): if True, draw velocity of individuals as arrows pointing forward. Defaults to False.
        fov (bool, optional): if True, draws circles corresponding to alignment radius Ra (empty circle) and cone of vision around each individual. Defaults to False.
        tailsec (float, optional): length of the trace left behind by moving individuals in the animation, in seconds of physical time. Defaults to 0.5.

    Returns:
        anim (matplotlib Animation): an Animation object
    """

    ## real number of points in the tail (because tailsec * dt would be too much points usually, and the animation would be slow)
    npts_in_tail = 12
    ## length of tail in timesteps
    tail = int(tailsec / dt) 
    
    ## base acceleration for saved movies, because otherwise they would be impossible to show on screen and too heavy to save. This means that we reduce the time resolution of movies by 'baseacc'. Increase it if movies are shaky
    baseacc = 10
    
    ## convert and get useful data
    theta = np.arctan2(traj.vel[1], traj.vel[0]) * 180 / np.pi
    phi = traj.params['phi']
    
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # this "lines" is a PathCollection, that contains the points for the tail of all individuals. Here we initialize it 
    lines = ax.scatter(
        *np.tile(traj.pos[..., 0][..., None], npts_in_tail).reshape(2, -1), 
        c=np.repeat(np.ones(traj.params['N']), npts_in_tail), 
        cmap='Greys', zorder=0, vmin=0, vmax=npts_in_tail
        )
    
    # this "line" is a PathCollection, that contains the signle point for the head (current position) of all individuals. Here we initialize it 
    line,  = ax.plot(*traj.pos[..., 0], 'ko', mfc='w')
    
    ## Initialize Collections to draw circles for Field of View or Arrows for directions, if needed 
    if fov:
        Rvis = [mpatches.Wedge(center=xy, r=traj.params['R'],  theta2=th+phi/2, theta1=th-phi/2, ec='none', fc='k', alpha=0.05, zorder=0, animated=True) for xy, th in zip(traj.pos[..., 0].T, theta[..., 0].T)]
        Ra   = [mpatches.Wedge(center=xy, r=traj.params['Ra'], theta2=th+phi/2, theta1=th-phi/2, ec='k', fc='none', alpha=0.3,  zorder=0, animated=True) for xy, th in zip(traj.pos[..., 0].T, theta[..., 0].T)]
        
        for cRvis, cRa in zip(Rvis, Ra):
            ax.add_patch(cRvis)
            ax.add_patch(cRa)

    if draw_quiv:
        quiv = ax.quiver(*traj.pos[..., 0], *traj.fwall[..., 0], angles='xy', scale_units='xy', scale=1)

    # Initialize the figure
    ax.axis('equal')
    ax.set_xlim([traj.pos[0].min(), traj.pos[0].max()])
    ax.set_ylim([traj.pos[1].min(), traj.pos[1].max()])
    ax.set_xlim([-0.5, W+0.5])
    ax.set_ylim([-0.5, H+0.5])
    ax.add_patch(plt.Rectangle((0, 0), W, H, fc='none', ec='k'))
    ax.axis('scaled')
    fig.tight_layout()
    title = ax.text(0.5, 1, "", bbox={'facecolor':'w', 'alpha':1, 'pad':5},
                transform=ax.transAxes, ha="center")
    
    
    ## Function to update the animation at each timestep 
    def update(t):

        tau = acc * t * baseacc
    
        line.set_data(*traj.pos[..., tau])
        lines.set_offsets(traj.pos[..., tau-tail//npts_in_tail * npts_in_tail:tau:tail//npts_in_tail].reshape(2, -1).T)
        lines.set_sizes(np.tile(np.arange(npts_in_tail)**2, traj.params['N']) /npts_in_tail**2 * 15)  
        lines.set_array(np.tile(np.arange(npts_in_tail), traj.params['N'])) 

        if draw_quiv: 
            quiv.set_offsets(traj.pos[..., tau].T)
            quiv.set_UVC(*traj.fwall[..., tau]*10)

        if fov:
            for cRvis, cRa, xy, th in zip(Rvis, Ra, traj.pos[..., tau].T, theta[..., tau].T):
                cRvis.set_center(xy)
                cRa.set_center(xy)
                cRvis.set_theta2(th+phi/2)
                cRvis.set_theta1(th-phi/2)
                cRa.set_theta2(th+phi/2)
                cRa.set_theta1(th-phi/2)

        title.set_text(f"Time = {tau * dt:.1f} sec")

        all_lines = [line, lines, title]

        if draw_quiv: 
            all_lines = all_lines + [quiv]
        if fov: 
            all_lines = all_lines + Ra + Rvis

        return  all_lines

    #generate the animation 
    anim = animation.FuncAnimation(fig, update, interval=dt*1e3 * baseacc, frames=int(traj.pos.shape[-1]//(acc * baseacc)), blit=True)
    
    plt.show()

    if save:
        rawname = traj.get_rawname()
        anim.save(os.path.join(VIDEOPATH, f'{rawname}.mp4'))
    
    return anim


