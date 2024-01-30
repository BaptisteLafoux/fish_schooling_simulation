from constants import *
import tqdm, glob, os 
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

########################
## Utils for merging a stack of .nc file after a batch run 

def merge_ds():

    dspaths = glob.glob(f'{TEMPSAVEPATH}/*.nc')
    dslist = []
    print(f'\n##############\n\t> Saving a merged file in {SAVEPATH} from {len(dspaths)} file located in {TEMPSAVEPATH}\n')

    for file in tqdm.tqdm(dspaths):
        ds_temp = xr.load_dataset(file)
        dslist.append(ds_temp)

    ds = xr.combine_by_coords(dslist, combine_attrs='drop_conflicts') 
    ds.to_netcdf(os.path.join(SAVEPATH, f'{filename}.nc'))

    for file in os.scandir(TEMPSAVEPATH):
        os.remove(file.path)

########################
## Utils for saving trajectories 

def get_rawname(params):
    return '_'.join([f'{key}={params[key]:06.3f}' for key in params.keys()])

def get_rep_number(params): 
    rep = 0 

    while os.path.exists(os.path.join(TEMPSAVEPATH, f'{get_rawname(params)}_{rep}.nc')):
        rep +=1

    return rep 

def get_filename(params):
    rep = get_rep_number(params)
    return f'{get_rawname(params)}_{rep}.nc'

def get_coord_dict(params): 

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


def animate(traj, acc=5, save=False, draw_quiv=False, fov=False, tailsec=0.5):

    npts_in_tail=12

    tail = int(tailsec / dt) 
    fig, ax = plt.subplots(figsize=(4, 4))

    theta = np.arctan2(traj.vel[1], traj.vel[0]) * 180 / np.pi
    phi = traj.params['phi']

    #lines, = ax.plot(*traj.pos[..., 0], 'k.', ms=1, mfc='0.4', zorder=0)
    lines = ax.scatter(*traj.pos[..., 0], c=np.ones(traj.params['N']), cmap='Greys', zorder=0, vmin=0, vmax=npts_in_tail)
    line,  = ax.plot(*traj.pos[..., 0], 'ko', mfc='w')
    if fov:
        Rvis = [mpatches.Wedge(center=xy, r=traj.params['R'],  theta2=th+phi/2, theta1=th-phi/2, ec='none', fc='k', alpha=0.05, zorder=0, animated=True) for xy, th in zip(traj.pos[..., 0].T, theta[..., 0].T)]
        Ra   = [mpatches.Wedge(center=xy, r=traj.params['Ra'], theta2=th+phi/2, theta1=th-phi/2, ec='k', fc='none', alpha=0.3,  zorder=0, animated=True) for xy, th in zip(traj.pos[..., 0].T, theta[..., 0].T)]
        
        for cRvis, cRa in zip(Rvis, Ra):
            ax.add_patch(cRvis)
            ax.add_patch(cRa)

    if draw_quiv:
        quiv = ax.quiver(*traj.pos[..., 0], *traj.fwall[..., 0], angles='xy', scale_units='xy', scale=1)

    ax.axis('equal')
    
    ax.set_xlim([traj.pos[0].min(), traj.pos[0].max()])
    ax.set_ylim([traj.pos[1].min(), traj.pos[1].max()])

    ax.set_xlim([-0.5, W+0.5])
    ax.set_ylim([-0.5, H+0.5])
    ax.add_patch(plt.Rectangle((0, 0), W, H, fc='none', ec='k'))

    # ax.axis('off')
    ax.axis('scaled')
    
    fig.tight_layout()

    

    title = ax.text(0.5, 1, "", bbox={'facecolor':'w', 'alpha':1, 'pad':5},
                transform=ax.transAxes, ha="center")
    
    baseacc = 5

    def update(t):

        tau = acc * t * baseacc
    
        line.set_data(*traj.pos[..., tau])
        #lines.set_data(*traj.pos[..., tau-tail:tau])

        lines.set_offsets(traj.pos[..., tau-tail:tau:tail//npts_in_tail].reshape(2, -1).T)

        lines.set_sizes(np.tile(np.arange(npts_in_tail)**2, traj.params['N'])/npts_in_tail**2 * 15)  
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

        if draw_quiv: all_lines = all_lines + [quiv]
        if fov: all_lines = all_lines + Ra + Rvis

        return  all_lines

    anim = animation.FuncAnimation(fig, update, interval=dt*1e3 * baseacc, frames=int(traj.pos.shape[-1]//(acc * baseacc)), blit=True)
    plt.show()

    if save:
        rawname = traj.get_rawname()
        anim.save(os.path.join(VIDEOPATH, f'{rawname}.mp4'))
    
    return anim


