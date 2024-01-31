"""
Created on Wed Jan 31 09:57:22 2023

Provides constants used in the numerical simulation and arrays of parameteres used to generate the lists of Simulation objects that are run in parallel in the batch script ('main.py')

The script is not intended to be directly executed; it a module that is used by other files
@author: BaptisteLafoux
"""

############# TIME/SPACE DOMAINE ################
# initial velocity norm of all fish 
v_ini       = 1

# duration of simulation in physical time
Treal       = 1000 #in seconds

# timestep in physical time
dt          = 1 / 50 #in second

# number of timesteps in the simulation 
T           = int(Treal // dt) + 1

# width and height of the swimming arena
W, H        = 30, 30
L           = max(W, H)
# percentage from 0 to 1 - fraction of the total timesteps saved after the simualtion 
percentage_of_pts_tosave = 1

############# VIDEO INFORMATION ################

# subsampling of the video to avoid mp4 files that are too heavy (different from acceleration)
subsamp     = int(1 / (20 * dt))
# number of parallel processors to use 
ncores      = 6

############# WALLS ################
# intensity of the wall force 
gammawall = 5
# characteristic range of the wall force 
delta     = 7

############# FILE INFORMATION ################
TEMPSAVEPATH    = 'temp'   # temp directory to save trash intermediate .nc files 
SAVEPATH        = 'output' #'z://numerical_simulation/output_cluster'  # final path for merged dataset
VIDEOPATH       = 'video' 

SAVEMETHOD      = 'timeseries' #['lastpoint', 'timeseries']
savevid         = False

############# SIMULATION PARAMETERS ################

# number of replicates of the same simulation with same parameters (but different random inital condition)
nrep    = 1 

# characteristic time of the auto-propulsion force 
taus    = [1]#np.geomspace(0.1, 10, 9)

# intensity of alignement force 
Js      = [1]#np.geomspace(0.1, 10, 9)

# number of fish 
Ns      = [50]

# (average) taget velocity 
v0s     = [0.5, 1, 1.5, 2]

# alignment radii 
Ras     = [7]

# intensity of attraction-repulsion force 
As      = [1]#np.geomspace(0.1, 10, 9)

# positive, noise intensity. The random force is generated from a 2D normal law centered in 0 and of width "eps"
epss    = [0.05]

# positive. Dispertion of the target velocity of individuals. if b=0, all individuals have the same target speed 
bs      = [0, 0.5] 

# angle of vision in degrees 
phis    = [300] 

# distance of vision 
Rs      = [5]

# the name of the .nc file to create in SAVEPATH after a batch 
batch_filename        = f'tau={taus[0] if len(taus)==1 else (str(min(taus)) + "-" + str(max(taus)))}_J={Js[0] if len(Js)==1 else (str(min(Js)) + "-" + str(max(Js)))}_N={Ns[0] if len(Ns)==1 else (str(min(Ns)) + "-" + str(max(Ns)))}_v0={v0s[0] if len(v0s)==1 else (str(min(v0s)) + "-" + str(max(v0s)))}_Ra={Ras[0] if len(Ras)==1 else (str(min(Ras)) + "-" + str(max(Ras)))}_a={As[0] if len(As)==1 else (str(min(As)) + "-" + str(max(As)))}_eps={epss[0] if len(epss)==1 else (str(min(epss)) + "-" + str(max(epss)))}_phi={phis[0] if len(phis)==1 else (str(min(phis)) + "-" + str(max(phis)))}_b={bs[0] if len(bs)==1 else (str(min(bs)) + "-" + str(max(bs)))}_R={Rs[0] if len(Rs)==1 else (str(min(Rs)) + "-" + str(max(Rs)))}'    