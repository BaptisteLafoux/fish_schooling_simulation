import numpy as np 

############# TIME/SPACE DOMAINE ################
v_ini       = 1

Treal       = 1000 #in seconds

dt          = 1 / 50

T           = int(Treal // dt) + 1

W, H        = 30, 30

L           = min(W, H)

percentage_of_pts_tosave = 1

############# VIDEO INFORMATION ################
subsamp     = int(1 / (20 * dt))
acc_video   = int(1 / (10 * dt)) 

ncores      = 6

############# WALLS ################
gammawall = 5
delta     = 7

############# FILE INFORMATION ################
TEMPSAVEPATH    = 'temp'   # temp directory to save trash intermediate .nc files 
SAVEPATH        = 'output' #'z://numerical_simulation/output_cluster'  # final path for merged dataset
VIDEOPATH       = 'video' 

SAVEMETHOD      = 'timeseries' #['lastpoint', 'timeseries']
savevid         = False


############# SIMU PARAMETERS ################
nrep    = 1

taus    = [1]#np.geomspace(0.1, 10, 9)
Js      = [1]#np.geomspace(0.1, 10, 9)
Ns      = [50]
v0s     = [0.5, 1, 1.5, 2]
Ras     = [7]
As      = [1]#np.geomspace(0.1, 10, 9)
epss    = [0.05]
bs      = [0, 0.5]
phis = [300]
Rs = [5]

filename        = f'tau={taus[0] if len(taus)==1 else (str(min(taus)) + "-" + str(max(taus)))}_J={Js[0] if len(Js)==1 else (str(min(Js)) + "-" + str(max(Js)))}_N={Ns[0] if len(Ns)==1 else (str(min(Ns)) + "-" + str(max(Ns)))}_v0={v0s[0] if len(v0s)==1 else (str(min(v0s)) + "-" + str(max(v0s)))}_Ra={Ras[0] if len(Ras)==1 else (str(min(Ras)) + "-" + str(max(Ras)))}_a={As[0] if len(As)==1 else (str(min(As)) + "-" + str(max(As)))}_eps={epss[0] if len(epss)==1 else (str(min(epss)) + "-" + str(max(epss)))}_phi={phis[0] if len(phis)==1 else (str(min(phis)) + "-" + str(max(phis)))}_b={bs[0] if len(bs)==1 else (str(min(bs)) + "-" + str(max(bs)))}_R={Rs[0] if len(Rs)==1 else (str(min(Rs)) + "-" + str(max(Rs)))}'    # .nc file to create in SAVEPATH