# File ionsim_checksimparams.py

""" Loads a simulation saved by runsim_labframe_fullsim_linrelease.py and prints the parameters used """

import numpy as np
loadfile = 'simulation_linrelease_residpot=0_rotframe_relonly_psi0=ground_5k,200,100m.npy'
(input_parameters, abscissa_vectors, _, _) = np.load(loadfile)
print(input_parameters)
print('nTimes = ' + str(len(abscissa_vectors['tvals'])))
print('nTerms = ' + str(len(abscissa_vectors['mvals'])))