# File ionsim_runsimulation_rotframe.py

""""""

import numpy as np
import module_functions as ion



# Set parameters here
# There are 4 parameters: the initial potential height "V0i", the final rotation frequency "frotf", the release time "trel", and the time to continue the simulation after release "thold2"
# Also name the save file here
V1     = 2e3
phi1   = 0
V0i    = 10e3
frotf  = 763
trel   = 200e-6
thold2 = 600e-6
# initial_state = ([1, 2], [1j, 1])
# nTimes = 101
savefile = 'simulation_linrelease_residpot=2k_rotframe_relonly_psi0=ground_10k,763,200u'

# Solve for time evolution
(c, psi, phi, tvals) = ion.time_evolution_rotframe_residpot(V0i, V1, phi1, frotf, trel, thold2)#, initial_state=initial_state, nTimes=nTimes)
(V0_vals, frot_vals, _) = ion.V0_frot_theta_fun_general(tvals, V0i, frotf, 0, 0, trel)
theta_vals = np.zeros_like(tvals)

# Calculate mvals (which is the only value to be saved which is not explicitly returned by the above functions)
(nTerms, nTimes) = c.shape
N = (nTerms-1)/2
mvals = np.array(range(-N, N+1))

# Set up save data as a tuple of 4 dictionaries:
# input_parameters: holds parameters entered under the above "Set parameters here" code block. Each is a scalar value.
# abscissa_vectors: 1-D arrays of time values "tvals", spatial angular coordinate values "phi", and m (Fourier coefficient index) values "mvals"
# potential_evolution: 1-D arrays of length len(tvals) describing the potential height "V0_vals", rotation frequency "frot_vals", and potential phase offset "theta_vals" as functions of time
# wavefn_evolution: 2-dimensional matrices "psi" and "c" describing the wavefunction's time-evolution.
#                   "psi" is a (len(phi) by len(tvals)) matrix, with rows corresponding to the coordinate (values of "phi") and columns corresponding to times (values of "tvals")
#                   "c" is a (len(mvals) by len(tvals)) matrix, with rows corresponding to the Fourier coefficient index (values of "mvals") and columns corresponding to times (values of "tvals")
input_parameters = {'V0i':V0i, 'frotf':frotf, 'tspin':0, 'thold1':0, 'trel':trel, 'thold2':thold2, 'V1':V1, 'phi1':phi1}
abscissa_vectors = {'tvals':tvals, 'phi':phi, 'mvals':mvals}
potential_evolution = {'V0_vals':V0_vals, 'frot_vals':frot_vals, 'theta_vals':theta_vals}
wavefn_evolution = {'psi':psi, 'c':c}

print('Saving data...')

savedata = (input_parameters, abscissa_vectors, potential_evolution, wavefn_evolution)
np.save(savefile, savedata)