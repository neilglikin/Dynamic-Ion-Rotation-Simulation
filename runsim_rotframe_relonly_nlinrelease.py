# File ionsim_runsimulation_rotframe_nlinrelease.py

"""  """

import numpy as np
import module_functions as ion



# Set parameters here
# There are 4 parameters: the initial potential height "V0i", the final rotation frequency "frotf", the release time "trel", and the time to continue the simulation after release "thold2"
# Also name the save file here
V0i    = 5e9
frotf  = 5e3
slowfactor = 0.5
initial_state = ([1, 2], [1j, 1])
savefile = 'simulation_nlinrelease_residpot=0_labframe_fullsim_psi0=ground_5G,5k,500m'

# Solve for time evolution
(c, psi, phi, tvals) = ion.time_evolution_rotframe_nlinrelease(V0i, frotf, slowfactor, initial_state=initial_state)
V0_vals    = ion.V0_fun_general_nlinrelease(tvals, V0i, slowfactor)
frot_vals  = frotf*np.ones_like(tvals)
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
input_parameters = {'V0i':V0i, 'frotf':frotf, 'tspin':0, 'thold1':0, 'trel':None, 'thold2':None}
abscissa_vectors = {'tvals':tvals, 'phi':phi, 'mvals':mvals}
potential_evolution = {'V0_vals':V0_vals, 'frot_vals':frot_vals, 'theta_vals':theta_vals}
wavefn_evolution = {'psi':psi, 'c':c}

print('Saving data...')

savedata = (input_parameters, abscissa_vectors, potential_evolution, wavefn_evolution)
np.save(savefile, savedata)