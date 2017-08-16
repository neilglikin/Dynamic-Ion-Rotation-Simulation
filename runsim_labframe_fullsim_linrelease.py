# File ionsim_runsimulation.py

""" Runs a simulation of single-ion spinup, hold, release, and hold; saves to a .npy file in the current directory """

import numpy as np
import module_functions as ion



# Set parameters here
# There are 6 parameters: the initial potential height "V0i", the final rotation frequency "frotf", the spinup time "tspin", the time to wait between spinup and release "thold1", the release time "trel", and the time to continue the simulation after release "thold2"
# Currently (NOT) set up so that the spin and release times are both equal ("tramp"), such that really there are 5 independent parameters to set
# Also name the save file here
V0i    = 50e3
frotf  = 4e3
tspin  = 800e-6
thold1 = 160e-6
trel   = 800e-6			# This line is where tspin=trel=tramp is (NOT) enforced
thold2 = 160e-6
initial_state = ([1, 2], [1j, 1])
savefile = 'simulation_linrelease_residpot=0_labframe_fullsim_psi0=ground_50k,4k,800u'

# Solve for time evolution
(c, psi, phi, tvals) = ion.time_evolution(V0i, frotf, tspin, thold1, trel, thold2, initial_state=initial_state)
(V0_vals, frot_vals, theta_vals) = ion.V0_frot_theta_fun_general(tvals, V0i, frotf, tspin, thold1, trel)

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
input_parameters = {'V0i':V0i, 'frotf':frotf, 'tspin':tspin, 'thold1':thold1, 'trel':trel, 'thold2':thold2}
abscissa_vectors = {'tvals':tvals, 'phi':phi, 'mvals':mvals}
potential_evolution = {'V0_vals':V0_vals, 'frot_vals':frot_vals, 'theta_vals':theta_vals}
wavefn_evolution = {'psi':psi, 'c':c}

print('Saving data...')

savedata = (input_parameters, abscissa_vectors, potential_evolution, wavefn_evolution)
np.save(savefile, savedata)