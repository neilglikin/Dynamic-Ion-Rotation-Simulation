import numpy as np
import module_functions as ion
import matplotlib.pyplot as plt
import time


nStates_toPlot = 100
frotf = 5e3
files = ['simulation_linrelease_residpot=0_rotframe_relonly_psi0=ground_50M,5k,10u.npy']
eigsavefile = '100eigenstates_101evenlyspacedsteps_V0i=50e6.npy'


(_, abscissa_vectors, potential_evolution, _) = np.load(files[0])
tvals   = abscissa_vectors['tvals']
V0_vals = potential_evolution['V0_vals']
levels = 1 + np.array(range(nStates_toPlot))
c_eig_all = []
for i in range(len(tvals)):
	print('Calculating eigenstates. ' + str((100.0*i)/len(tvals)) + '%...')
	V0 = V0_vals[i]
	(_, _, _, c_eig_all_current) = ion.eigenvals_eigenstates(levels, V0, frotf)
	c_eig_all.append(c_eig_all_current)
np.save(eigsavefile, c_eig_all)