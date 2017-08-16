import numpy as np
import module_functions as ion
import matplotlib.pyplot as plt

(_, abscissa_vectors, potential_evolution, _) = np.load('time_evolution_5k,500,5_nlinrelease.npy')
tvals     = abscissa_vectors['tvals']
V0_vals   = potential_evolution['V0_vals']

plt.plot(tvals, V0_vals, '.')
plt.show()