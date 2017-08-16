# File ionsim_plotangmomvstime.py

""" Loads a simulation saved by runsim_labframe_fullsim_linrelease.py and plots the expectation value of angular momentum vs. time """

from module_constants import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim8



# Set file to load
loadfile = 'time_evolution_5k,200,50m_test.npy'

# Extract saved dictionaries from loadfile and assign the relevant contents to variables here
(input_parameters, abscissa_vectors, potential_evolution, wavefn_evolution) = np.load(loadfile)
tspin     = input_parameters['tspin']
thold1    = input_parameters['thold1']
trel      = input_parameters['trel']
tvals     = abscissa_vectors['tvals']
mvals     = abscissa_vectors['mvals']
frot_vals = potential_evolution['frot_vals']
c         = wavefn_evolution['c']

# Construct 1-D array of angular momentum (Lz) expectation values vs. time
nTimes = len(tvals)
Lz_exp = np.zeros(nTimes)
for i in range(nTimes):
	Lz_exp[i] = sum(mvals*np.absolute(c[:,i])**2)

# Assign new time constants for convenience
tsh = tspin+thold1
tshr = tspin+thold1+trel

# Create plot
fig = plt.figure()
line1, = plt.plot(1000*tvals, M*r**2*2*pi*frot_vals/hbar, label='I*omega_rot', color='k', linestyle='-', linewidth=1)
line2, = plt.plot(1000*tvals, Lz_exp,                     label='<L_z>')
plt.legend(handles=[line2, line1], loc=4)
# Draw 3 dashed vertical lines to show (1) end of spinup, (2) beginning of release, and (3) end of release
axes = plt.gca()
(ymin, ymax) = axes.get_ylim()
axes.plot([1000*tspin, 1000*tspin], [ymin, ymax], color='k', linestyle='--', linewidth=1)
axes.plot([1000*tsh,   1000*tsh],   [ymin, ymax], color='k', linestyle='--', linewidth=1)
axes.plot([1000*tshr,  1000*tshr],  [ymin, ymax], color='k', linestyle='--', linewidth=1)
# Also draw the 
axes.set_ylim(ymin, ymax)	# Reset y-axis limits back to what they were before the vertical lines were drawn, as this will have automatically changed the limits
axes.set_xlabel('Time (ms)')	
axes.set_ylabel('Mean angular momentum (hbar)')
plt.show()