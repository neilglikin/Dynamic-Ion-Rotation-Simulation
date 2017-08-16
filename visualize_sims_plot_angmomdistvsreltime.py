# File ionsim_plot_cvsreltime.py

""" Takes a set of simulation files whose parameters differ only in release time, and produces a heat plot of amplitude distribution in angular momentum space versus release time """

import matplotlib.pyplot as plt
import numpy as np


# Edit these parameters
# N = 500								# Maximum value of m (positive and negative) to plot
# rel_times = [[50e-6,  '50u'],		# List of release times; column 0 is for the x-axis of the plot; column 1 is for the filenames. They should match up row by row.
# 			 [100e-6, '100u'],
# 			 [200e-6, '200u'],
# 			 [500e-6, '500u'],
# 			 [1e-3,   '1m'],
# 			 [2e-3,   '2m']]
# x_append = 5e-3						# This extra release time will be added to the x-axis; otherwise the last one doesn't actually get plotted
# loadfile_base = 'time_evolution_rotframe_5G,5k,'
# loadfile_end = ''
N = 30								# Maximum value of m (positive and negative) to plot
rel_times = [[10e-3,   '10m'],
	         [30e-3,  '30m'],		# List of release times; column 0 is for the x-axis of the plot; column 1 is for the filenames. They should match up row by row.
			 [100e-3, '100m'],
			 [300e-3, '300m'],
			 [1,      '1'],
			 [3,      '3']]
x_append = 10						# This extra release time will be added to the x-axis; otherwise the last one doesn't actually get plotted
loadfile_base = 'simulation_linrelease_residpot=0_rotframe_relonly_psi0=ground_1k,200,'
loadfile_end = ''
verticalline = 0.81


# Set x and y (release times and values of m, respectively) for the plot, and initialize z (will become the magnitude squared of the amplitude |c_m|^2)
x_1Darr = [rel_time[0] for rel_time in rel_times]
y_1Darr = range(-N, N+1)
z = np.zeros((len(y_1Darr), len(x_1Darr)+1))
m_std = np.zeros_like(x_1Darr)

# Set z values and standard deviations of m distributions
for i in range(len(x_1Darr)):
	loadfile = loadfile_base + rel_times[i][1] + loadfile_end + '.npy'
	c = np.load(loadfile)[3]['c'][:, -1]						# This picks out the values of c for the end of the simulation (i.e. after release)

	N_file = int((len(c)-1)/2)
	c_index_touse = range((N_file-N), (N_file-N) + 2*N + 1)		# This is a list of indices from c to pick out for plotting. This will ensure that the number of m values plotted for all simulations is the same (equal to 2*N+1), even if a different set of m values was used in the different simulations
	c = c[c_index_touse]
	
	z[:, i] = np.absolute(c)**2

	mvals = range(-N, N+1)
	m_mean = sum(np.absolute(c)**2 * mvals)
	m_std[i] = np.sqrt(sum(np.absolute(c)**2 * (mvals-m_mean)**2))

x_1Darr.append(x_append)
x, y = np.meshgrid(x_1Darr, y_1Darr)

# Plot!
plt.figure(1)
ax1 = plt.gca()
ax1.set_xscale('log')
ax1.set_xlabel('Release time (s)')
ax1.set_ylabel('Angular momentum quantum number m')
plt.pcolormesh(x, y, z, cmap='jet')
plt.colorbar(label='Magnitude squared of amplitude, |c|^2')
plt.plot([verticalline, verticalline], [-N, N], color='w', linestyle='--')

# plt.figure(2)
# ax2 = plt.gca()
# ax2.set_xscale('log')
# ax2.set_xlabel('Release time (s)')
# ax2.set_ylabel('Std. deviation of ang. momentum distribution')
# plt.plot(x_1Darr[:-1], m_std, linestyle='None', marker='.', markersize=15)

plt.show()