import numpy as np
import module_functions as ion
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time






# loadfile = 'time_evolution_rotframe_5G,5k,1m.npy'
# nStates_toPlot = 50

# (input_parameters, abscissa_vectors, potential_evolution, wavefn_evolution) = np.load(loadfile)
# c_tevol_all = wavefn_evolution['c']
# tvals   	= abscissa_vectors['tvals']
# mvals   	= abscissa_vectors['mvals']
# V0_vals 	= potential_evolution['V0_vals']
# frotf  		= input_parameters['frotf']

# levels = 1 + np.array(range(nStates_toPlot))
# x = tvals
# y = levels
# (x, y) = np.meshgrid(x, y)
# z = np.zeros_like(x)



# # plt.ion()
# # line1, = plt.plot([], [])
# # line2, = plt.plot([], [])
# # ax = plt.gca()
# # ax.set_xlim(0, 100)
# # ax.set_ylim(-1, 1)

# for i in range(len(tvals)):
# 	print(str((100.0*i)/len(tvals)) + '%...')

# 	c_tevol = c_tevol_all[:, i]

# 	V0 = V0_vals[i]
# 	(_, _, _, c_eig_all) = ion.eigenvals_eigenstates(levels, V0, frotf)

# 	for j in range(nStates_toPlot):
# 		c_eig = c_eig_all[:, j]

# 		N_tevol = (len(c_tevol)-1)/2
# 		N_eig   = (len(c_eig)-1)/2
# 		if N_tevol > N_eig:
# 			N_diff  = N_tevol - N_eig
# 			c_tevol = c_tevol[N_diff:-N_diff]
# 		elif N_eig > N_tevol:
# 			N_diff  = N_eig - N_tevol
# 			c_eig = c_eig[N_diff:-N_diff]

# 		z[j, i] = np.absolute(np.sum(np.multiply(c_tevol, np.conj(c_eig))))**2

# 		ecks = range(len(c_tevol))

# 		# line1.set_data(ecks, c_tevol.real)
# 		# line2.set_data(ecks, c_eig.real)
# 		# plt.draw()
# 		# plt.title('t = ' + str(tvals[i]) + ', level ' + str(levels[j]) + ', inner product = ' + str(z[j,i]))
# 		# plt.pause(1)
		




releasetime = '1 ms'
(x, y, z) = np.load('eigstatesvstime_linrelease_residpot=0_rotframe_relonly_psi0=ground_5G,5k,1m.npy')
x = x[0, :]
y = y[:, 0]

nTimes  = len(x)
nLevels = len(y)
dx = x[1] - x[0]

x = np.append(x, [x[-1]+dx])
y = np.append(y, nLevels+1)
(x, y) = np.meshgrid(x, y)
z = np.append(z, np.zeros((nLevels, 1)), axis=1)
z = np.append(z, np.zeros((1, nTimes+1)), axis=0)

plt.figure(1)
ax1 = plt.gca()
ax1.set_title(releasetime + ' Release Time')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Eigenstate')
plt.pcolormesh(x, y, z, cmap='jet', norm=colors.LogNorm())
# plt.pcolormesh(x, y, z, cmap='jet')
plt.colorbar(label='Occupation')
plt.show()