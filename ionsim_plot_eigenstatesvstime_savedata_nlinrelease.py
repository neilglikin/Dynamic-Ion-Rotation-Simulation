import numpy as np
import module_functions as ion
import matplotlib.pyplot as plt
import time





def runthis(loadfile, nStates_toPlot, eigsavefile):
	print('Running ' + loadfile + '...')

	(input_parameters, abscissa_vectors, potential_evolution, wavefn_evolution) = np.load(loadfile)
	c_tevol_all = wavefn_evolution['c']
	tvals   	= abscissa_vectors['tvals']
	mvals   	= abscissa_vectors['mvals']
	V0_vals 	= potential_evolution['V0_vals']
	frotf  		= input_parameters['frotf']

	levels = 1 + np.array(range(nStates_toPlot))
	x = tvals
	y = levels
	(x, y) = np.meshgrid(x, y)
	z = np.zeros_like(x)



	# plt.ion()
	# line1, = plt.plot([], [])
	# line2, = plt.plot([], [])
	# ax = plt.gca()
	# ax.set_xlim(0, 100)
	# ax.set_ylim(-1, 1)




	c_eig_all_all = np.load(eigsavefile)

	for i in range(len(tvals)):
		print(str((100.0*i)/len(tvals)) + '%...')

		c_tevol = c_tevol_all[:, i]

		V0 = V0_vals[i]
		# (_, _, _, c_eig_all) = ion.eigenvals_eigenstates(levels, V0, frotf)
		c_eig_all = c_eig_all_all[i]

		for j in range(nStates_toPlot):
			c_eig = c_eig_all[:, j]

			N_tevol = (len(c_tevol)-1)/2
			N_eig   = (len(c_eig)-1)/2
			if N_tevol > N_eig:
				N_diff  = N_tevol - N_eig
				c_tevol = c_tevol[N_diff:-N_diff]
			elif N_eig > N_tevol:
				N_diff  = N_eig - N_tevol
				c_eig = c_eig[N_diff:-N_diff]

			z[j, i] = np.absolute(np.sum(np.multiply(c_tevol, np.conj(c_eig))))**2

			# ecks = range(len(c_tevol))
			# line1.set_data(ecks, c_tevol.real)
			# line2.set_data(ecks, c_eig.real)
			# plt.draw()
			# plt.title('t = ' + str(tvals[i]) + ', level ' + str(levels[j]) + ', inner product = ' + str(z[j,i]))
			# plt.pause(1)

	savefile = loadfile[:-4] + '_eigstatesvstime'
	np.save(savefile, (x, y, z))






nStates_toPlot = 25
frotf = 5e3
files = ['time_evolution_5G,5k,1_nlinrelease.npy',
		 'time_evolution_5G,5k,3_nlinrelease.npy']
eigsavefile = '100eigenstates_101nonlinsteps_V0i=5e9.npy'


(_, _, potential_evolution, _) = np.load(files[0])
V0_vals = potential_evolution['V0_vals']
levels = 1 + np.array(range(nStates_toPlot))
c_eig_all = []
for i in range(len(V0_vals)):
	print('Calculating eigenstates. ' + str((100.0*i)/len(V0_vals)) + '%...')
	V0 = V0_vals[i]
	(_, _, _, c_eig_all_current) = ion.eigenvals_eigenstates(levels, V0, frotf)
	c_eig_all.append(c_eig_all_current)
np.save(eigsavefile, c_eig_all)




# for file in files:
# 	runthis(file, nStates_toPlot, eigsavefile)