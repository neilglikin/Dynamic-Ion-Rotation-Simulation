import numpy as np




loadfile = 'simulation_nlinrelease_residpot=0_labframe_fullsim_psi0=ground_5k,500,1.npy'
(input_parameters, abscissa_vectors, potential_evolution, wavefn_evolution) = np.load(loadfile)
c_tevol_all = wavefn_evolution['c']
tvals   	= abscissa_vectors['tvals']
mvals   	= abscissa_vectors['mvals']
V0_vals 	= potential_evolution['V0_vals']
frotf  		= input_parameters['frotf']

for i in range(len(tvals)):
	c = c_tevol_all[:, i]
	print( np.sum(np.multiply(c, np.conj(c))) )
	print( np.sum(np.absolute(c)**2) )