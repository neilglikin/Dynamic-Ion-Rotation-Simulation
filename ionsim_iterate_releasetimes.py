import module_functions as ion

V0i = 1e3
frotf = 1e3
initial_state = ([1, 2], [1j, 1])
# trel_vals = [50e-6, 100e-6, 200e-6, 500e-6, 1e-3, 2e-3, 5e-3]
# filename_ends = ['50u', '100u', '200u', '500u', '1m', '2m', '5m']
trel_vals = [10e-3]
filename_ends = ['10m']

for i in range(len(trel_vals)):
	trel = trel_vals[i]
	filename_end = filename_ends[i]
	filename = 'simulation_linrelease_residpot=0_rotframe_relonly_psi0=ground_1k,1k,' + filename_end

	print('Running simulation ' + filename)
	ion.runsimulation_rotframe(filename, V0i, frotf, trel, initial_state=initial_state)