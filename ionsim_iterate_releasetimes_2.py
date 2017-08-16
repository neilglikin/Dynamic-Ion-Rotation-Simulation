import module_functions as ion

V0i = 5e9
frotf = 5000
initial_state = ([3, 4], [1j, 1])
# trel_vals = [50e-6, 100e-6, 200e-6, 500e-6, 1e-3, 2e-3, 5e-3]
# filename_ends = ['50u', '100u', '200u', '500u', '1m', '2m', '5m']
trel_vals = [2e-3, 5e-3]
filename_ends = ['2m', '5m']

for i in range(len(trel_vals)):
	trel = trel_vals[i]
	filename_end = filename_ends[i]
	filename = 'time_evolution_rotframe_5G,5k,' + filename_end + '_firstexcited'

	print('Running simulation ' + filename)
	ion.runsimulation_rotframe(filename, V0i, frotf, trel, initial_state=initial_state)