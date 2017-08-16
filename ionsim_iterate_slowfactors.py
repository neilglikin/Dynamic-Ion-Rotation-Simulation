import module_functions as ion

V0i = 5e9
frotf = 5000
initial_state = ([1, 2], [1j, 1])
slowfactor_vals = [1, 3, 10, 30, 100]
filename_ends = ['1', '3', '10', '30', '100']

for i in range(len(slowfactor_vals)):
	slowfactor = slowfactor_vals[i]
	filename_end = filename_ends[i]
	filename = 'time_evolution_5G,5k,' + filename_end + '_nlinrelease'

	print('Running simulation ' + filename)
	ion.runsimulation_nlinrelease(filename, V0i, frotf, slowfactor, initial_state=initial_state)