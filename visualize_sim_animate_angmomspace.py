# File ionsim_animatecoeffs.py

""" Loads a simulation saved by runsim_labframe_fullsim_linrelease.py and animates the Fourier coefficients evolving in time """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim8



# Set file to load and how long animation should last
loadfile = 'simulation_nlinrelease_residpot=0_rotframe_relonly_psi0=ground_5G,5k,1.npy'
totalAnimTime = 7.5      # seconds
save = False
savefile = loadfile[:-4] + '_coeffs.gif'
maxnFrames = 101

# Extract saved dictionaries from loadfile and assign the relevant contents to variables here
(_, abscissa_vectors, potential_evolution, wavefn_evolution) = np.load(loadfile)
tvals     = abscissa_vectors['tvals']
mvals     = abscissa_vectors['mvals']
V0_vals   = potential_evolution['V0_vals']
frot_vals = potential_evolution['frot_vals']
c         = wavefn_evolution['c']


tvals = tvals[-152:]
V0_vals = V0_vals[-152:]
frot_vals = frot_vals[-152:]
c = c[:, -152:]



# Find where axis limits should be set
N = mvals[-1]
xmax = N
cr_max = np.amax(np.absolute(c.real))
ci_max = np.amax(np.absolute(c.imag))
c_max  = np.amax(np.absolute(c))
ymax = np.ceil(10*max(cr_max, ci_max, c_max))/10.0		# Round up to nearest 10ths place

# Animate!
fig = plt.figure()
# Create axis with limits and labels
ax1 = plt.axes(xlim=(-xmax, xmax),
	           ylim=(-ymax, ymax),
	           xlabel='m',
	           ylabel='c')
# Initialize lines and text handles
line1, = ax1.plot([], [], linestyle='-', marker='.', label='Re c', color='b')
line2, = ax1.plot([], [], linestyle='-', marker='.', label='Im c', color='r')
line3, = ax1.plot([], [], linestyle='-', marker='.', label='|c|',  color='g')
txt_time = ax1.text(-0.95*N, -0.7*ymax, '')
txt_V0   = ax1.text(-0.95*N, -0.8*ymax, '')
txt_frot = ax1.text(-0.95*N, -0.9*ymax, '')
# Define legend
plt.legend(handles=[line1, line2, line3], loc=4)
# This section of code makes sure that the number of frames in the animation does not exceed maxnFrames
# framelist is the list of indices in the list of times to make into frames in the animation. If nTimes is larger than maxnFrames, then maxnFrames evenly spaced indices will be selected out of range(nTimes)
nTimes = len(tvals)
if nTimes <= maxnFrames:
    nFrames = nTimes
    framelist = range(nTimes)
else:
    nFrames = maxnFrames
    framelist = np.zeros(maxnFrames, dtype='int')
    di = (nTimes-1.0)/(maxnFrames-1.0)
    for i in range(maxnFrames):
        framelist[i] = int(round(i*di))

# nFrames = 101
# Define initalize and animation functions for FuncAmimation
def init():
	# Initialize lines to have no data
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3
def animate(i):
	# Sets data to plot and text to display for the ith frame
	global tvals, V0_vals, frot_vals, mvals, c, framelist
	
	i = 0

	i = framelist[i]
	t, V0, frot = tvals[i], V0_vals[i], frot_vals[i]
	line1.set_data(mvals, c.real[:,i])
	line2.set_data(mvals, c.imag[:,i])
	line3.set_data(mvals, np.absolute(c[:,i]))
	txt_time.set_text('t = '    + str(t))
	txt_V0.  set_text('V0 = '   + str(V0))
	txt_frot.set_text('frot = ' + str(frot))
	return line1, line2, line3, txt_time, txt_V0, txt_frot
nTimes = len(tvals)
ani = anim8.FuncAnimation(fig, animate, init_func=init, frames=nFrames, interval=1000.0*totalAnimTime/nFrames, blit=True)
plt.show()

if save:
	print('Saving .gif...')
	ani.save(savefile, dpi=80, writer='imagemagick')