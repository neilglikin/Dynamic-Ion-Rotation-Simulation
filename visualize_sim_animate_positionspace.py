# File ionsim_animatewavefn.py

""" Loads a simulation saved by runsim_labframe_fullsim_linrelease.py and animates the wavefunction evolving in time """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim8



# Set file to load and how long animation should last
loadfile = 'simulation_linrelease_residpot=0_labframe_fullsim_psi0=ground_50k,1k,5m.npy'
totalAnimTime = 25      # seconds
save = False
savefile = loadfile[:-4] + '_wavefn.gif'
maxnFrames = 151

# Extract saved dictionaries from loadfile and assign the relevant contents to variables here
(_, abscissa_vectors, potential_evolution, wavefn_evolution) = np.load(loadfile)
tvals      = abscissa_vectors['tvals']
phi        = abscissa_vectors['phi']
V0_vals    = potential_evolution['V0_vals']
frot_vals  = potential_evolution['frot_vals']
theta_vals = potential_evolution['theta_vals']
psi        = wavefn_evolution['psi']

# Find where axes should be set
psir_max = np.amax(np.absolute(psi.real))
psii_max = np.amax(np.absolute(psi.imag))
psi2_max = np.amax(np.absolute(psi)**2)
y1max = np.ceil(max(psir_max, psii_max, psi2_max))
y2max = V0_vals[0]

# Animate!
fig = plt.figure()
# Create two axes with labels and limits; ax1 to scale for psi and ax2 to scale for V
ax1 = plt.axes(xlim=(0, 2*np.pi),
               ylim=(-y1max, y1max),
               xlabel='phi',
               ylabel='psi')
ax2 = ax1.twinx()
ax2.set_xlim((0, 2*np.pi))
ax2.set_ylim((-y2max, y2max))
ax2.set_ylabel('V')
# Define line and text handles
line1, = ax1.plot([], [], label='Re psi',  color='b')
line2, = ax1.plot([], [], label='Im psi',  color='r')
line3, = ax1.plot([], [], label='|psi|^2', color='g')
line4, = ax2.plot([], [], label='V',       color='k')
txt_time = ax1.text(0.3, -0.7*y1max, '')
txt_V0   = ax1.text(0.3, -0.8*y1max, '')
txt_frot = ax1.text(0.3, -0.9*y1max, '')
# Define legend
plt.legend(handles=[line1, line2, line3, line4], loc=4)
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

# Define initalize and animation functions for FuncAmimation
def init():
    # Initialize lines to have no data
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    return line1, line2, line3, line4
def animate(i):
    # Sets data to plot and text to display for the ith frame
    global phi, psi, tvals, V0_vals, frot_vals, theta_vals, framelist
    
    # i = 74

    i = framelist[i]
    t, V0, frot, theta = tvals[i], V0_vals[i], frot_vals[i], theta_vals[i]
    line1.set_data(phi, psi.real[:,i])
    line2.set_data(phi, psi.imag[:,i])
    line3.set_data(phi, np.absolute(psi[:,i])**2)
    line4.set_data(phi, V0/2*(np.cos(2*(phi-theta))+1))
    txt_time.set_text('t = '    + str(t))
    txt_V0.  set_text('V0 = '   + str(V0))
    txt_frot.set_text('frot = ' + str(frot))
    return line1, line2, line3, line4, txt_time, txt_V0, txt_frot
ani = anim8.FuncAnimation(fig, animate, init_func=init, frames=nFrames, interval=1000.0*totalAnimTime/nFrames, blit=True)
plt.show()

if save:
    print('Saving .gif...')
    ani.save(savefile, dpi=80, writer='imagemagick')