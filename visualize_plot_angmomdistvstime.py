import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


file = 'simulation_nlinrelease_residpot=0_rotframe_relonly_psi0=ground_5G,5k,1.npy'
(_, abscissa_vectors, potential_evolution, wavefn_evolution) = np.load(file)
tvals      = abscissa_vectors['tvals']
mvals      = abscissa_vectors['mvals']
c          = wavefn_evolution['c']



# dt = tvals[1] - tvals[0]
# tvals = np.append(tvals, dt)

(x, y) = np.meshgrid(tvals, mvals)
z = np.abs(c)**2





plt.figure(1)
ax1 = plt.gca()
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angular momentum quantum number m')
plt.pcolormesh(x, y, z, cmap='jet', vmin=0.01, vmax=0.5)
plt.colorbar(label='Magnitude squared of amplitude, |c|^2')

verticalline = 0.0113
N = mvals[-1]
plt.plot([verticalline, verticalline], [-N, N], color='w', linestyle='--')

plt.show()