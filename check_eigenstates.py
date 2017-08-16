from module_constants import *
import numpy as np
import module_functions as ion
import matplotlib.pyplot as plt
import matplotlib.animation as anim8


levels = np.array(range(1, 22, 4))
V0 = 5e3
frot = 5e2
(Etilde, psitilde, phi, cm) = ion.eigenvals_eigenstates(levels, V0, frot)
nTerms = len(cm[:, 0])
N = int((nTerms-1)/2)
m = range(-N, N+1)




w_ho = np.sqrt(2*h*V0/(M*r**2))

nSigma = 6


Lz_spread1 = ((2*M*r**2*h*V0)/hbar**2)**0.25
Lz_center = M*r**2 * 2*pi*frot/hbar
N_needed = np.select([levels<2*(2*h*V0)/(hbar*w_ho), levels>=2*(2*h*V0)/(hbar*w_ho)], [Lz_center + (np.sqrt(levels) + nSigma)*Lz_spread1, Lz_center + levels])






# fig1 = plt.figure(1)
# ax1 = plt.gca()
# line1a, = ax1.plot([], [])
# line1b, = ax1.plot([], [])
# line1c, = ax1.plot([], [])
# ax1.set_xlim(0, 2*np.pi)
# ax1.set_ylim(0, 1)
# def init1():
#     line1a.set_data([], [])
#     line1b.set_data([], [])
#     line1c.set_data([], [])
#     ax1.set_title('')
#     return line1a, line1b, line1c
# def animate1(i):
# 	level = levels[i]
# 	psitilde_ = psitilde[:, i]
# 	line1a.set_data(phi, psitilde_.real)
# 	line1b.set_data(phi, psitilde_.imag)
# 	line1c.set_data(phi, np.absolute(psitilde_)**2)
# 	ax1.set_title('level ' + str(level))
# 	return line1a, line1b, line1c, ax1
# ani1 = anim8.FuncAnimation(fig1, animate1, init_func=init1, frames=len(levels), interval=1000.0*10/len(levels), blit=False)




fig2 = plt.figure(2)
ax2 = plt.gca()
line2a, = ax2.plot([], [])
line2b, = ax2.plot([], [])
line2c, = ax2.plot([], [])
ax2.set_xlim(-N, N)
ax2.set_ylim(0, 1)
def init2():
    line2a.set_data([], [])
    line2b.set_data([], [])
    line2c.set_data([], [])
    ax2.set_title('')
    return line2a, line2b, line2c
def animate2(i):
	level = levels[i]
	c = cm[:, i]
	N_neededed = N_needed[i]
	line2a.set_data(m, np.absolute(c))
	line2b.set_data([ N_neededed,  N_neededed], [0, 1])
	line2c.set_data([-N_neededed, -N_neededed], [0, 1])
	ax2.set_title('level ' + str(level))
	return line2a, line2b, line2c, ax2
ani2 = anim8.FuncAnimation(fig2, animate2, init_func=init2, frames=len(levels), interval=1000.0*10/len(levels), blit=False)





# fig3 = plt.figure(3)
# ax3 = plt.gca()
# line3, = ax3.plot([], [])
# ax3.set_xlim(0, 2*np.pi)
# ax3.set_ylim(0, 4*V0)
# plt.plot(phi, 0.5*(V0*(np.cos(2*phi)+1)))
# def init3():
#     line3.set_data([], [])
#     ax3.set_title('')
#     return line3,
# def animate3(i):
# 	level = levels[i]
# 	E = Etilde[i]
# 	line3.set_data([0, 2*np.pi], [E, E])
# 	ax3.set_title('level ' + str(level))
# 	return line3, ax3
# ani3 = anim8.FuncAnimation(fig3, animate3, init_func=init3, frames=len(levels), interval=1000.0*10/len(levels), blit=False)



plt.show()