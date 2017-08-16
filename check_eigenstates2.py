from module_constants import *
import numpy as np
import module_functions as ion
import matplotlib.pyplot as plt
import matplotlib.animation as anim8



level = 52
frot = 0
V0s = np.linspace(0, 2e3, num=51)


ms = []
cs = []
Es = []
for i in range(len(V0s)):
	V0 = V0s[i]
	print(V0)
	(E, _, _, c) = ion.eigenvals_eigenstates(level, V0, frot)
	nTerms = len(c)
	N = int((nTerms-1)/2.0)
	m = range(-N, N+1)

	ms.append(m)
	cs.append(c)
	Es.append(E)
Nmax = np.max(np.max(ms))


fig1 = plt.figure(1)
ax1 = plt.gca()
line1, = ax1.plot([], [])
ax1.set_xlim(-Nmax, Nmax)
ax1.set_ylim(0, 1)
def init1():
    line1.set_data([], [])
    ax1.set_title('')
    return line1, ax1
def animate1(i):
	m = ms[i]
	c = cs[i]
	E = Es[i]
	V0 = V0s[i]
	line1.set_data(m, np.absolute(c))
	ax1.set_title('V0 = ' + str(int(round(V0))) + ', E = ' + str(int(round(E))))
	return line1, ax1
ani1 = anim8.FuncAnimation(fig1, animate1, init_func=init1, frames=len(V0s), interval=1000.0*10/len(V0s), blit=False)

plt.show()