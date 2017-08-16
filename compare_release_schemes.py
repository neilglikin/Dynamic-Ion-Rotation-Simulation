import numpy as np
import matplotlib.pyplot as plt



t = np.linspace(0, 0.0113, 1000)
V0_lin  = 5e9 - (5e9/0.0113)*t
V0_nlin = 5e9/((1+1.66e6*t)**2)

fig = plt.figure()
ax  = plt.axes()
ax.plot(t, V0_lin)
ax.plot(t, V0_nlin)
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()