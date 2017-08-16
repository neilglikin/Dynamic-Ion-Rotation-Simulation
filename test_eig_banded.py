from module_constants import *
import numpy as np
import scipy.linalg as scla
from timeit import default_timer as timer
import matplotlib.pyplot as plt



def construct_Hamiltonian(V0, frot, nTerms):	
	N = int((nTerms-1)/2)
	mvals = np.array(range(-N, N+1))

	H = np.zeros((nTerms, nTerms))								# Initialize the Hamiltonian matrix in Fourier basis
	diag_elems = (hbar/(2*pi))/(2*M*r**2)*(mvals**2) - frot*mvals + V0/2
	H += np.diag(diag_elems)									# The diagonal elements are the array diag_elems, constructed in the line above
	H += np.eye(nTerms, nTerms,  2) * V0/4						# The off-diagonal elements are constants, and are only nonzero on the second diagonals above and below the main diagonal.
	H += np.eye(nTerms, nTerms, -2) * V0/4

	return H


nTerms = 51
H = construct_Hamiltonian(1e4, 0, nTerms)
levels = np.array(range(1, 11))

# start_eigh = timer()
# E_eigh_all, c_eigh_all = np.linalg.eigh(H)
# end_eigh = timer()
# print(end_eigh - start_eigh)

H_band = np.zeros((3, nTerms))
H_band[0, 2:] = np.diagonal(H, offset=2)
H_band[2, :]  = np.diagonal(H)
start_band = timer()
E_band_all, c_band_all = scla.eig_banded(H_band)
# E_band_all, c_band_all = scla.eig_banded(H_band, select='i', select_range=(0, 9))
end_band = timer()
print(end_band - start_band)

# E_eigh = E_eigh_all[levels-1]
# c_eigh = c_eigh_all[levels-1]

E_band = E_band_all[levels-1]
c_band = c_band_all[levels-1]

# plt.plot(E_eigh)
plt.plot(E_band)
plt.show()