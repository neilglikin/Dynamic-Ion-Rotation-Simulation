# File ionsim_functions.py

""" Functions to be used in simulation files """

import numpy as np
import scipy.linalg as scla
from module_constants import *
from qutip import *

def calc_harmonic_freq(V0):
    """
    """
    return np.sqrt(2*h*V0/(M*r**2))/(2*pi)



def calc_nterms_needed(levels, V0, frot):
    """
    Determines how many Fourier terms to keep in order to accurately calculate the eigenvalue of the 'level'th energy level in a potential of height V0, rotating at frequency frot
    Based on an empirical fit
    Trustworthy for V0 up to 1e10, frot up to 5e3

    inputs:
    levels: array or scalar indicating which levels (with level 1 defined as that which has the lowest "pseudo-energy" eigenvalue); size of return will be equal to size of this
    V0: scalar; height of the potential (Hz)
    frot: scalar; rotational frequency (Hz)

    returns:
    nTerms_int: array or scalar (same shape as 'levels'); number of terms needed, as an integer
    """

    levels = np.array(levels, ndmin=1, copy=False)  # Convert to array if 'levels' is a scalar
    # a, b, c, d, e = 0.0036, 0.29, 5.4, 0.071, 19			# Fit parameters
    # nTerms_flt = 2*(V0**(a*levels+b) + (c*levels+(d*frot+e)))
    w_ho = np.sqrt(2*h*V0 / (M*r**2))
    Lz_spread1 = ((2*M*r**2 * h*V0) / hbar**2) ** 0.25
    Lz_center = M*r**2 * 2*pi*frot/hbar

    if w_ho != 0:
        nTerms_flt = 2 * np.select(
            [levels < 2*(2*h* V0)/(hbar*w_ho),             levels >= 2* (2*h*V0)/(hbar*w_ho)],
            [Lz_center + (np.sqrt(levels) + 8)*Lz_spread1, Lz_center + levels])
    else:
        nTerms_flt = Lz_center + levels

    nTerms_int = nTerms_flt.astype(int)  # Convert to integer
    addonemore = np.logical_not(nTerms_int % 2)  # If element is even, adds one to make it odd
    nTerms_int += addonemore
    if len(levels) == 1:
        nTerms_int = np.asscalar(nTerms_int)  # Convert back to scalar if input 'levels' was a scalar
    return nTerms_int



def eigenvals_eigenstates(levels, V0, frot, V0_fornterms=None, frot_fornterms=None):
    """
    Returns pseudoenergy eigenvalues, eigenvectors as Fourier components, and eigenfunctions in position space for the 'level'th level(s) of a transformed Hamiltonian with potential height V0 rotating at frequency frot

    inputs:
    levels: array or scalar indicating which levels (with level 1 defined as that which has the lowest "pseudo-energy" eigenvalue); size of returns will depend on size of this
    V0: scalar; height of the potential (Hz)
    frot: scalar; rotational frequency (Hz)
    V0_fornterms: scalar; a value of V0 which is used only to calculate nTerms (to be used in case, e.g., more terms are desired). Equal to V0 by default
    frot_fornterms: scalar; a value of frot which is used only to calculate nTerms (to be used in case, e.g., more terms are desired). Equal to frot by default

    returns:
    Etilde: array of pseudoenergy eigenvalues, one for each element of the input 'levels' (Hz)
    psitilde: array of normalized wavefunction values, with the first index being at phi=0 and the last at phi=2pi. If 'levels' was an array, columns are wavefunctions corresponding to each entry in 'levels' in order
    phi: 1-D array of phi values, from 0 to 2*pi
    cm: array of Fourier coefficients for psitilde, the basis functions being exp(i*m*phi). Rows (if 'levels' was an array) of cm contain values of m ranging from -N to +N for some integer N; different columns correspond to the different elements of the input 'levels'
    """

    # Calculate number of terms (size of Hilbert space) needed
    # If no value for V0/frot_fornterms was specified, then default to V0/frot
    if V0_fornterms is None:
        V0_fornterms = V0
    if frot_fornterms is None:
        frot_fornterms = frot
    levels = np.array(levels, ndmin=1, copy=False)  # Convert to array if 'levels' is a scalar
    maxlv  = np.max(levels)
    nTerms = calc_nterms_needed(maxlv, V0_fornterms,
                                frot_fornterms)  # Calculate how many terms to keep in the Fourier series based on largest value in 'levels'

    # Calculate N such that the range of integers from -N to +N contains nTerms numbers, as well as the array mvals containing those numbers
    N = int((nTerms-1)/2)
    mvals = np.array(range(-N, N+1))

    # Construct Hamiltonian in Fourier basis and diagonalize it
    # Basis functions are e^(i*m*phi), with m running from -N to N corresponding to array indices 0 to nTerms-1, respecively (N is equal to (nTerms-1)/2)
    H = np.zeros((nTerms, nTerms))  # Initialize the Hamiltonian matrix in Fourier basis
    diag_elems = (hbar/(2*pi))/(2*M*r**2) * (mvals**2) - frot*mvals + V0/2
    H += np.diag(diag_elems)  # The diagonal elements are the array diag_elems, constructed in the line above
    H += np.eye(nTerms, nTerms,  2) * V0/4  # The off-diagonal elements are constants, and are only nonzero on the second diagonals above and below the main diagonal.
    H += np.eye(nTerms, nTerms, -2) * V0/4

    # Find eigenvalues and eigenvectors of H; since H is Hermitian, eigh conveniently returns the (always real) eigenvalues in ascending order
    # Etilde_all, cm_all = np.linalg.eigh(H)
    H_band = np.zeros((3, nTerms))
    H_band[0, 2:] = np.diagonal(H, offset=2)
    H_band[2, :]  = np.diagonal(H)
    levels_0ind   = np.subtract(levels, 1)  # Subtract 1 from elements of 'levels' to zero-index them
    Etilde_all, cm_all = scla.eig_banded(H_band, select='i', select_range=(0, np.max(levels_0ind)))

    # Select desired eigenvalues and eigenvectors
    # levels_0ind = np.subtract(levels, 1)			# Subtract 1 from elements of 'levels' to zero-index them
    Etilde = Etilde_all[levels_0ind]  # Pick out desired elements of Etilde_all, specified by 'levels'
    cm = cm_all[:, levels_0ind]  # Pick out desired columns of cm_all, specified by 'levels' (each column is an eigenvector)

    # Construct position- (phi-) space wavefunctions
    nPts = nTerms*10  # Number of partitions in which to divide the range phi=0 to phi=2pi
    nLvls = len(levels)
    phi = np.linspace(0, 2*pi, nPts, endpoint=True)
    psitilde = np.zeros((nPts, nLvls), dtype='complex')
    for i in range(nLvls):
        for j in range(nTerms):
            m = mvals[j]
            psitilde[:, i] += 1/np.sqrt(2*pi) * cm[j, i] * np.exp(1j *m*phi)

    # Reduce back dimensions of output arrays if original input for 'levels' was a scalar
    if len(levels) == 1:
        Etilde = np.asscalar(Etilde)
        psitilde = psitilde.reshape(len(psitilde))
        cm = cm.reshape(len(cm))

    return Etilde, psitilde, phi, cm



def V0_frot_theta_fun_general(t, V0i, frotf, tspin, thold1, trel):
    """
    Gives instantaneous rotation frequency, potential height, and phase offset of potential at times t
    Uses the case of linear ramping; initially the rotation frequency is zero and the potential height is V0i.
    Beginning at t=0, the rotation frequency is linearly ramped up to frotf for a time tspin, and then nothing changes for a time thold1
    The potential is then linearly released from V0i to 0 for a time trel

    inputs:
    t: 1-D array of time values (s)
    V0i: scalar; initial potential height (Hz)
    frotf: scalar; final rotational frequency (Hz)
    tspin: scalar; rotation frequency acceleration time (s)
    thold1: scalar; time to wait between ramping up frequency and ramping down potential height (s)
    trel: scalar; pinning amplitude release time (s)

    returns:
    V0: 1-D array of values of V0 corresponding to times of t; 1-D array, same size as t
    frot: 1-D array of values of frot corresponding to times of t; 1-D array, same size as t
    theta: 1-D array of values of theta (phase of rotation) corresponding to times of t; 1-D array, same size as t
    """
    tsh  = tspin + thold1
    tshr = tspin + thold1 + trel

    if tspin != 0:
        V0    = np.select([t < tsh, t < tshr,               t >= tshr],
                          [V0i,     V0i - V0i/trel*(t-tsh), 0])
        frot  = np.select([t < 0,  t < tspin,     t >= tspin],
                          [0,      frotf/tspin*t, frotf])
        theta = np.select([t < 0,  t < tspin,                   t >= tspin],
                          [0,      2*pi*1.0/2*frotf/tspin*t**2, 2*pi * (1.0/2*frotf*tspin + frotf*(t-tspin))])
    else:
        V0 = np.select([t < tsh, t < tshr,               t >= tshr],
                       [V0i,     V0i - V0i/trel*(t-tsh), 0])
        frot = frotf*np.ones_like(t)
        theta = 2*pi*frotf*t

    return V0, frot, theta



def nlintimetoE0(V0i):
    """
    """
    R = np.sqrt(h * V0i / (2 * M * r ** 2))
    E0 = hbar ** 2 / (2 * M * r ** 2) / h
    return 1 / R * (1 + np.sqrt(V0i / E0))



def V0_fun_general_nlinrelease(t, V0i, slowfactor):
    """
    """
    R = np.sqrt(h * V0i / (2 * M * r ** 2))
    t_end0 = nlintimetoE0(V0i)
    t_end = t_end0 * slowfactor

    V0 = np.select([t < t_end, t > t_end],
                   [V0i / (1 + R * t / slowfactor) ** 2, 0])

    return V0



def qobj_hamiltonian(V0i, frotf, tspin, thold1, trel, nTerms):
    """
    Constructs a time-dependent Hamiltonian in the lab frame as a QuTiP quantum object in the angular momentum basis.

    inputs:
    V0i: scalar; initial potential height (Hz)
    frotf: scalar; final rotational freuqency (Hz)
    tspin: scalar; rotation frequency acceleration time (s)
    thold1: scalar; time to wait between ramping up frequency and ramping down potential height (s)
    trel: scalar; pinning amplitude release time (s)
    nTerms: scalar; size of the Hilbert space (the Hamiltonian will be consructed as an nTerms x nTerms operator)

    returns:
    H: QuTiP quantum object; Hamiltonian operator in the Fourier/momentum basis
    """

    # V0_frot_theta_fun takes a time and returns the value of V0 and frot at that time.
    V0_frot_theta_fun = lambda t: V0_frot_theta_fun_general(t, V0i, frotf, tspin, thold1, trel)

    # H1,2,3coeff are time-dependent scalar coefficients multiplying parts of the Hamiltonian (for QuTiP)
    def H1_coeff(t, args):
        (V0, _, _) = V0_frot_theta_fun(t)
        coeff = V0 / 2
        return coeff

    def H2_coeff(t, args):
        (V0, _, theta) = V0_frot_theta_fun(t)
        coeff = V0 / 4 * np.exp(2 * 1j * theta)
        return coeff

    def H3_coeff(t, args):
        (V0, _, theta) = V0_frot_theta_fun(t)
        coeff = V0 / 4 * np.exp(-2 * 1j * theta)
        return coeff

    N = int((nTerms - 1) / 2)

    H0_arr = np.zeros((nTerms, nTerms))
    m = np.array(range(-N, N + 1))
    diag_elems = (hbar / (2 * pi)) / (2 * M * r ** 2) * (m ** 2)
    H0_arr += 2 * pi * np.diag(diag_elems)
    H0_qobj = Qobj(H0_arr)

    # Define diagonal dynamic part of Hamiltonian as a quantum object H1_qobj
    H1_qobj = 2 * pi * qeye(nTerms)

    # Define off-diagonal dynamic parts of Hamiltonian as quantum objects H2_qobj and H3_qobj
    H2_arr = 2 * pi * np.eye(nTerms, nTerms, 2)
    H2_qobj = Qobj(H2_arr)

    H3_arr = 2 * pi * np.eye(nTerms, nTerms, -2)
    H3_qobj = Qobj(H3_arr)

    # Construct Hamiltonian for QuTiP input
    H = [H0_qobj, [H1_qobj, H1_coeff], [H2_qobj, H2_coeff], [H3_qobj, H3_coeff]]

    return H



def qobj_hamiltonian_rotframe(V0i, frotf, trel, nTerms):
    """
    Constructs a time-dependent Hamiltonian in the rotating frame as a QuTiP quantum object in the angular momentum basis.
    This Hamiltonian includes only release, which begins at time t=0 and ends at time t=trel.

    inputs:
    V0i: scalar; initial potential height (Hz)
    frotf: scalar; final rotational freuqency (Hz)
    trel: scalar; pinning amplitude release time (s)
    nTerms: scalar; size of the Hilbert space (the Hamiltonian will be consructed as an nTerms x nTerms operator)

    returns:
    H: QuTiP quantum object; Hamiltonian operator in the Fourier/momentum basis
    """

    # V0_frot_theta_fun takes a time and returns the value of V0 and frot at that time.
    V0_frot_theta_fun = lambda t: V0_frot_theta_fun_general(t, V0i, frotf, 0, 0, trel)

    # H1,2,3coeff are time-dependent scalar coefficients multiplying parts of the Hamiltonian (for QuTiP)
    def H1_coeff(t, args):
        (V0, _, _) = V0_frot_theta_fun(t)
        coeff = V0 / 2
        return coeff

    def H23_coeff(t, args):
        (V0, _, _) = V0_frot_theta_fun(t)
        coeff = V0 / 4
        return coeff

    N = int((nTerms - 1) / 2)
    m = np.array(range(-N, N + 1))

    H0_arr = np.zeros((nTerms, nTerms))
    diag_elems = (hbar / (2 * pi)) / (2 * M * r ** 2) * (m ** 2) - frotf * m
    H0_arr += 2 * pi * np.diag(diag_elems)
    H0_qobj = Qobj(H0_arr)

    # Define diagonal dynamic part of Hamiltonian as a quantum object H1_qobj
    H1_qobj = 2 * pi * qeye(nTerms)

    # Define off-diagonal dynamic parts of Hamiltonian as quantum objects H2_qobj and H3_qobj
    H2_arr = 2 * pi * np.eye(nTerms, nTerms, 2)
    H2_qobj = Qobj(H2_arr)

    H3_arr = 2 * pi * np.eye(nTerms, nTerms, -2)
    H3_qobj = Qobj(H3_arr)

    # Construct Hamiltonian for QuTiP input
    H = [H0_qobj, [H1_qobj, H1_coeff], [H2_qobj, H23_coeff], [H3_qobj, H23_coeff]]

    return H



def qobj_hamiltonian_rotframe_residpot(V0i, V1, phi1, frotf, trel, nTerms):
    """
    Constructs a time-dependent Hamiltonian in the rotating frame as a QuTiP quantum object in the angular momentum basis.
    This Hamiltonian includes only release, which begins at time t=0 and ends at time t=trel.

    inputs:
    V0i: scalar; initial potential height (Hz)
    frotf: scalar; final rotational freuqency (Hz)
    trel: scalar; pinning amplitude release time (s)
    nTerms: scalar; size of the Hilbert space (the Hamiltonian will be consructed as an nTerms x nTerms operator)

    returns:
    H: QuTiP quantum object; Hamiltonian operator in the Fourier/momentum basis
    """

    # V0_frot_theta_fun takes a time and returns the value of V0 and frot at that time.
    V0_frot_theta_fun = lambda t: V0_frot_theta_fun_general(t, V0i, frotf, 0, 0, trel)

    # H1,2,3coeff are time-dependent scalar coefficients multiplying parts of the Hamiltonian (for QuTiP)
    def H1_coeff(t, args):
        (V0, _, _) = V0_frot_theta_fun(t)
        coeff = V0 / 2
        return coeff

    def H2_coeff(t, args):
        (V0, _, theta) = V0_frot_theta_fun(t)
        coeff = V0/4 + V1/4 * np.exp(-2*1j*theta-phi1)
        return coeff

    def H3_coeff(t, args):
        (V0, _, theta) = V0_frot_theta_fun(t)
        coeff = V0/4 + V1/4 * np.exp(2*1j*theta+phi1)
        return coeff

    N = int((nTerms - 1) / 2)
    m = np.array(range(-N, N + 1))

    H0_arr = np.zeros((nTerms, nTerms))
    diag_elems = (hbar / (2 * pi)) / (2 * M * r ** 2) * (m ** 2) - frotf * m
    H0_arr += 2 * pi * np.diag(diag_elems)
    H0_qobj = Qobj(H0_arr)

    # Define diagonal dynamic part of Hamiltonian as a quantum object H1_qobj
    H1_qobj = 2 * pi * qeye(nTerms)

    # Define off-diagonal dynamic parts of Hamiltonian as quantum objects H2_qobj and H3_qobj
    H2_arr = 2 * pi * np.eye(nTerms, nTerms, 2)
    H2_qobj = Qobj(H2_arr)

    H3_arr = 2 * pi * np.eye(nTerms, nTerms, -2)
    H3_qobj = Qobj(H3_arr)


    # Construct Hamiltonian for QuTiP input
    H = [H0_qobj, [H1_qobj, H1_coeff], [H2_qobj, H2_coeff], [H3_qobj, H3_coeff]]

    return H




def qobj_hamiltonian_rotframe_nlinrelease(V0i, frotf, nTerms, slowfactor):
    """
    """

    # V0_frot_theta_fun takes a time and returns the value of V0 and frot at that time.
    V0_fun_nlinrelease = lambda t: V0_fun_general_nlinrelease(t, V0i, slowfactor)

    # H1,2,3coeff are time-dependent scalar coefficients multiplying parts of the Hamiltonian (for QuTiP)
    def H1_coeff(t, args):
        V0 = V0_fun_nlinrelease(t)
        coeff = V0 / 2
        return coeff

    def H23_coeff(t, args):
        V0 = V0_fun_nlinrelease(t)
        coeff = V0 / 4
        return coeff

    N = int((nTerms - 1) / 2)
    m = np.array(range(-N, N + 1))

    H0_arr = np.zeros((nTerms, nTerms))
    diag_elems = (hbar / (2 * pi)) / (2 * M * r ** 2) * (m ** 2) - frotf * m
    H0_arr += 2 * pi * np.diag(diag_elems)
    H0_qobj = Qobj(H0_arr)

    # Define diagonal dynamic part of Hamiltonian as a quantum object H1_qobj
    H1_qobj = 2 * pi * qeye(nTerms)

    # Define off-diagonal dynamic parts of Hamiltonian as quantum objects H2_qobj and H3_qobj
    H2_arr = 2 * pi * np.eye(nTerms, nTerms, 2)
    H2_qobj = Qobj(H2_arr)

    H3_arr = 2 * pi * np.eye(nTerms, nTerms, -2)
    H3_qobj = Qobj(H3_arr)

    # Construct Hamiltonian for QuTiP input
    H = [H0_qobj, [H1_qobj, H1_coeff], [H2_qobj, H23_coeff], [H3_qobj, H23_coeff]]

    return H



def time_evolution(V0i, frotf, tspin, thold1, trel, thold2, initial_state=([1, 2], [1j, 1])):
    """
    Calculates the time evolution of the 'initial_state'th eigenstate of a static potential with height V0i,
    which is accelerated up to a rotational rate of frotf over a time tspin, held for a time thold1, reduced in height
    from V0i to 0 over a time trel, and then held again for a time thold2.

    inputs:
    V0i: scalar; initial potential height (Hz)
    frotf: scalar; final rotational freuqency (Hz)
    tspin: scalar; rotation frequency acceleration time (s)
    thold1: scalar; time to wait between ramping up frequency and ramping down potential height (s)
    trel: scalar; pinning amplitude release time (s)
    thold2: scalar; time to wait after ramping down potential height (s)
    initial_state: tuple of two equally-sized 1-D arrays;
                    initial_state defines an arbitrary linear combination of eigenstates of the initial Hamiltonian
                    The first element of the tuple is an array of integers, listing the eigenstates which are to be included in the linear combination. 1 corresponds to the state of lowest eigenvalue, 2 the next lowest, etc.
                    The second element is an array of complex coefficients which will premultiply the eigenstates listed in the first array, in order, to create the linear combination. The coefficients need not be normalized.
                    If the desired initial state is an eigenstate of the initial Hamiltonian, the elements of initial_state may be scalars.

    returns:
    c: array of Fourier coefficients for psi, the basis functions being exp(i*m*phi). Rows of Am contain values of m ranging from -N to +N for some integer N; each column is a time step
    psi: array of wavefunction values, with the first index of each column being at phi=0 and the last at phi=2pi. Each column is a time step
    phi: 1-D array of phi values, from 0 to 2*pi
    tvals: 1-D array of times, with nTimes elements

    It is worth mentioning why the default value for initial_state is what it is.
    For an initial potential height of at least roughly ~100 Hz, the two lowest eigenstates are two nearly-degenerate harmonic oscillator states, each having about equal probability amplitude in each well.
    This particular linear combination, i|1> + |2>, puts the wavepacket in only one well.
    """

    print('Running simulation...')

    # Option for QuTiP's differential equation solver; emperically seems to need to be a large number
    options = Options(nsteps=1000000)

    # Calculate the time scales in the problem, and hence determine how many time steps to ultimately return, and construct array of times
    if frotf == 0:
        t_rot = 0
    else:
        t_rot = 1.0 / frotf  # Rotational period
    t_ho = 2 * pi * np.sqrt(M * r ** 2 / (2 * h * V0i))  # Harmonic oscillator period
    t_scales = np.array([tspin, thold1, trel, t_rot,
                         t_ho])  # Array of all time scales in problem (excluding thold2 because there are no interesting dynamics after release)
    t_scales = t_scales[t_scales != 0]  # Remove all zero elements
    t_smallest = min(t_scales)  # Find the smallest
    t_step = t_smallest / 15.0  # make time step 1/15 of smallest time scale in problem
    t_sim_total = tspin + thold1 + trel + thold2  # total simulation time
    nTimes = np.ceil(t_sim_total / t_step).astype(int)  # calculate number of time steps, rounding up to nearest integer
    tvals = np.linspace(0, t_sim_total, nTimes)  # finally, produce array of time steps

    # Get Fourier coeffients of the initial state and determine the size nTerms of the Hilbert space
    (_, _, _, c0_eigenvecs) = eigenvals_eigenstates(initial_state[0], V0i, 0, frot_fornterms=frotf)

    # This block prepares the variables c0_eigenvecs and initial_state_coeffs to be matrix-multiplied together in the line of code following this block, which requires c0_eigenvecs to be a 2-D array and initial_state_coeffs to be a 1-D array
    initial_state_coeffs = initial_state[1]
    initial_state_coeffs = np.array(initial_state_coeffs, ndmin=1,
                                    copy=False)  # This line converts the second element of the input tuple initial_state to a 1-element array if it was a scalar. Otherwise, it leaves it as a 1-element array.
    if len(initial_state_coeffs) == 1:
        c0_eigenvecs = np.transpose(np.array(c0_eigenvecs, ndmin=2,
                                             copy=False))  # This line converts c0_eigenvecs to a 2-D array if it wasn't already

    # Calculate the initial state as specified in initial_state in momentum space, then normalize
    c0_unnorm = np.matmul(c0_eigenvecs, initial_state_coeffs)
    c0 = c0_unnorm / np.linalg.norm(c0_unnorm)
    nTerms = len(c0)

    # Initial state as QuTiP quantum object in the Fourier/angular momentum basis
    c0_qobj = Qobj(c0)

    # Construct Hamiltonian
    H = qobj_hamiltonian(V0i, frotf, tspin, thold1, trel, nTerms)

    # Solve time dynamics using QuTiP
    output = mesolve(H, c0_qobj, tvals, progress_bar=True, options=options)

    print('Contructing return variables...')

    # Organize quantum object outputs into the array c, containing the Fouerier coefficients as column vectors, each column being a time step
    states_qobj = output.states
    c_t = np.array([state_qobj.full().flatten() for state_qobj in states_qobj])
    c = np.transpose(c_t)

    # Reconstruct the wavefunctions as column vectors of psi, each column being a time step
    nPts = nTerms * 10
    phi = np.linspace(0, 2 * pi, nPts, endpoint=True)
    psi = np.zeros((nPts, nTimes), dtype='complex')
    for i in range(nTerms):
        m = i - (nTerms - 1) / 2
        psi += 1 / np.sqrt(2 * pi) * np.outer(np.exp(1j * m * phi), c[i, :])

    return c, psi, phi, tvals



def time_evolution_releaseonly(V0i, frotf, trel, thold2, initial_state=([1, 2], [1j, 1])):
    """
    Same as time_evolution(), except the simulation begins at release. Therefore there are no inputs for the times tspin and thold1.
    The initial state in this case is made up of a linear combination of eigenstates of the rotating-frame Hamiltonian, but the simulation itself takes place in the lab frame.
    """

    print('Running simulation...')

    # Option for QuTiP's differential equation solver; emperically seems to need to be a large number
    options = Options(nsteps=1000000)

    # Calculate the time scales in the problem, and hence determine how many time steps to ultimately return, and construct array of times
    # In this simulation, as opposed to the one that includes spinup, the harmonic oscillator period is not relevant.
    if frotf == 0:
        t_rot = 0
    else:
        t_rot = 1.0 / frotf  # Rotational period
    t_scales = np.array([trel,
                         t_rot])  # Array of all time scales in problem (excluding thold2 because there are no interesting dynamics after release)
    t_scales = t_scales[t_scales != 0]  # Remove all zero elements
    t_smallest = min(t_scales)  # Find the smallest
    t_step = t_smallest / 15.0  # make time step 1/15 of smallest time scale in problem
    t_sim_total = trel + thold2  # total simulation time
    nTimes = np.ceil(t_sim_total / t_step).astype(int)  # round up to nearest integer
    tvals = np.linspace(0, t_sim_total, nTimes)  # array of time steps

    # Get Fourier coeffients of the initial state and determine the size nTerms of the Hilbert space
    (_, _, _, c0_eigenvecs) = eigenvals_eigenstates(initial_state[0], V0i, frotf)

    # This block prepares the variables c0_eigenvecs and initial_state[1] to be matrix-multiplied together in the line of code following this block, which requires c0_eigenvecs to be a 2-D array and initial_state[1] to be a 1-D array
    initial_state[1] = np.array(initial_state[1], ndmin=1,
                                copy=False)  # This line converts the second element of the input tuple initial_state to a 1-element array if it was a scalar. Otherwise, it leaves it as a 1-element array.
    if len(initial_state[1]) == 1:
        c0_eigenvecs = np.transpose(np.array(c0_eigenvecs, ndmin=2,
                                             copy=False))  # This line converts c0_eigenvecs to a 2-D array if it wasn't already

    # Calculate the initial state as specified in initial_state in momentum space, then normalize
    c0_unnorm = np.matmul(c0_eigenvecs, initial_state[1])
    c0 = c0_unnorm / np.linalg.norm(c0_unnorm)
    nTerms = len(c0)

    # Initial state as QuTiP quantum object in the Fourier/angular momentum basis
    c0_qobj = Qobj(c0)

    # Construct Hamiltonian
    H = qobj_hamiltonian(V0i, frotf, 0, 0, trel, nTerms)

    # Solve time dynamics using QuTiP
    output = mesolve(H, c0_qobj, tvals, progress_bar=True, options=options)

    print('Contructing return variables...')

    # Organize quantum object outputs into the array c, containing the Fouerier coefficients as column vectors, each column being a time step
    states_qobj = output.states
    c_t = np.array([state_qobj.full().flatten() for state_qobj in states_qobj])
    c = np.transpose(c_t)

    # Reconstruct the wavefunctions as column vectors of psi, each column being a time step
    nPts = nTerms * 10
    phi = np.linspace(0, 2 * pi, nPts, endpoint=True)
    psi = np.zeros((nPts, nTimes), dtype='complex')
    for i in range(nTerms):
        m = i - (nTerms - 1) / 2
        psi += 1 / np.sqrt(2 * pi) * np.outer(np.exp(1j * m * phi), c[i, :])

    return c, psi, phi, tvals



def time_evolution_rotframe(V0i, frotf, trel, thold2, initial_state=([1, 2], [1j, 1]), nTimes=101):
    """
    Same as time_evolution_releaseonly(), except in the rotating frame.
    """

    print('Running simulation...')

    # Option for QuTiP's differential equation solver; emperically seems to need to be a large number
    options = Options(nsteps=1000000000)

    t_sim_total = trel + thold2  # total simulation time
    # nTimes = 101  # screw it; in this case, the only interesting time scale is the release time, so we can just make nTimes a constant at 101
    tvals = np.linspace(0, t_sim_total, nTimes)  # array of time steps

    # Get Fourier coeffients of the initial state and determine the size nTerms of the Hilbert space
    (_, _, _, c0_eigenvecs) = eigenvals_eigenstates(initial_state[0], V0i, frotf)
    c0_unnorm = np.matmul(c0_eigenvecs, initial_state[1])
    c0 = c0_unnorm / np.linalg.norm(c0_unnorm)
    nTerms = len(c0)

    # Initial state as QuTiP quantum object in the Fourier/angular momentum basis
    psi0 = Qobj(c0)

    # Construct Hamiltonian
    H = qobj_hamiltonian_rotframe(V0i, frotf, trel, nTerms)

    # Solve time dynamics using QuTiP
    output = mesolve(H, psi0, tvals, progress_bar=True, options=options)

    print('Contructing return variables...')

    # Organize quantum object outputs into the array c, containing the Fouerier coefficients as column vectors, each column being a time step
    states_qobj = output.states
    c_t = np.array([state_qobj.full().flatten() for state_qobj in states_qobj])
    c = np.transpose(c_t)

    # Reconstruct the wavefunctions as column vectors of psi, each column being a time step
    nPts = nTerms * 10
    phi = np.linspace(0, 2 * pi, nPts, endpoint=True)
    psi = np.zeros((nPts, nTimes), dtype='complex')
    for i in range(nTerms):
        m = i - (nTerms - 1) / 2
        psi += 1 / np.sqrt(2 * pi) * np.outer(np.exp(1j * m * phi), c[i, :])

    return c, psi, phi, tvals



def time_evolution_rotframe_residpot(V0i, V1, phi1, frotf, trel, thold2, initial_state=([1, 2], [1j, 1]), nTimes=101):
    """
    """

    print('Running simulation...')

    # Option for QuTiP's differential equation solver; emperically seems to need to be a large number
    options = Options(nsteps=1000000000)

    t_sim_total = trel + thold2  # total simulation time
    # nTimes = 101  # screw it; in this case, the only interesting time scale is the release time, so we can just make nTimes a constant at 101
    tvals = np.linspace(0, t_sim_total, nTimes)  # array of time steps

    # Get Fourier coeffients of the initial state and determine the size nTerms of the Hilbert space
    (_, _, _, c0_eigenvecs) = eigenvals_eigenstates(initial_state[0], V0i, frotf)
    c0_unnorm = np.matmul(c0_eigenvecs, initial_state[1])
    c0 = c0_unnorm / np.linalg.norm(c0_unnorm)
    nTerms = len(c0)

    # Initial state as QuTiP quantum object in the Fourier/angular momentum basis
    psi0 = Qobj(c0)

    # Construct Hamiltonian
    H = qobj_hamiltonian_rotframe_residpot(V0i, V1, phi1, frotf, trel, nTerms)

    # Solve time dynamics using QuTiP
    output = mesolve(H, psi0, tvals, progress_bar=True, options=options)

    print('Contructing return variables...')

    # Organize quantum object outputs into the array c, containing the Fouerier coefficients as column vectors, each column being a time step
    states_qobj = output.states
    c_t = np.array([state_qobj.full().flatten() for state_qobj in states_qobj])
    c = np.transpose(c_t)

    # Reconstruct the wavefunctions as column vectors of psi, each column being a time step
    nPts = nTerms * 10
    phi = np.linspace(0, 2 * pi, nPts, endpoint=True)
    psi = np.zeros((nPts, nTimes), dtype='complex')
    for i in range(nTerms):
        m = i - (nTerms - 1) / 2
        psi += 1 / np.sqrt(2 * pi) * np.outer(np.exp(1j * m * phi), c[i, :])

    return c, psi, phi, tvals



def time_evolution_rotframe_nlinrelease(V0i, frotf, slowfactor, initial_state=([1, 2], [1j, 1])):
    """
    """

    print('Running simulation...')

    # Option for QuTiP's differential equation solver; emperically seems to need to be a large number
    options = Options(nsteps=1000000000)

    t_sim_total = (1.10) * slowfactor * nlintimetoE0(V0i)  # total simulation time
    nTimes = 101  # screw it; in this case, the only interesting time scale is the release time, so we can just make nTimes a constant at 101
    tvals = np.linspace(0, t_sim_total, nTimes)  # array of time steps

    # Get Fourier coeffients of the initial state and determine the size nTerms of the Hilbert space
    (_, _, _, c0_eigenvecs) = eigenvals_eigenstates(initial_state[0], V0i, frotf)
    c0_unnorm = np.matmul(c0_eigenvecs, initial_state[1])
    c0 = c0_unnorm / np.linalg.norm(c0_unnorm)
    nTerms = len(c0)

    # Initial state as QuTiP quantum object in the Fourier/angular momentum basis
    psi0 = Qobj(c0)

    # Construct Hamiltonian
    H = qobj_hamiltonian_rotframe_nlinrelease(V0i, frotf, nTerms, slowfactor)

    # Solve time dynamics using QuTiP
    output = mesolve(H, psi0, tvals, progress_bar=True, options=options)

    print('Contructing return variables...')

    # Organize quantum object outputs into the array c, containing the Fouerier coefficients as column vectors, each column being a time step
    states_qobj = output.states
    c_t = np.array([state_qobj.full().flatten() for state_qobj in states_qobj])
    c = np.transpose(c_t)

    # Reconstruct the wavefunctions as column vectors of psi, each column being a time step
    nPts = nTerms * 10
    phi = np.linspace(0, 2 * pi, nPts, endpoint=True)
    psi = np.zeros((nPts, nTimes), dtype='complex')
    for i in range(nTerms):
        m = i - (nTerms - 1) / 2
        psi += 1 / np.sqrt(2 * pi) * np.outer(np.exp(1j * m * phi), c[i, :])

    return c, psi, phi, tvals



def runsimulation_rotframe(filename, V0i, frotf, trel, thold2=0, initial_state=([1, 2], [1j, 1])):
    """
    Same thing as the file runsim_labframe_fullsim_linrelease.py; to be used for iterating simulations.
    """

    # Solve for time evolution
    (c, psi, phi, tvals) = time_evolution_rotframe(V0i, frotf, trel, thold2, initial_state=initial_state)
    (V0_vals, frot_vals, _) = V0_frot_theta_fun_general(tvals, V0i, frotf, 0, 0, trel)
    theta_vals = np.zeros_like(tvals)

    # Calculate mvals (which is the only value to be saved which is not explicitly returned by the above functions)
    (nTerms, nTimes) = c.shape
    N = (nTerms - 1) / 2
    mvals = np.array(range(-N, N + 1))

    # Set up save data as a tuple of 4 dictionaries:
    # input_parameters: holds parameters entered under the above "Set parameters here" code block. Each is a scalar value.
    # abscissa_vectors: 1-D arrays of time values "tvals", spatial angular coordinate values "phi", and m (Fourier coefficient index) values "mvals"
    # potential_evolution: 1-D arrays of length len(tvals) describing the potential height "V0_vals", rotation frequency "frot_vals", and potential phase offset "theta_vals" as functions of time
    # wavefn_evolution: 2-dimensional matrices "psi" and "c" describing the wavefunction's time-evolution.
    #                   "psi" is a (len(phi) by len(tvals)) matrix, with rows corresponding to the coordinate (values of "phi") and columns corresponding to times (values of "tvals")
    #                   "c" is a (len(mvals) by len(tvals)) matrix, with rows corresponding to the Fourier coefficient index (values of "mvals") and columns corresponding to times (values of "tvals")
    input_parameters = {'V0i': V0i, 'frotf': frotf, 'tspin': 0, 'thold1': 0, 'trel': trel, 'thold2': thold2}
    abscissa_vectors = {'tvals': tvals, 'phi': phi, 'mvals': mvals}
    potential_evolution = {'V0_vals': V0_vals, 'frot_vals': frot_vals, 'theta_vals': theta_vals}
    wavefn_evolution = {'psi': psi, 'c': c}

    print('Saving data...')
    savedata = (input_parameters, abscissa_vectors, potential_evolution, wavefn_evolution)
    np.save(filename, savedata)



def runsimulation_nlinrelease(filename, V0i, frotf, slowfactor, initial_state=([1, 2], [1j, 1])):
    """
    """

    (c, psi, phi, tvals) = time_evolution_rotframe_nlinrelease(V0i, frotf, slowfactor, initial_state=initial_state)
    V0_vals = V0_fun_general_nlinrelease(tvals, V0i, slowfactor)
    frot_vals = frotf * np.ones_like(tvals)
    theta_vals = np.zeros_like(tvals)

    # Calculate mvals (which is the only value to be saved which is not explicitly returned by the above functions)
    (nTerms, nTimes) = c.shape
    N = (nTerms - 1) / 2
    mvals = np.array(range(-N, N + 1))

    # Set up save data as a tuple of 4 dictionaries:
    # input_parameters: holds parameters entered under the above "Set parameters here" code block. Each is a scalar value.
    # abscissa_vectors: 1-D arrays of time values "tvals", spatial angular coordinate values "phi", and m (Fourier coefficient index) values "mvals"
    # potential_evolution: 1-D arrays of length len(tvals) describing the potential height "V0_vals", rotation frequency "frot_vals", and potential phase offset "theta_vals" as functions of time
    # wavefn_evolution: 2-dimensional matrices "psi" and "c" describing the wavefunction's time-evolution.
    #                   "psi" is a (len(phi) by len(tvals)) matrix, with rows corresponding to the coordinate (values of "phi") and columns corresponding to times (values of "tvals")
    #                   "c" is a (len(mvals) by len(tvals)) matrix, with rows corresponding to the Fourier coefficient index (values of "mvals") and columns corresponding to times (values of "tvals")
    input_parameters = {'V0i': V0i, 'frotf': frotf, 'tspin': 0, 'thold1': 0, 'trel': None, 'thold2': None}
    abscissa_vectors = {'tvals': tvals, 'phi': phi, 'mvals': mvals}
    potential_evolution = {'V0_vals': V0_vals, 'frot_vals': frot_vals, 'theta_vals': theta_vals}
    wavefn_evolution = {'psi': psi, 'c': c}

    print('Saving data...')

    savedata = (input_parameters, abscissa_vectors, potential_evolution, wavefn_evolution)
    np.save(filename, savedata)
