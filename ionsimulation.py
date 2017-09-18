import os
import numpy as np
import qutip as qt
import dill
import time
import matplotlib.pyplot as plt
import matplotlib.animation as anim8



#================================================================================================================================================================================================
#
# Constants
# 
#================================================================================================================================================================================================



# Define constants (all units in SI)
M = 6.64e-26      # Calcium ion mass
r = 3e-6          # Confinement radius
I = M*r**2        # Moment of inertia
pi = np.pi
hbar = 1.054e-34
h = 2*pi*hbar




#================================================================================================================================================================================================
#
# Class definitions
# 
#================================================================================================================================================================================================


class Wavefunction:
    def __init__(self, wavefn_angmomspace, m_limits):
        """
        """

        # This ensures that if the input array for wavefn_angmomspace is 1-dimensional, it is coerced into a 2-dimensional column array
        wavefn_angmomspace = np.array(wavefn_angmomspace)
        if wavefn_angmomspace.ndim != 2:
            wavefn_angmomspace = np.array(wavefn_angmomspace, ndmin=2).T

        # Array of values of the angular momentum quantum number m; serves as the index for the state vector in angular momentum space
        self.m_vals = mlimits_to_mvals(m_limits)

        # Array of state vectors in angular momentum space; each column is a state vector, with each row's element corresponding to the corresponding value of m in m_vals
        self.angmomspace = wavefn_angmomspace

        # Array of position-space values phi, ranging from 0 to 2pi. Serves as the index for the state vector in position space. Contains 10 times as many points as the highest value of m, so that the position space resolution is high enough
        n_phi_vals = 10*max(np.abs(self.m_vals))
        self.phi_vals = np.linspace(0, 2*pi, n_phi_vals)

        # Array of state vectors in position space; each column is a state vector, with each row's element corresponding to the corresponding value of phi in phi_vals
        # This is calculated given the state vector in angular momentum space
        n_states = np.size(self.angmomspace, 1)
        wavefn_positionspace = np.zeros((n_phi_vals, n_states), dtype='complex')
        for i in range(len(self.m_vals)):
            m = self.m_vals[i]
            wavefn_positionspace += 1/np.sqrt(2*pi)*np.outer(np.exp(1j*m*self.phi_vals), self.angmomspace[i, :])
        self.positionspace = wavefn_positionspace

    def angmomspace(self):
        return self.angmomspace

    def m_vals(self):
        return self.m_vals

    def positionspace(self):
        return self.positionspace

    def phi_vals(self):
        return self.phi_vals

    def n_states(self):
        return np.size(self.angmomspace, 1)

    def m_limits(self):
        return [self.m_vals[0], self.m_vals[-1]]

    def n_m_vals(self):
        return len(self.m_vals)

    def n_phi_vals(self):
        return len(self.phi_vals)

    def magnitude_angmomspace(self):
        return np.abs(self.angmomspace)

    def magnitude2_angmomspace(self):
        return np.abs(self.angmomspace)**2

    def magnitude_positionspace(self):
        return np.abs(self.positionspace)

    def magnitude2_positionspace(self):
        return np.abs(self.positionspace)**2

    def subset(self, new_states=None, new_m_limits=None):
        if new_states is None:
            new_states = range(self.n_states())
        if new_m_limits is None:
            new_m_limits = self.m_limits()

        new_m_min = new_m_limits[0]
        new_m_max = new_m_limits[1]
        old_m_min = self.m_limits()[0]
        new_angmomspace = self.angmomspace[new_m_min-old_m_min:new_m_max-old_m_min+1, new_states]

        return Wavefunction(new_angmomspace, new_m_limits)



class Hamiltonian_Static:
    def __init__(self, hamiltonian_qobj, m_limits):
        self.qobj = hamiltonian_qobj
        self.m_vals = mlimits_to_mvals(m_limits)

    def m_limits(self):
        return [self.m_vals[0], self.m_vals[-1]]



class Simulation:
    def __init__(self, states, hamiltonian_dynamic, time_vals=None, input_params=None, hamiltonian_time_params=None, timetorun=None):
        self.states = states
        self.hamiltonian_dynamic = hamiltonian_dynamic
        self.time_vals = time_vals
        self.input_params = input_params
        self.hamiltonian_time_params = hamiltonian_time_params
        self.timetorun = timetorun

    def n_time_vals(self):
        return len(self.time_vals)

    def set_time_vals(self, time_vals):
        self.time_vals = time_vals

    def set_input_params(self, input_params):
        self.input_params = input_params

    def set_hamiltonian_time_params(self, hamiltonian_time_params):
        self.hamiltonian_time_params = hamiltonian_time_params

    def set_timetorun(self, timetorun):
        self.timetorun = timetorun




#================================================================================================================================================================================================
#
# Misc. small functions
# 
#================================================================================================================================================================================================


def get_loaddir(loadfile, folder):
    if folder is None:
        loaddir = loadfile
    else:
        currentfolder = os.path.dirname(os.path.realpath(__file__))
        loaddir = currentfolder + '/' + folder + '/' + loadfile
    return loaddir



def get_savedir(savefile, folder):
    if folder is None:
        savedir = savefile
    else:
        currentfolder = os.path.dirname(os.path.realpath(__file__))
        savefolder = currentfolder + '/' + folder
        if not os.path.isdir(savefolder):
            os.mkdir(savefolder)
        savedir = savefolder + '/' + savefile
    return savedir



def calculate_time_values(t_duration, t_start=0, n_time_steps=101):
    #
    return np.linspace(t_start, t_start+t_duration, num=n_time_steps)



def mlimits_to_mvals(m_limits):
    m_min = m_limits[0]
    m_max = m_limits[1]
    return np.array(range(m_min, m_max+1))



def mlimits_to_nmvals(m_limits):
    #
    return len(mlimits_to_mvals(m_limits))
    


def calculate_m_limits(V0, f_rot, level, spreadfactor=8):
    Lz_spread_ground = ((2*I*h*V0) / hbar**2) ** 0.25
    Lz_center = I*2*pi*f_rot/hbar
    harmonic_angfrequency = np.sqrt(2*h*V0/I)

    if 0.5*(level + 1/2)*harmonic_angfrequency < V0:
        m_min = int(Lz_center - (np.sqrt(level) + spreadfactor)*Lz_spread_ground)
        m_max = int(Lz_center + (np.sqrt(level) + spreadfactor)*Lz_spread_ground)
    else:
        m_min = int(Lz_center - max((level + spreadfactor)*Lz_spread_ground, 10))
        m_max = int(Lz_center + max((level + spreadfactor)*Lz_spread_ground, 10))

    return [m_min, m_max]



def calculate_m_limits_fullsim(V0_vals, f_rot_vals, initstate_eigbasis):
    f_rot_min = min(f_rot_vals)
    f_rot_max = max(f_rot_vals)
    V0_max = max(V0_vals)
    max_level = max(initstate_eigbasis[0])

    m_min = calculate_m_limits(V0_max, f_rot_min, max_level)[0]
    m_max = calculate_m_limits(V0_max, f_rot_max, max_level)[1]

    return [m_min, m_max]



def V0cutuff_to_reltime(V0_i, slowfactor, V0_cutoff):
    a = np.sqrt(h*V0_i/(2*I))
    return slowfactor * 1/a * (np.sqrt(V0_i/V0_cutoff) - 1)



def reltime_to_V0cutoff(V0_i, slowfactor, t_release):
    """
    """
    return V0_i/(1 + a*t_release/slowfactor)**2



def inner_product(vector1, vector2):
    #
    return np.sum(np.multiply(vector1, np.conj(vector2)))




#================================================================================================================================================================================================
#
# Functions for constructing Hamiltonians and states
# 
#================================================================================================================================================================================================


def V0_function_linearrelease(V0_i, t_startrelease, t_release):
    if t_release != 0:
        def V0_function(t):
            V0 = np.select([t <  t_startrelease,
                            t <  t_startrelease+t_release,
                            t >= t_startrelease+t_release],
                           [V0_i,
                            V0_i - V0_i/t_release*(t-t_startrelease),
                            0])
            return V0
    else:
        def V0_function(t):
            V0 = np.select([t <  t_startrelease,
                            t >= t_startrelease],
                           [V0_i,
                            0])
            return V0

    return V0_function



def V0_function_nonlinearrelease(V0_i, t_startrelease, t_release, slowfactor):
    def V0_function(t):
        V0 = np.select([t <  t_startrelease,
                        t <  t_startrelease+t_release,
                        t >= t_startrelease+t_release],
                       [V0_i,
                        V0_i/(1+a*((t-t_startrelease)/slowfactor))**2,
                        0])
        return V0

    return V0_function



def frot_function_linearspinup(f_rot_f, t_startspinup, t_spinup):
    if t_spinup != 0:
        def frot_function(t):
            f_rot = np.select([t <  t_startspinup,
                               t <  t_startspinup+t_spinup,
                               t >= t_startspinup+t_spinup],
                              [0,
                               f_rot_f/t_spinup*(t-t_startspinup),
                               f_rot_f])
            return f_rot
    else:
        def frot_function(t):
            f_rot = np.select([t <  t_startspinup,
                               t >= t_startspinup],
                              [0,
                               f_rot_f])
            return f_rot

    return frot_function



def theta_function_linearspinup(f_rot_f, t_startspinup, t_spinup):
    if t_spinup != 0:
        def theta_function(t):
            theta = np.select([t <  t_startspinup,
                               t <  t_startspinup+t_spinup,
                               t >= t_startspinup+t_spinup],
                              [0,
                               2*pi * 1/2.0*f_rot_f/t_spinup*(t-t_startspinup)**2,
                               2*pi * (1/2.0*f_rot_f*t_spinup + f_rot_f*(t-(t_startspinup+t_spinup)))])
            return theta
    else:
        def theta_function(t):
            theta = np.select([t <  t_startspinup,
                               t >= t_startspinup],
                              [0,
                               2*pi*f_rot_f*(t-t_startspinup)])
            return theta

    return theta_function



def construct_dynamic_hamiltonian(V0_function, theta_function, f_rot_frame, V0_resid, theta_resid, m_limits):
    m_vals = mlimits_to_mvals(m_limits)
    n_dims = len(m_vals)

    H_constant = np.diag( (1/h)*(hbar**2*m_vals**2)/(2*I) - f_rot_frame*m_vals )

    H_timedep_diag0 = np.eye(n_dims)
    def H_timedep_diag0_coeff(t, args):
        V0 = V0_function(t)
        return V0/2.0

    H_timedep_diagP2 = np.eye(n_dims, k=2)
    def H_timedep_diagP2_coeff(t, args):
        V0 = V0_function(t)
        theta = theta_function(t)
        return V0/4.0 * np.exp(2j*(theta - 2*pi*f_rot_frame*t)) \
             + V0_resid/4.0 * np.exp(2j*(theta_resid - 2*pi*f_rot_frame*t))

    H_timedep_diagM2 = np.eye(n_dims, k=-2)
    def H_timedep_diagM2_coeff(t, args):
        V0 = V0_function(t)
        theta = theta_function(t)
        return V0/4.0 * np.exp(-2j*(theta - 2*pi*f_rot_frame*t)) \
             + V0_resid/4.0 * np.exp(-2j*(theta_resid - 2*pi*f_rot_frame*t))

    H_constant       = qt.Qobj(2*pi*H_constant)
    H_timedep_diag0  = qt.Qobj(2*pi*H_timedep_diag0)
    H_timedep_diagP2 = qt.Qobj(2*pi*H_timedep_diagP2)
    H_timedep_diagM2 = qt.Qobj(2*pi*H_timedep_diagM2)

    return [H_constant,
           [H_timedep_diag0,  H_timedep_diag0_coeff],
           [H_timedep_diagP2, H_timedep_diagP2_coeff],
           [H_timedep_diagM2, H_timedep_diagM2_coeff]]



def construct_static_hamiltonian(V0, theta, f_rot_frame, V0_resid, theta_resid, m_limits):
    m_vals = mlimits_to_mvals(m_limits)
    n_dims = len(m_vals)

    H_0 = np.diag( (1/h)*(hbar**2*m_vals**2)/(2*I) - f_rot_frame*m_vals )

    H_1 = np.eye(n_dims)
    H_1_coeff = V0/2.0

    H_2 = np.eye(n_dims, k=2)
    H_2_coeff = V0/4.0 * np.exp(2j*theta) \
                           + V0_resid/4.0 * np.exp(2j*theta_resid)

    H_3 = np.eye(n_dims, k=-2)
    H_3_coeff = V0/4.0 * np.exp(-2j*theta) \
                           + V0_resid/4.0 * np.exp(-2j*theta_resid)

    hamiltonian_qobj = qt.Qobj(H_0 + H_1_coeff*H_1
                                   + H_2_coeff*H_2
                                   + H_3_coeff*H_3)

    hamiltonian_static = Hamiltonian_Static(hamiltonian_qobj, m_limits)
    return hamiltonian_static



def compute_static_hamiltonian(hamiltonian_dynamic, time, m_limits):
    H_constant = hamiltonian_dynamic[0]
    H_timedeps       = [elem[0] for elem in hamiltonian_dynamic[1:]]
    H_timedep_coeffs = [elem[1] for elem in hamiltonian_dynamic[1:]]

    n_timedep_terms = len(H_timedeps)

    hamiltonian_qobj = qt.Qobj(np.zeros_like(H_constant.full()))
    hamiltonian_qobj += H_constant
    for i in range(n_timedep_terms):
        H_timedep       = H_timedeps[i]
        H_timedep_coeff = H_timedep_coeffs[i]
        hamiltonian_qobj += H_timedep_coeff(time, None) * H_timedep

    hamiltonian_qobj /= 2*pi
    hamiltonian_static = Hamiltonian_Static(hamiltonian_qobj, m_limits)
    return hamiltonian_static



def diagonalize_hamiltonian(hamiltonian_static, levels):
    levels = np.array(levels, ndmin=1)

    hamiltonian_qobj = hamiltonian_static.qobj
    (eigenvalues, eigenvectors) = hamiltonian_qobj.eigenstates()
    energies = eigenvalues[levels]

    eigenstates_array_transpose = [eigenvector.full().flatten() for eigenvector in eigenvectors[levels]]
    eigenstates_array = np.transpose(eigenstates_array_transpose)

    m_limits = hamiltonian_static.m_limits()
    eigenstates = Wavefunction(eigenstates_array, m_limits)

    return (energies, eigenstates)



def compute_state(hamiltonian_static, state_in_eigbasis):
    levels = state_in_eigbasis[0]
    ceofficients = state_in_eigbasis[1]

    (_, eigenstates) = diagonalize_hamiltonian(hamiltonian_static, levels)
    eigenstates_array = eigenstates.angmomspace

    state_array = np.zeros_like(eigenstates_array[:, 0])
    for i in range(len(levels)):
        state_array += ceofficients[i]*eigenstates_array[:, i]
    state_array /= np.linalg.norm(state_array)

    m_limits = hamiltonian_static.m_limits()
    state = Wavefunction(state_array, m_limits)
    return state




#================================================================================================================================================================================================
#
# Functions for running simulatoins
# 
#================================================================================================================================================================================================



def run_simulation(initial_state, hamiltonian_dynamic, time_vals, input_params, hamiltonian_time_params, options=qt.Options(nsteps=1000000)):
    initial_state_qobj = qt.Qobj(initial_state.angmomspace)

    print('Running simulation...')
    qutip_output = qt.mesolve(hamiltonian_dynamic, initial_state_qobj, time_vals, progress_bar=True, options=options)

    print('Constructing simulation object...')
    wavefns_angmomspace_qobj = qutip_output.states
    wavefns_angmomspace_transpose = np.array([wavefn.full().flatten() for wavefn in wavefns_angmomspace_qobj])
    wavefns_angmomspace = np.transpose(wavefns_angmomspace_transpose)
    m_limits = initial_state.m_limits()
    wavefns = Wavefunction(wavefns_angmomspace, m_limits)
    hamiltonian_dynamic_obj = dill.dumps(hamiltonian_dynamic)
    simulation_object = Simulation(wavefns, hamiltonian_dynamic_obj, time_vals=time_vals, input_params=input_params, hamiltonian_time_params=hamiltonian_time_params)

    return simulation_object



def simulate_linear_release(savefile, V0_i, f_rot_f, f_rot_frame, V0_resid, theta_resid, t_spinup, t_hold1, t_release, t_hold2, initstate_eigbasis=([0, 1], [1j, 1]), save=True, folder=None):
    
    start_time = time.time()

    time_vals = calculate_time_values(t_spinup + t_hold1 + t_release + t_hold2)
    
    V0_function = V0_function_linearrelease(V0_i, t_spinup+t_hold1, t_release)
    f_rot_function = frot_function_linearspinup(f_rot_f, 0, t_spinup)
    theta_function = theta_function_linearspinup(f_rot_f, 0, t_spinup)

    V0_vals = V0_function(time_vals)
    f_rot_vals = f_rot_function(time_vals)
    theta_vals = theta_function(time_vals)

    m_limits = calculate_m_limits_fullsim(V0_vals, f_rot_vals, initstate_eigbasis)

    hamiltonian_dynamic = construct_dynamic_hamiltonian(V0_function, theta_function, f_rot_frame, V0_resid, theta_resid, m_limits)

    initial_hamiltonian = compute_static_hamiltonian(hamiltonian_dynamic, 0, m_limits)
    initial_state = compute_state(initial_hamiltonian, initstate_eigbasis)

    input_params = {'V0_i': V0_i,
                    'f_rot_f': f_rot_f,
                    'f_rot_frame': f_rot_frame,
                    'V0_resid': V0_resid,
                    'theta_resid': theta_resid,
                    't_spinup': t_spinup,
                    't_hold1': t_hold1,
                    't_release': t_release,
                    't_hold2': t_hold2,
                    'initstate_eigbasis': initstate_eigbasis}
    hamiltonian_time_params = {'V0_vals': V0_vals,
                               'f_rot_vals': f_rot_vals,
                               'theta_vals': theta_vals}

    simulation = run_simulation(initial_state, hamiltonian_dynamic, time_vals, input_params, hamiltonian_time_params)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    simulation.set_timetorun(elapsed_time)

    if save:
        print('Saving data...')
        savedir = get_savedir(savefile, folder)
        np.save(savedir, [simulation])



def simulate_nonlinear_release(savefile, V0_i, f_rot_f, f_rot_frame, V0_resid, theta_resid, t_spinup, t_hold1, t_release, t_hold2, slowfactor, initstate_eigbasis=([0, 1], [1j, 1]), save=True, folder=None):
    start_time = time.time()

    time_vals = calculate_time_values(t_spinup + t_hold1 + t_release + t_hold2)
    
    V0_function = V0_function_nonlinearrelease(V0_i, t_spinup+t_hold1, t_release, slowfactor)
    f_rot_function = frot_function_linearspinup(f_rot_f, 0, t_spinup)
    theta_function = theta_function_linearspinup(f_rot_f, 0, t_spinup)

    V0_vals = V0_function(time_vals)
    f_rot_vals = f_rot_function(time_vals)
    theta_vals = theta_function(time_vals)

    m_limits = calculate_m_limits_fullsim(V0_vals, f_rot_vals, initstate_eigbasis)

    hamiltonian_dynamic = construct_dynamic_hamiltonian(V0_function, theta_function, f_rot_frame, V0_resid, theta_resid, m_limits)

    initial_hamiltonian = compute_static_hamiltonian(hamiltonian_dynamic, 0, m_limits)
    initial_state = compute_state(initial_hamiltonian, initstate_eigbasis)

    input_params = {'V0_i': V0_i,
                    'f_rot_f': f_rot_f,
                    'f_rot_frame': f_rot_frame,
                    'V0_resid': V0_resid,
                    'theta_resid': theta_resid,
                    't_spinup': t_spinup,
                    't_hold1': t_hold1,
                    't_release': t_release,
                    't_hold2': t_hold2,
                    'slowfactor': slowfactor,
                    'initstate_eigbasis': initstate_eigbasis}
    hamiltonian_time_params = {'V0_vals': V0_vals,
                               'f_rot_vals': f_rot_vals,
                               'theta_vals': theta_vals}

    simulation = run_simulation(initial_state, hamiltonian_dynamic, time_vals, input_params, hamiltonian_time_params)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    simulation.set_timetorun(elapsed_time)

    if save:
        print('Saving data...')
        savedir = get_savedir(savefile, folder)
        np.save(savedir, [simulation])





#================================================================================================================================================================================================
#
# Functions for analyzing completed simulations
# 
#================================================================================================================================================================================================



def check_sim_parameters(loadfile):
	simulation = np.load(loadfile)[0]
	print(simulation.input_params)
	print({'n_time_vals': simulation.n_time_vals()})
	print({'timetorun': simulation.timetorun})



def animate_angmomspace(loadfile, total_animation_time=10, max_n_frames=201, save=False, savefile=None):
    simulation = np.load(loadfile)[0]

    time_vals   = simulation.time_vals
    angmomspace = simulation.states.angmomspace
    m_vals      = simulation.states.m_vals
    V0_vals     = simulation.hamiltonian_time_params['V0_vals']
    f_rot_vals  = simulation.hamiltonian_time_params['f_rot_vals']

    psi_real_max = np.amax(np.absolute(angmomspace.real))
    psi_imag_max = np.amax(np.absolute(angmomspace.imag))
    psi_max  = np.amax(np.absolute(angmomspace))

    x_max = m_vals[-1]
    x_min = m_vals[0]
    y_max = np.ceil(10*max(psi_real_max, psi_imag_max, psi_max))/10.0
    y_min = -y_max

    fig = plt.figure()
    ax = plt.axes(xlim=(x_min, x_max),
                  ylim=(y_min, y_max),
                  xlabel='m',
                  ylabel='psi')
    line1, = ax.plot([], [], linestyle='-', marker='.', label='Re c', color='b')
    line2, = ax.plot([], [], linestyle='-', marker='.', label='Im c', color='r')
    line3, = ax.plot([], [], linestyle='-', marker='.', label='|c|',  color='g')

    txt_time = ax.text(x_min+0.05*(x_max-x_min), 0.7*y_min, '')
    txt_V0   = ax.text(x_min+0.05*(x_max-x_min), 0.8*y_min, '')
    txt_frot = ax.text(x_min+0.05*(x_max-x_min), 0.9*y_min, '')

    plt.legend(handles=[line1, line2, line3], loc=4)

    n_time_steps = len(time_vals)
    if n_time_steps <= max_n_frames:
        n_frames = n_time_steps
        framelist = range(n_frames)
    else:
        n_frames = max_n_frames
        framelist = np.zeros(max_n_frames, dtype='int')
        di = (n_time_steps-1.0)/(max_n_frames-1.0)
        for i in range(max_n_frames):
            framelist[i] = int(round(i*di))

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return line1, line2, line3

    def animate(i):
        i = framelist[i]
        
        t     = time_vals[i]
        V0    = V0_vals[i]
        f_rot = f_rot_vals[i]

        line1.set_data(m_vals, angmomspace.real[:,i])
        line2.set_data(m_vals, angmomspace.imag[:,i])
        line3.set_data(m_vals, np.absolute(angmomspace[:,i]))

        txt_time.set_text('t = '     + str(t))
        txt_V0.  set_text('V0 = '    + str(V0))
        txt_frot.set_text('f_rot = ' + str(f_rot))
        return line1, line2, line3, txt_time, txt_V0, txt_frot

    ani = anim8.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1000.0*total_animation_time/n_frames, blit=True)
    plt.show()

    if save:
        if savefile is None:
            savefile = loadfile[:-4] + '_angmomspace_animation.gif'
        print('Saving .gif...')
        ani.save(savefile, dpi=80, writer='imagemagick')



def animate_positionspace(loadfile, frame='withpotential', total_animation_time=10, max_n_frames=201, save=False, savefile=None):
    simulation = np.load(loadfile)[0]

    time_vals     = simulation.time_vals
    positionspace = simulation.states.positionspace
    phi_vals      = simulation.states.phi_vals
    V0_vals       = simulation.hamiltonian_time_params['V0_vals']
    f_rot_vals    = simulation.hamiltonian_time_params['f_rot_vals']
    theta_vals    = simulation.hamiltonian_time_params['theta_vals']
    f_rot_frame   = simulation.input_params['f_rot_frame']
    V0_i          = simulation.input_params['V0_i']





    n_time_vals = len(time_vals)

    if frame == 'lab':
        psi_toplot = positionspace
        theta_toplot = theta_vals

    elif frame == 'rot':
        psi_toplot = np.zeros_like(positionspace)
        for i in range(n_time_vals):
            phi_shift = phi_vals - np.mod(2*pi*f_rot_frame*time_vals[i], 2*np.pi)
            psi_toswap_lefttoright = positionspace[phi_shift <  0, i]
            psi_toswap_righttoleft = positionspace[phi_shift >= 0, i]
            psi_toplot[:, i] = np.append(psi_toswap_righttoleft, psi_toswap_lefttoright)
        theta_toplot = theta_vals - 2*pi*f_rot_frame*time_vals

    elif frame == 'withpotential':
        psi_toplot = np.zeros_like(positionspace)
        for i in range(n_time_vals):
            phi_shift = phi_vals - np.mod(theta_vals[i], 2*np.pi)
            psi_toswap_lefttoright = positionspace[phi_shift <  0, i]
            psi_toswap_righttoleft = positionspace[phi_shift >= 0, i]
            psi_toplot[:, i] = np.append(psi_toswap_righttoleft, psi_toswap_lefttoright)
        theta_toplot = np.zeros_like(theta_vals)

    else:
        raise ValueError('Argument "frame" must have the value "lab", "rot", or "withpotential".')








    psi_real_max = np.amax(np.absolute(positionspace.real))
    psi_imag_max = np.amax(np.absolute(positionspace.imag))
    psi2_max  = np.amax(np.absolute(positionspace)**2)

    x_max = 2*pi
    x_min = 0
    y1_max = np.ceil(max(psi_real_max, psi_imag_max, psi2_max))
    y1_min = -y1_max
    y2_max = V0_i
    y2_min = -y2_max


    fig = plt.figure()
    ax1 = plt.axes(xlim=(x_min, x_max),
                   ylim=(y1_min, y1_max),
                   xlabel='phi',
                   ylabel='psi')
    ax2 = ax1.twinx()
    ax2.set_xlim((x_min, x_max))
    ax2.set_ylim((y2_min, y2_max))
    ax2.set_ylabel('V')

    line1, = ax1.plot([], [], label='Re psi',  color='b')
    line2, = ax1.plot([], [], label='Im psi',  color='r')
    line3, = ax1.plot([], [], label='|psi|^2', color='g')
    line4, = ax2.plot([], [], label='V',       color='k')

    txt_time = ax1.text(0.3, 0.7*y1_min, '')
    txt_V0   = ax1.text(0.3, 0.8*y1_min, '')
    txt_frot = ax1.text(0.3, 0.9*y1_min, '')

    plt.legend(handles=[line1, line2, line3, line4], loc=4)

    n_time_steps = len(time_vals)
    if n_time_steps <= max_n_frames:
        n_frames = n_time_steps
        framelist = range(n_frames)
    else:
        n_frames = max_n_frames
        framelist = np.zeros(max_n_frames, dtype='int')
        di = (n_time_steps-1.0)/(max_n_frames-1.0)
        for i in range(max_n_frames):
            framelist[i] = int(round(i*di))

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        return line1, line2, line3, line4

    def animate(i):
        i = framelist[i]
        
        t     = time_vals[i]
        V0    = V0_vals[i]
        f_rot = f_rot_vals[i]

        line1.set_data(phi_vals, psi_toplot.real[:,i])
        line2.set_data(phi_vals, psi_toplot.imag[:,i])
        line3.set_data(phi_vals, np.absolute(psi_toplot[:,i])**2)
        line4.set_data(phi_vals, V0/2*(np.cos(2*(phi_vals-theta_toplot[i]))+1))
        txt_time.set_text('t = '    + str(t))
        txt_V0.  set_text('V0 = '   + str(V0))
        txt_frot.set_text('frot = ' + str(f_rot))
        return line1, line2, line3, line4, txt_time, txt_V0, txt_frot

    ani = anim8.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1000.0*total_animation_time/n_frames, blit=True)
    plt.show()

    if save:
        if savefile is None:
            savefile = loadfile[:-4] + '_positionspace_animation.gif'
        print('Saving .gif...')
        ani.save(savefile, dpi=80, writer='imagemagick')



def plot_angmom_vs_time(loadfile, folder=None, verticallines=None):
    loaddir = get_loaddir(loadfile, folder)
    simulation = np.load(loaddir)[0]

    time_vals   = simulation.time_vals
    angmomspace = simulation.states.angmomspace
    m_vals      = simulation.states.m_vals
    f_rot_vals  = simulation.hamiltonian_time_params['f_rot_vals']
    t_spinup    = simulation.input_params['t_spinup']
    t_hold1     = simulation.input_params['t_hold1']
    t_release   = simulation.input_params['t_release']

    # Construct 1-D array of angular momentum (Lz) expectation values vs. time
    n_time_vals = len(time_vals)

    Lz_exp = [sum(m_vals*np.absolute(angmomspace[:,i])**2) for i in range(n_time_vals)]

    # Assign new time constants for convenience
    t_sh  = t_spinup + t_hold1
    t_shr = t_spinup + t_hold1 + t_release

    # Create plot
    fig = plt.figure()
    line1, = plt.plot(1000*time_vals, M*r**2*2*pi*f_rot_vals/hbar, label='I*omega_rot', color='k', linestyle='-', linewidth=1)
    line2, = plt.plot(1000*time_vals, Lz_exp,                      label='<L_z>')
    plt.legend(handles=[line2, line1], loc=4)

    # Draw 3 dashed vertical lines to show (1) end of spinup, (2) beginning of release, and (3) end of release
    axes = plt.gca()
    (y_min, y_max) = axes.get_ylim()
    axes.plot([1000*t_spinup, 1000*t_spinup], [y_min, y_max], color='k', linestyle='--', linewidth=1)
    axes.plot([1000*t_sh,     1000*t_sh],     [y_min, y_max], color='k', linestyle='--', linewidth=1)
    axes.plot([1000*t_shr,    1000*t_shr],    [y_min, y_max], color='k', linestyle='--', linewidth=1)
    # Also draw the 
    axes.set_ylim(y_min, y_max)   # Reset y-axis limits back to what they were before the vertical lines were drawn, as this will have automatically changed the limits
    axes.set_xlabel('Time (ms)')    
    axes.set_ylabel('Mean angular momentum (hbar)')
    plt.show()



def plot_angmomdist_vs_time(loadfile):
    simulation = np.load(loadfile)[0]
    time_vals = simulation.time_vals
    m_vals = simulation.states.m_vals
    angmom_magnitude2 = simulation.states.magnitude2_angmomspace()

    x = time_vals
    y = m_vals

    extra_x = x[-1] + (x[-1]-x[-2])
    extra_y = y[-1] + 1
    x = np.append(x, extra_x)
    y = np.append(y, extra_y)
    (x, y) = np.meshgrid(x, y)

    z = np.zeros_like(x)
    z[:-1, :-1] = angmom_magnitude2

    plt.figure(1)
    ax1 = plt.gca()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angular momentum quantum number m')
    plt.pcolormesh(x, y, z, cmap='jet')
    plt.colorbar(label='Magnitude squared of amplitude, |c|^2')
    plt.show()



def plot_eigstatepops_vs_time(loadfile, n_statestoplot=50):
    simulation = np.load(loadfile)[0]
    levels = range(n_statestoplot)

    time_vals   = simulation.time_vals
    m_limits    = simulation.states.m_limits()
    V0_vals     = simulation.hamiltonian_time_params['V0_vals']
    theta_vals  = simulation.hamiltonian_time_params['theta_vals']
    f_rot_vals  = simulation.hamiltonian_time_params['f_rot_vals']
    V0_resid    = simulation.input_params['V0_resid']
    theta_resid = simulation.input_params['theta_resid']
    # hamiltonian_dynamic = dill.loads(simulation.hamiltonian_dynamic)

    x = time_vals
    y = levels

    extra_x = x[-1] + (x[-1]-x[-2])
    extra_y = y[-1] + 1
    x = np.append(x, extra_x)
    y = np.append(y, extra_y)
    (x, y) = np.meshgrid(x, y)

    z = np.zeros_like(x)
    # for i, time in enumerate(time_vals):
    for i in range(len(time_vals)):
        # hamiltonian_static = compute_static_hamiltonian(hamiltonian_dynamic, time, m_limits)
        hamiltonian_static = construct_static_hamiltonian(V0_vals[i], theta_vals[i], f_rot_vals[i], V0_resid, theta_resid, m_limits)

        print('Diagonalizing Hamiltonian for time step ' + str(i+1) + '/' + str(len(time_vals)) + '...')
        (_, eigstates) = diagonalize_hamiltonian(hamiltonian_static, levels)
        eigstates = eigstates.angmomspace

        state = simulation.states.angmomspace[:, i]

        #z[:, i] = [np.absolute(inner_product(state, eigstate))**2 for eigstate in eigstates]
        for j in range(n_statestoplot):
            z[j, i] = np.absolute(inner_product(state, eigstates[:, j]))**2

    plt.figure(1)
    ax1 = plt.gca()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Eigenstate')
    plt.pcolormesh(x, y, z, cmap='jet')
    plt.colorbar(label='Magnitude squared of population of eigenstate')
    plt.show()



def plot_angmomdist_vs_reltime(loadfiles, verticalline=None):
    simulations = [np.load(loadfile)[0] for loadfile in loadfiles]
    rel_times = [simulation.input_params['t_release'] for simulation in simulations]
    m_min = max([simulation.states.m_limits()[0] for simulation in simulations])
    m_max = min([simulation.states.m_limits()[1] for simulation in simulations])
    final_states = [simulation.states.subset(new_states=-1, new_m_limits=[m_min, m_max]) for simulation in simulations]


    x = rel_times
    y = range(m_min, m_max+1)

    x_sorted = sorted(x)
    
    extra_x = x_sorted[-1] * x_sorted[-1]/x_sorted[-2]
    extra_y = y[-1] + 1
    x = np.append(x, extra_x)
    y = np.append(y, extra_y)
    (x, y) = np.meshgrid(x, y)

    z = np.zeros_like(x)
    for i, final_state in enumerate(final_states):
        z[:-1, i] = final_state.magnitude2_angmomspace().flatten()

    plt.figure(1)
    ax1 = plt.gca()
    ax1.set_xscale('log')
    ax1.set_xlabel('Release time (s)')
    ax1.set_ylabel('Angular momentum quantum number m')
    plt.pcolormesh(x, y, z, cmap='jet')
    plt.colorbar(label='Magnitude squared of final amplitude, |c|^2')
    if verticalline is not None:
        plt.plot([verticalline, verticalline], [m_min, m_max], color='w', linestyle='--')
    plt.show()






if __name__ == "__main__":
    file = 'firstsim.npy'
    plot_angmom_vs_time(file)