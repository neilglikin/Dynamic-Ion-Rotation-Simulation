import ionsimulation as ion

savefile    = 'firstsim'
V0_i        = 1e4
f_rot_f     = 1e3
f_rot_frame = 0
V0_resid    = 0
theta_resid = 0
t_spinup    = 5e-3
t_hold1     = 5e-3
t_release   = 5e-3
t_hold2     = 5e-3

ion.simulate_linear_release(savefile, V0_i, f_rot_f, f_rot_frame, V0_resid, theta_resid, t_spinup, t_hold1, t_release, t_hold2)