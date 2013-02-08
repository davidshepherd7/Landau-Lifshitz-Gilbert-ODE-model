
from math import sin, cos, tan, log, atan2, acos, pi, sqrt, exp
import scipy as sp
import scipy.integrate
import operator as op
from scipy.optimize import newton_krylov
import scipy.optimize.nonlin
import functools as ft
import itertools as it

import simplellg.utils as utils

# PARAMETERS
MAX_ALLOWED_DT_SCALING_FACTOR = 4.0
MIN_ALLOWED_DT_SCALING_FACTOR = 0.8
TIMESTEP_FAILURE_DT_SCALING_FACTOR = 0.5

MIN_ALLOWED_TIMESTEP = 1e-8
MAX_ALLOWED_TIMESTEP = 1e8


# TODO

# Fix data flow in _odeint--having updated and non-updated lists around is
# strange...


class FailedTimestepError(Exception):
     def __init__(self, scaling_factor):
         self.scaling_factor = scaling_factor
     def __str__(self):
         return repr(self.scaling_factor)


def _timestep_scheme_dispatcher(label):
    """Pick the functions needed for this method. Returns functions for a
    time residual, a timestep adaptor and any intialisation actions needed.
    """
    label = label.lower()

    if label == 'bdf2':
        return bdf2_residual, None, ft.partial(higher_order_start,2)

    elif label == 'bdf2 ab':
        return bdf2_residual, bdf2_ab_time_adaptor, ft.partial(higher_order_start,2)

    elif label == 'bdf1':
        return bdf1_residual, None, None

    elif label == 'midpoint':
        return midpoint_residual, None, None

    elif label == 'midpoint ab':
        return midpoint_residual, midpoint_ab_time_adaptor, ft.partial(higher_order_start,2)

    else:
        s = "Method '"+label+"' not recognised."
        raise ValueError(s)


def higher_order_start(order, func, ys, ts, dts):
    """ Run a few steps of midpoint method, useful for generating extra
    initial data for multi-step methods.
    """
    while len(ys) < order:
        ys, ts, dts = _odeint(func, ys, ts, dts+[dts[-1]],\
                              ts[-1] + dts[-1], midpoint_residual)
    return ys, ts, dts


def odeint(func, y0, tmax, dt,
           method = 'bdf2',
           target_error = None,
           actions_after_timestep = None):

    # Select the method and adaptor
    time_residual, time_adaptor, initialisation_actions = \
      _timestep_scheme_dispatcher(method)

    ts = [0.0] # List of times (floats)
    ys = [sp.array(y0, ndmin=1)] # List of y vectors (ndarrays)
    dts = [dt] # List of time steps (floats)

    # Now call the actual function to do the work
    return _odeint(func, ys, ts, dts, tmax, time_residual,
                   target_error, time_adaptor, initialisation_actions,
                   actions_after_timestep)


def _odeint(func, ys, ts, dts, tmax, time_residual,
            target_error = None, time_adaptor = None,
            initialisation_actions = None, actions_after_timestep = None):

    if initialisation_actions is not None:
        ys, ts, dts = initialisation_actions(func, ys, ts, dts)

    # Get the dt requested and remove it from the list (in case it fails).
    dt = dts[-1]
    dts = dts[:-1]

    # Main timestepping loop
    # ============================================================
    while ts[-1] < tmax:

        # Fill in the residual for calculating dydt and the known values:
        # t, dt, previous y values (in reverse order, done as a slice)
        # ready for the Newton solver.
        residual = ft.partial(time_residual, func, ts[-1], dts+[dt], ys)

        # Try to solve the system, using the previous y as an initial
        # guess. If it fails reduce dt and try again.
        try: ynp1 = newton_krylov(residual, ys[-1], f_tol=1e10)
        except sp.optimize.nonlin.NoConvergence:
            dt = _scale_timestep(dt, TIMESTEP_FAILURE_DT_SCALING_FACTOR, True)
            continue

        # Execute any post-step actions requested (e.g. renormalisation,
        # simplified mid-point method update).
        uts, uys, udts = ts+[ts[-1]+dt], ys+[ynp1], dts+[dt]
        if actions_after_timestep is not None:
            uts, uys, udts = actions_after_timestep(uts, uys, udts)

        # Calculate the next value of dt if needed
        if time_adaptor is not None:
            try: dt = time_adaptor(uts, uys, target_error)
            except FailedTimestepError:
                dt = _scale_timestep(dt, TIMESTEP_FAILURE_DT_SCALING_FACTOR, True)
                continue

        # Update results storage (don't do this earlier in case the time step fails).
        ys.append(uys[-1])
        ts.append(uts[-1])
        dts.append(udts[-1])

    return ys, ts, dts


# Timestepper residual calculation functions
# ============================================================

def midpoint_residual(base_residual, t, dts, ys, ynp1):
    dt = dts[-1]
    yn = ys[-1]

    ymid = (ynp1 + yn)/2.0
    tmid = t + (dt/2.0)
    dydt = (ynp1 - yn)/dt
    return base_residual(tmid, ymid, dydt)


def bdf1_residual(base_residual, t, dts, ys, ynp1):
    dt = dts[-1]
    yn = ys[-1]

    dydt = (ynp1 - yn) / dt
    return base_residual(t+dt, ynp1, dydt)


def bdf2_residual(base_residual, t, dts, ys, ynp1):
    dt = dts[-1]
    dtprev = dts[-2]
    yn = ys[-1]
    ynm1 = ys[-2]

    a = (2*dt + dtprev)*dtprev
    prefactor = (2*dt + dtprev)/(dt + dtprev)
    alphas = [1.0/dt, -1.0/dt - dt/a, dt/a]

    dydt = prefactor * sp.dot(alphas, [ynp1, yn, ynm1])
    return base_residual(t+dt, ynp1, dydt)


def trapezoid_rule_residual(base_residual, t, dts, ys, ynp1):


    return base_residual()

# Adaptive timestepping functions
# ============================================================

def _scale_timestep(dt, scaling_factor, failed_step_flag=False):
    """ Scale dt by a scaling factor. Mostly this function is only needed
    for error checking.
    """

    # If the error is too bad (scaling factor too small) then reject the
    # step.
    if scaling_factor < MIN_ALLOWED_DT_SCALING_FACTOR \
      and not failed_step_flag:
        raise FailedTimestepError(scaling_factor)

    # If the scaling factor is really large just use the max
    if scaling_factor > MAX_ALLOWED_DT_SCALING_FACTOR:
        scaling_factor = MAX_ALLOWED_DT_SCALING_FACTOR


    # If the timestep would still get too big then return the max
    if scaling_factor * dt > MAX_ALLOWED_TIMESTEP:
        return MAX_ALLOWED_TIMESTEP

    # If the timestep would become too small then fail
    elif scaling_factor * dt < MIN_ALLOWED_TIMESTEP:
        error = "Tried to reduce dt to "+ str(scaling_factor * dt) +\
          " which is less than the minimum of "+ str(MIN_ALLOWED_TIMESTEP)
        raise ValueError(error)

    # Otherwise scale the timestep normally
    else:
        return scaling_factor * dt

def bdf2_ab_time_adaptor(ts, ys, target_error):
    """ Calculate a new timestep size based on estimating the error from
    the previous timestep. Weighting numbers stolen from oopmh-lib.
    """

    bdf_solutions = ys[-1]

    dt = ts[-1] - ts[-2]
    dtprev = ts[-2] - ts[-3]
    dtratio = (1.0 * dtprev) / dt

    # Set up weights
    weights = [0.0, 1 - dtratio**2, dtratio**2, (1 + dtratio) * dt]
    error_weight = (1 + dtratio)**2 / \
      (1.0 + 3*dtratio + 4*(dtratio**2) + 2*(dtratio**3))

    # Calculate error estimate
    predictor_solutions = sum(it.imap(op.mul, weights, ys[::-1]))
    errors_est = error_weight * (bdf_solutions - predictor_solutions)

    # Get a norm if we have an array, otherwise just take the absolute
    # value
    try: error_norm = sp.linalg.norm(errors_est, 2)
    except ValueError: error_norm = abs(errors_est)

    scaling_factor = ((target_error / error_norm)**0.33)
    return _scale_timestep(dt, scaling_factor)


def midpoint_ab_time_adaptor(ts, ys, target_error):
    """

    See notes: 7/2/2013 for the algebra on calculating the AB2
    solution. See "mathematica_adaptive_midpoint.m for algebra to get a
    local truncation error out of all this.
    """

    dt = ts[-1] - ts[-2]
    dtprev = ts[-2] - ts[-3]
    ynp1_MP = ys[-1]
    yn_MP = ys[-2]
    ynm1_MP = ys[-3]


    # Get explicit adams-bashforth 2 solution (variable timestep -- uses
    # steps at n + 1/2 and n - 1/2)
    # ============================================================

    # Get y derivatives at previous midpoints
    ydot_nphalf = (ys[-1] - ys[-2])/dt
    ydot_nmhalf = (ys[-2] - ys[-3])/dtprev

    # Calculate the corresponding timesteps
    AB_dt = 0.5*dt
    AB_dtprev = 0.5*dt + 0.5*dtprev

    # Calculate using AB2 variable timestep formula
    ynp1_AB2 = ys[-2] + 0.5*AB_dt*ydot_nphalf + \
      (AB_dt**2 * (ydot_nphalf - ydot_nmhalf)) / (2 * AB_dtprev)


    # Choose an estimate for the time derivatives of y at time tn
    # ============================================================

    # Finite difference dy/dt
    ydotn = (ynp1_MP - ynm1_MP)/(dt + dtprev)

    # Ignore d2y/dt2 for now (conservative estimate).
    ydotdotn = 0

    # alternatively FD it or something


    # Get the truncation error
    # ============================================================

    error = (-6.0 * (ynp1_AB2 - ynp1_MP) * ydotn + 5.0 * dt**3 * ydotdotn**2 ) \
        / (6.0 * ( -10.0 + ydotn))

    # Compute + use scaling factor
    # ============================================================

    # Get a norm if we have an array, otherwise just take the absolute
    # value
    try: error_norm = sp.linalg.norm(error, 2)
    except ValueError: error_norm = abs(error)

    scaling_factor = ((target_error / error_norm)**0.33)

    return _scale_timestep(dt, scaling_factor)


# Testing
# ============================================================
import matplotlib.pyplot as plt


def test_bad_timestep_handling():
    """ Check that rejected timesteps work.
    """
    def residual(t, y, dydt): return y - dydt
    tmax = 0.001

    dts = [1e-6, 1.0]
    ts = [0.0, 0.0+dts[0]]
    ys = map(lambda x: sp.array(exp(x), ndmin=1), ts)
    ys, ts, dts = _odeint(residual, ys, ts, dts, tmax,
                          midpoint_residual, 1e-5, midpoint_ab_time_adaptor)
    utils.assertAlmostEqual(ys[-1][0], exp(tmax), 10 * 1e-5)


def test_exp_timesteppers():

    # Auxilary checking function
    def check_exp_timestepper(method, tol):
        def residual(t, y, dydt): return y - dydt
        tmax = 1.0
        dt = 0.001
        ys, ts, dts = odeint(residual, [exp(0.0)], tmax, dt = dt,
                         method = method)
        utils.assertAlmostEqual(ys[-1], exp(tmax), tol)

    # List of test parameters
    methods = [('bdf2', 1e-5),
               ('bdf1', 1e-2), # First order method...
               ('midpoint', 1e-5),]

    # Generate tests
    for meth, tol in methods:
        yield check_exp_timestepper, meth, tol


def test_vector_timesteppers():

    # Auxilary checking function
    def check_vector_timestepper(method, tol):
        def residual(t, y, dydt):
            return sp.array([-1.0 * sin(t), y[1]]) - dydt
        tmax = 1.0
        ys, ts, dts = odeint(residual, [cos(0.0), exp(0.0)], tmax, dt = 0.001,
                             method = method)

        utils.assertAlmostEqual(ys[-1][0], cos(tmax), tol[0])
        utils.assertAlmostEqual(ys[-1][1], exp(tmax), tol[1])

    # List of test parameters
    methods = [('bdf2', [1e-4, 1e-4]),
               ('bdf1', [1e-2, 1e-2]), # First order methods suck...
               ('midpoint', [1e-4, 1e-4]),]

    # Generate tests
    for meth, tol in methods:
        yield check_vector_timestepper, meth, tol


def test_adaptive_dt():

    # Aux checking function
    def check_adaptive_dt(method, tol, steps):
        def residual(ts, ys, dydt): return ys - dydt
        tmax = 1.0
        ys, ts, dts = odeint(residual, [exp(0.0)], tmax, dt = 1e-6, method = method,
                             target_error = tol)

        # plt.plot(ts,ys,'--', ts, map(exp, ts))
        # dts = utils.dts_from_ts(ts)
        # plt.figure()
        # plt.plot(ts[:-1],dts)
        # plt.show()

        utils.assertAlmostEqual(ys[-1][0],exp(tmax), requested_tol * 5)
        utils.assertAlmostEqual(len(ys), steps, steps * 0.05)

    methods = [('bdf2 ab', 1e-3, 250),
               ('midpoint ab', 1e-3, 303),
               ]
    for meth, requested_tol, allowed_steps in methods:
        yield check_adaptive_dt, meth, requested_tol, allowed_steps
