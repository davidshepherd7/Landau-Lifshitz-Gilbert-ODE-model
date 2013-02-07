
from math import sin, cos, tan, log, atan2, acos, pi, sqrt, exp
import scipy as sp
import scipy.integrate
import operator as op
from scipy.optimize import newton_krylov
import scipy.optimize.nonlin
import functools as ft
import itertools as it

import simplellg.utils as utils



class FailedTimestepError(Exception):
     def __init__(self, scaling_factor):
         self.scaling_factor = scaling_factor
     def __str__(self):
         return repr(self.scaling_factor)


def _timestep_scheme_dispatcher(label):
    """Pick the functions needed for this method."""
    label = label.lower()

    if label == 'bdf2':
        return bdf2_residual, None

    elif label == 'bdf2 ab':
        return bdf2_residual, bdf2_ab_time_adaptor

    elif label == 'bdf1':
        return bdf1_residual, None

    elif label == 'midpoint':
        return midpoint_residual, None

    else:
        raise ValueError("Method "+method+" not recognised.")


def odeint(func, y0, tmax, dt,
           method = 'bdf2',
           target_error = None,
           actions_after_timestep = None):

    # Select the method and adaptor
    time_residual, time_adaptor = _timestep_scheme_dispatcher(method)

    ts = [0.0] # List of times (floats)
    ys = map(sp.asarray, y0) # List y vectors (ndarrays)
    dts = [] # List of time steps (floats)

    if (method.lower() == 'bdf2' or method.lower() == 'bdf2 ab') \
        and len(y0) < 2:
        # If needed get another initial value using midpoint
        # method. Midpoint method is used because it maintains second order
        # accuracy.
        ys, ts, dts = odeint(func, ys, ts[-1] + dt, dt, method = 'midpoint')


    # Main timestepping loop
    # ============================================================
    while ts[-1] < tmax:

        # Fill in the residual for calculating dydt and the known values:
        # t, dt, previous y values (in reverse order, done as a slice)
        # ready for the Newton solver.
        residual = ft.partial(time_residual, func, ts[-1], dts+[dt], ys)

        # Try to solve the system, using the previous y as an initial
        # guess.
        try: ynp1 = newton_krylov(residual, ys[-1])
        except sp.optimize.nonlin.NoConvergence:
            # If it failed to converge then half the timestep and try again.
            dt /= 2
            continue

        # Update results (don't do this earlier in case the time step fails).
        ys.append(ynp1)
        ts.append(ts[-1] + dt)
        dts.append(dt)

        # Execute any post-step actions requested (e.g. renormalisation,
        # simplified mid-point method update).
        if actions_after_timestep is not None:
            ts, ys, dts = actions_after_timestep(ts, ys, dts)

        # Calculate the next value of dt if needed
        if time_adaptor is not None:
            dt = time_adaptor(ts, ys, target_error)

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


# Adaptive timestepping functions
# ============================================================

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

    # If the error is too bad then reject the step
    if scaling_factor < 0.4:
        raise FailedTimestepError

    # Using error estimate and target error calculate a new time step size.
    return scaling_factor * dt

    #

# Testing
# ============================================================
import matplotlib.pyplot as plt


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
        ys, ts, dts = odeint(residual, [[cos(0.0), exp(0.0)]], tmax, dt = 0.001,
                             method = method)
        # plt.plot(ts,ys,'--', ts, map(exp, ts), '-', ts, map(cos, ts))
        # dts = utils.dts_from_ts(ts)
        # plt.show()

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

        utils.assertAlmostEqual(ys[-1],exp(tmax), requested_tol * 2)
        utils.assertAlmostEqual(len(ys), steps, steps * 1.0/10)

    methods = [('bdf2 ab', 1e-3, 250),
               # ('bdf1', 1e-5, 100),
               # ('midpoint ab', 1e-5, 100)
               ]
    for meth, requested_tol, allowed_steps in methods:
        yield check_adaptive_dt, meth, requested_tol, allowed_steps
