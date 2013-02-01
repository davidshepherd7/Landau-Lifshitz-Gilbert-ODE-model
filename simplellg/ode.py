
from math import sin, cos, tan, log, atan2, acos, pi, sqrt, exp
import scipy as sp
import scipy.integrate
import operator as op
from scipy.optimize import newton_krylov
import scipy.optimize.nonlin
import functools as ft
import itertools as it

import simplellg.utils as utils


# TODO:

# ADptive timestepping

def noaction(a,b): return a, b


def odeint(func, y0, tmax, dt,
           method = 'bdf2',
           target_error = None,
           adaptive = False,
           actions_after_timestep = noaction):

    ts = [0.0] # ts is a list of times (floats)
    ys = map(sp.asarray, y0) # ys is a list of ndarrays

    # Construct the appropriate residual. Residuals should now be functions
    # of t, dt, yprev and ynp1 (only).
    # ============================================================
    if method.lower() == 'bdf2':
        residual = ft.partial(bdf_residual, 2, func)
        default_dt_adaptor = bdf2_dt_adaptor

        # If needed get another initial value using midpoint
        # method. Midpoint method is used because it maintains second order
        # accuracy.
        if len(y0) < 2:
            ts, ys = odeint(func, ys, ts[-1] + dt, dt, method = 'midpoint')

    elif method.lower() == 'bdf1':
        residual = ft.partial(bdf_residual, 1, func)
        default_dt_adaptor = None
    elif method.lower() == 'midpoint':
        residual = ft.partial(midpoint_residual, func)
        default_dt_adaptor = None
    else:
        raise ValueError("Method "+method+" not recognised.")

    # Assign a timestep adaptor
    # ============================================================
    if hasattr(adaptive, '__call__'):
        dt_adaptor = adaptive
    elif adaptive is True:
        dt_adaptor = default_dt_adaptor
    elif adaptive is False:
        dt_adaptor = lambda ts,ys,tol:ts[-1] - ts[-2] # Dummy
    else:
        raise ValueError("adaptive value " + str(adaptive) + "not recognised.")


    # Main timestepping loop
    # ============================================================
    while ts[-1] < tmax:

        tnp1 = ts[-1] + dt

        # Fill in the known values: t, dt and previous y values (in reverse
        # order, done as a slice) ready for the Newton solver.
        final_residual = ft.partial(residual, ts[-1], dt, ys[::-1])

        # Solve the system, using the previous y as an initial guess.
        try: ynp1 = newton_krylov(final_residual, ys[-1])
        except sp.optimize.nonlin.NoConvergence:
            # If it failed to converge then half the timestep and try again.
            dt /= 2
            continue

        # Update results
        ys.append(ynp1)
        ts.append(tnp1)

        # Execute any post-step actions requested (e.g. renormalisation,
        # simplified mid-point method update).
        ts, ys = actions_after_timestep(ts, ys)

        # Calculate the next value of dt using whatever function has been
        # set.
        dt = dt_adaptor(ts, ys, target_error)

    return ts, ys


# Timestepper residual calculation functions
# ============================================================

def bdf_coeffs(order, name):
    """Get coefficients for bdf methods. From Atkinson, Numerical Solution
    of Ordinary Differential Equations.
    """
    b_cs = [{'beta' : 1.0, 'alphas' : [1.0]},
            {'beta' : 2.0/3, 'alphas' : [4.0/3, -1.0/3]},
            {'beta': 6.0/11, 'alphas' : [18.0/11 -9.0/11, 2.0/11]},]
    return b_cs[order-1][name]


def midpoint_residual(base_residual, t, dt, y_prev, ynp1):
    ymid = (ynp1 + y_prev[0])/2.0
    tmid = t + (dt/2.0)
    dydt = (ynp1 - y_prev[0])/dt
    return base_residual(tmid, ymid, dydt)


def bdf_residual(order, base_residual, t, dt, y_prev, ynp1):
    alphas = bdf_coeffs(order, 'alphas')
    beta =  bdf_coeffs(order, 'beta')
    dydt = (ynp1 - sum(it.imap(op.mul, alphas, y_prev))) / (beta * dt)
    return base_residual(t + dt, ynp1, dydt)

# Adaptive timestepping functions
# ============================================================

def bdf2_dt_adaptor(ts, ys, target_error):
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

    # Using error estimate and target error calculate a new time step size.
    return ((target_error / error_norm)**0.33) * dt

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
        ts, ys = odeint(residual, [exp(0.0)], tmax, dt = dt,
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
            return sp.array([-1 * sin(t), y[1]]) - dydt
        tmax = 1.0
        ts, ys = odeint(residual, [[cos(0.0), exp(0.0)]], tmax, dt = 0.001,
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
        ts, ys = odeint(residual, [exp(0.0)], tmax, dt = 1e-6, method = method,
                        adaptive=True, target_error = tol)

        # plt.plot(ts,ys,'--', ts, map(exp, ts))
        # dts = utils.dts_from_ts(ts)
        # plt.figure()
        # plt.plot(ts[:-1],dts)
        # plt.show()

        utils.assertAlmostEqual(ys[-1],exp(tmax), requested_tol * 2)
        utils.assertAlmostEqual(len(ys), steps, steps * 1.0/10)

    methods = [('bdf2', 1e-3, 328),
               # ('bdf1', 1e-5, 100),
               # ('midpoint', 1e-5, 100)
               ]
    for meth, requested_tol, allowed_steps in methods:
        yield check_adaptive_dt, meth, requested_tol, allowed_steps
