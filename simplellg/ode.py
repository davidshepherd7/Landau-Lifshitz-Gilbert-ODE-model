
from math import sin, cos, tan, log, atan2, acos, pi, sqrt, exp
import scipy as sp
import scipy.integrate
import operator as op
from scipy.optimize import newton_krylov
import functools as ft
import itertools as it

import simplellg.utils as utils


# TODO:

# ADptive timestepping

def noaction(a,b): return a, b


def odeint(func, y0, tmax, dt = None, method = 'bdf2',
           actions_after_timestep = noaction):

    # Don't deal with adaptive stuff yet.
    if dt is None:
        raise NotImplementedError("Adaptive timestepping not implemented.")

    ts = [0.0] # ts is a list of times (floats)
    ys = map(sp.asarray, y0) # ys is a list of ndarrays

    # Construct the appropriate residual. Residuals should now be functions
    # of t, dt, yprev and ynp1 (only).
    if method.lower() == 'bdf2':
        residual = ft.partial(bdf_residual, 2, func)

        # If needed get another initial value using midpoint
        # method. Midpoint method is used because it maintains second order
        # accuracy.
        if len(y0) < 2:
            ts, ys = odeint(func, ys, ts[-1] + dt, dt, method = 'midpoint')

    elif method.lower() == 'bdf1':
        residual = ft.partial(bdf_residual, 1, func)
    elif method.lower() == 'midpoint':
        residual = ft.partial(midpoint_residual, func)
    else:
        raise ValueError("Method "+method+" not recognised.")

    # The main timestepping loop:
    while ts[-1] < tmax:
        tnp1 = ts[-1] + dt

        # Fill in the known values: t, dt and previous y values (in reverse
        # order, done as a slice) ready for the Newton solver.
        final_residual = ft.partial(residual, ts[-1], dt, ys[::-1])

        # Solve the system, using the previous y as an initial guess.
        ynp1 = newton_krylov(final_residual, ys[-1])

        # Update results
        ys.append(ynp1)
        ts.append(tnp1)

        # Execute any post-step actions requested (e.g. renormalisation,
        # simplified mid-point method update).
        ts, ys = actions_after_timestep(ts, ys)

    return ts, ys


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


#

# Testing
# ============================================================

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
