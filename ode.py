
from math import sin, cos, tan, log, atan2, acos, pi, sqrt, exp
import scipy as sp
import scipy.integrate
import operator as op
from scipy.optimize import newton_krylov

import utils

# Hopefully in the style of scipy.integrate.ode* i.e.
# values_at_t = odeint(func, y0, t, args=(), Dfun=None,
#                      full_output=0, rtol=None, atol=None, tcrit=None)

# But we also need support for (easy) constant timestepping and for
# choosing the method used, so add dt and method optional parameters.


# TODO:

# Need to add option to input additional starting value, I think the extra
# lower order step is costing us accuracy.


def odeint(func, y0, tmax, dt = None, method = 'bdf2'):
    """

    func: should be a function of time, the previous y values and dydt
    which gives a residual.

    e.g. dy/dt = -y

    becomes

    def func(t, y, dydt): return dydt + t
    """
    # Don't deal with adaptive stuff yet.
    if dt is None:
        raise NotImplementedError("Adaptive timestepping not implemented.")

    if method.lower() == 'bdf2':
        return bdf(func, y0, tmax, dt = dt, order = 2)
    elif method.lower() == 'bdf1':
        return bdf(func, y0, tmax, dt = dt, order = 1)
    elif method.lower() == 'midpoint':
        return midpoint(func, y0, tmax, dt = dt)
    else:
        raise ValueError("Method "+method+" not recognised.")


def bdf_coeffs(order, name):
    """Get coefficients for bdf methods. From Atkinson, Numerical Solution
    of Ordinary Differential Equations.
    """
    b_cs = [{'beta' : 1.0, 'alphas' : [1.0]},
            {'beta' : 2.0/3, 'alphas' : [4.0/3, -1.0/3]},
            {'beta': 6.0/11, 'alphas' : [18.0/11 -9.0/11, 2.0/11]},]
    return b_cs[order-1][name]

def bdf(func, y0, tmax, dt, order):

    # Get the bdf coefficients
    alphas = bdf_coeffs(order, 'alphas')
    beta =  bdf_coeffs(order, 'beta')

    # ts is a list of times (floats)
    ts = [0.0]

    # ys is a list of y values, possibly a tuple, vector or float. It must
    # be the same as what the newton solve retuns.
    ys = map(sp.asarray,y0)

    # Run some lower order bdf steps to get starting y values if needed
    if len(ys) < order:
        temp_ts, ys = bdf(func, ys, ts[-1] + dt, dt, order - 1)
        ts.append(temp_ts[-1])

    # The main timestepping loop:
    while ts[-1] < tmax:

        # Get most recent order+1 values of y
        y_prev = sp.array(ys[-1 : -1*(order+1) : -1])

        # Form the function to minimise
        def discretised_residual_func(ynp1):
            """Compute the residual from the given func, current time,
            previous y values and the input (an "ansatz" for y at the next
            time).
            """
            # ??ds not sure why there is a -1 factor here...
            dydt = (ynp1 - sum(map(op.mul, y_prev, alphas))) / (beta * dt)
            tnp1 = ts[-1] + dt
            return func(tnp1, ynp1, dydt)

        # Solve the system using the previous y as an initial guess
        ynp1 = newton_krylov(discretised_residual_func, ys[-1])

        print
        print
        # Update results
        ys.append(ynp1)
        ts.append(ts[-1] + dt)

    # ??ds convert to correct return type?

    return ts, ys

# ??ds refactor to combine with bdf
def midpoint(func, y0, tmax, dt):

    # ts is a list of times (floats)
    ts = [0.0]

    # ys is a list of y values, possibly a tuple, vector or float. It must
    # be the same as what the newton solve retuns.
    ys = map(sp.asarray,y0)

    while ts[-1] < tmax:

        # Form the function to minimise
        def discretised_residual_func(ynp1):
            """Compute the residual from the given func, current time,
            previous y values and the input (an "ansatz" for y at the next
            time).
            """
            ymid = (ynp1 + ys[-1])/2.0
            tmid = ts[-1] + (dt/2.0)
            dydt = (ynp1 - ys[-1])/dt
            return func(tmid, ymid, dydt)

        # Solve the system using the previous y as an initial guess
        ynp1 = newton_krylov(discretised_residual_func, ys[-1])

        # Update results
        ys.append(ynp1)
        ts.append(ts[-1] + dt)

    return ts, ys

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
