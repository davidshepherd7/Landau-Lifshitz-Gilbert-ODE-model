
from math import sin, cos, tan, log, atan2, acos, pi, sqrt, exp
import scipy as sp
import scipy.integrate
import operator as op

import utils

# Hopefully in the style of scipy.integrate.ode* i.e.
# values_at_t = odeint(func, y0, t, args=(), Dfun=None,
#                      full_output=0, rtol=None, atol=None, tcrit=None)

# But we also need support for (easy) constant timestepping and for
# choosing the method used, so add dt and method optional parameters.


# TODO:

# Need to add option to input additional starting value, I think the extra
# lower order step is costing us accuracy.

# Possibly need to do something better with the finite differenced
# Jacobian, it's not very accurate.

def odeint(func, y0, t, dt = None, method = 'bdf2'):
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
        return bdf(func, y0, t, dt = dt, order = 2)
    elif method.lower() == 'bdf1':
        return bdf(func, y0, t, dt = dt, order = 1)
    elif method.lower() == 'midpoint':
        return midpoint(func, y0, t, dt = dt)
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
            dydt = -1 * (sum(map(op.mul, y_prev, alphas)) - ynp1) / (beta * dt)
            tnp1 = ts[-1] + dt
            return func(tnp1, ynp1, dydt)

        # Solve the system using the previous y as an initial guess
        ynp1 = newton_solve(discretised_residual_func,
                            ys[-1], tol=1e-8)

        # Update results
        ys.append(ynp1)
        ts.append(ts[-1] + dt)

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
        ynp1 = newton_solve(discretised_residual_func,
                            ys[-1], tol=1e-8)

        # Update results
        ys.append(ynp1)
        ts.append(ts[-1] + dt)

    return ts, ys

def newton_solve(func, x0, tol=1e-8):
    """Find x such that max(func(x)) < tol. Optionally specify an initial
     guess x0, otherwise zeroes will be used.

    E.g. if f(x) = residuals then the result is the solution after a timestep.
    """

    # Make sure this is an array otherwise solvers will fail
    residual = sp.array(func(x0), ndmin = 1)
    max_res = sp.amax(abs(residual))

    # Initialise x, must be an array so that += works
    x = sp.array(x0, ndmin = 1)

    while max_res > tol:
        # Calculate the Jacobian of f by finite differencing.
        jacobian = fd_jacobian(func, x)

        # Solve the system to get dx. We want dx such that f(x + dx) = 0
        # (to within tol). ??ds offer other solvers?
        dx = sp.linalg.solve(jacobian, -1 * residual)

        # Update to the new and improved x
        x += dx

        # Update residual
        residual = sp.array(func(x), ndmin = 1)
        max_res = sp.amax(abs(residual))

    return x


def fd_jacobian(func, x):
    # Need to make sure these are vectors with dimension 1.
    fx = sp.array(func(x), ndmin = 1)
    x = sp.array(x, ndmin = 1)

    dx = 1e-8

    # Matrix with one row for each output of f and one column for each x_j.
    J = sp.zeros((fx.shape[0], x.shape[0]))

    for j, xj in enumerate(x):
        temp_x = x.copy() # Explicit copy otherwise we end up modifying a
                          # "view" of x.
        temp_x[j] += dx
        J[:,j] = (func(temp_x) - fx) / dx

    return J


    #

# Testing
# ============================================================

def test_fd_jacobian():
    def f(x): return sp.array([ x[0]**2 + x[1]**2, x[1]])
    def J_f(x): return sp.array([[2*x[0], 2*x[1]], [0.0, 1.0]])
    x = sp.array([3.0, 5.0])
    J = fd_jacobian(f, x)
    Jdiff = J - J_f(x)
    utils.assertAlmostZero(sp.amax(Jdiff), 1e-5)


# Check that we can solve for a vector f using the newton solve.
def test_vector_newton_solve():
    def f(x):
        # From http://v8doc.sas.com/sashtml/iml/chap8/sect4.htm
        return sp.array([x[0] + x[1] - x[0]*x[1] +2, x[0] * exp(-x[1]) -1])

    x_result = newton_solve(f, sp.zeros(2))
    x_error = sp.amax(x_result - [0.0977731, -2.325106])
    utils.assertAlmostZero(x_error, 1e-6)


# Check that we can solve for sqrt(4) correctly using the newton solver.
def test_scalar_newton_solve():
    def f(x): return x[0]**2 - 4
    x = newton_solve(f, sp.array([1.0]))
    x_error = x - sp.array(2,ndmin=1)
    utils.assertAlmostZero(x_error, 1e-6)


def test_exp_timesteppers():

    # Auxilary checking function
    def check_exp_timestepper(method, tol):
        def residual(t, y, dydt): return y - dydt
        tmax = 1.0
        dt = 0.001
        ts, ys = odeint(residual, [exp(0.0)], tmax, dt = dt,
                         method = method)
        utils.assertAlmostEqual(ys[-1][0], exp(tmax), tol)

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
