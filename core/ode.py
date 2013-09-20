from __future__ import division
from __future__ import absolute_import

import scipy as sp
import scipy.integrate
import scipy.linalg
import scipy.optimize
import functools as ft
import itertools as it
import copy
import sys
import random
import sympy

from math import sin, cos, tan, log, atan2, acos, pi, sqrt, exp
from scipy.interpolate import krogh_interpolate
from scipy.linalg import norm
from functools import partial as par


import simpleode.core.utils as utils


# PARAMETERS
MAX_ALLOWED_DT_SCALING_FACTOR = 3.0
MIN_ALLOWED_DT_SCALING_FACTOR = 0.75
TIMESTEP_FAILURE_DT_SCALING_FACTOR = 0.5

MIN_ALLOWED_TIMESTEP = 1e-8
MAX_ALLOWED_TIMESTEP = 1e8


# TODO:

# Move time adaptivity try/except into general_time_adaptor, use
# general_time_adaptor everywhere.

# Make arguments and order consistent: ts, ys, dys, others. Including
# stepper functions!

# Rename order->n_start in higher order starts


# Data storage notes
# ============================================================

# Y values and time values are stored in lists throughout (for easy
# appending).

# The final value in each of these lists (accessed with [-1]) is the most
# recent.

# Almost always the most recent value in the list is the current
# guess/result for y_np1/t_np1 (i.e. y_{n+1}, t_{n+1}, i.e. the value being
# calculated), the previous is y_n, etc.

# Denote values of y/t at timestep using eg. y_np1 for y at step n+1. Use
# 'h' for 0.5, 'm' for minus, 'p' for plus (since we can't include .,-,+ in
# variable names).



# Random notes:
# ============================================================

# Try to always use fractions rather than floating point values because
# this allows us to use the same functions in sympy for algebraic
# computations without loss of accuracy.


class FailedTimestepError(Exception):
    def __init__(self, new_dt):
        self.new_dt = new_dt

    def __str__(self):
        return "Exception: timestep failed, next timestep should be "\
            + repr(self.new_dt)


class ConvergenceFailure(Exception): pass


def _timestep_scheme_factory(method):
    """Construct the functions for the named method. Input is either a
    string with the name of the method or a dict with the name and a list
    of parameters.

    Returns a triple of functions for:
    * a time residual
    * a timestep adaptor
    * intialisation actions
    """

    _method_dict = {}

    # If it's a dict then get the label attribute, otherwise assume it's a
    # string...
    if isinstance(method, dict):
        _method_dict = method
    else:
        _method_dict = {'label' : method}

    label = _method_dict.get('label').lower()

    if label == 'bdf2':
        return bdf2_residual, None, par(higher_order_start, 2)

    elif label == 'bdf3':
        return bdf3_residual, None, par(higher_order_start, 3)

    elif label == 'bdf2 mp':
        adaptor = par(general_time_adaptor,
                      lte_calculator=bdf2_mp_lte_estimate,
                      method_order=2)
        return (bdf2_residual, adaptor, par(higher_order_start, 3))

    elif label == 'bdf2 ebdf3':
        dydt_func = _method_dict.get('dydt_func', None)
        lte_est = par(ebdf3_lte_estimate, dydt_func=dydt_func)
        adaptor = par(general_time_adaptor,
                      lte_calculator=lte_est,
                      method_order=2)
        return (bdf2_residual, adaptor, par(higher_order_start, 4))

    elif label == 'bdf1':
        return bdf1_residual, None, None

    elif label == 'imr':
        return imr_residual, None, None

    elif label == 'imr ab':
        # Get values from dict (with defaults if not set).
        n_start = _method_dict.get('n_start', 2)
        ab_start = _method_dict.get('ab_start_point', 't_n')
        use_y_np1_in_interp = _method_dict.get('use_y_np1_in_interp', False)
        explicit_derivative = _method_dict.get('explicit_derivative', None)
        fudge_factor = _method_dict.get('fudge_factor', 1.0)

        interp = _method_dict.get('interpolator',
                                  par(my_interpolate, n_interp=n_start,
                                      use_y_np1_in_interp=use_y_np1_in_interp))

        adaptor = par(general_time_adaptor,
                      lte_calculator=imr_ab_lte_estimate,
                      method_order=2,
                      ab_start_point=ab_start,
                      interpolator=interp,
                      explicit_derivative=explicit_derivative,
                      fudge_factor=fudge_factor)

        return imr_residual, adaptor, par(higher_order_start, n_start)

    elif label == 'imr ebdf3':
        dydt_func = _method_dict.get('dydt_func', None)
        lte_est = par(ebdf3_lte_estimate, dydt_func=dydt_func)
        adaptor = par(general_time_adaptor,
                      lte_calculator=lte_est,
                      method_order=2)
        return imr_residual, adaptor, par(higher_order_start, 5)

    elif label == 'imr ebdf3dynm1':
        dydt_func = _method_dict.get('dydt_func', None)
        lte_est = par(ebdf3_dynm1_lte_estimate, dydt_func=dydt_func)
        adaptor = par(general_time_adaptor,
                      lte_calculator=lte_est,
                      method_order=2)
        return imr_residual, adaptor, par(higher_order_start, 5)

    elif label == 'imr f13':

        F_func = _method_dict.get('F_func', None)

        lte_est = par(f13_lte_est, F_func=F_func)
        adaptor = par(general_time_adaptor,
                      lte_calculator=lte_est,
                      method_order=2)
        return imr_residual, adaptor, par(higher_order_start, 5)

    elif label == 'imr w18':
        import simpleode.algebra.two_predictor as tp
        p1 = _method_dict['p1']
        p2 = _method_dict['p2']
        ynph_approximation = _method_dict.get('ynph_approx', "bdf2")
        dynph_approximation = _method_dict.get('dynph_approx', "imr")
        lte_est = tp.generate_predictor_pair_lte_est(p1, p2, ynph_approximation,
                                                     dynph_approximation)
        adaptor = par(general_time_adaptor,
                      lte_calculator=lte_est,
                      method_order=2)
        return imr_residual, adaptor, par(higher_order_start, 5)

    elif label == 'trapezoid':
        # TR is actually self starting but due to technicalities with
        # getting derivatives of y from implicit formulas we need an extra
        # starting point.
        return TrapezoidRuleResidual(), None, par(higher_order_start, 2)

    elif label == 'tr ab':

        dydt_func = _method_dict.get('dydt_func')

        adaptor = par(general_time_adaptor,
                      lte_calculator=tr_ab_lte_estimate,
                      dydt_func=dydt_func,
                      method_order=2)

        return TrapezoidRuleResidual(), adaptor, par(higher_order_start, 2)

    else:
        message = "Method '"+label+"' not recognised."
        raise ValueError(message)


def higher_order_start(order, func, ys, ts):
    """ Run a few steps of imr with a very small timestep.
    Useful for generating extra initial data for multi-step methods.
    """
    starting_dt = 1e-6
    while len(ys) < order:
        ys, ts = _odeint(func, ys, ts, starting_dt, ts[-1] + starting_dt,
                         imr_residual)
    return ys, ts


def odeint(func, y0, tmax, dt, method='bdf2', target_error=None, **kwargs):
    """
    Integrate the residual "func" with initial value "y0" to time
    "tmax". If non-adaptive (target_error=None) then all steps have size
    "dt", otherwise "dt" is used for the first step and later steps are
    automatically decided using the adaptive scheme.

    newton_tol : specify Newton tolerance (used for minimisation of residual).

    actions_after_timestep : function to modify t_np1 and y_np1 after
    calculation (takes ts, ys as input args, returns modified t_np1, y_np1).

    Actually just a user friendly wrapper for _odeint. Given a method name
    (or dict of time integration method parameters) construct the required
    functions using _timestep_scheme_factory(..), set up data storage and
    integrate the ODE using _odeint.

    Any other arguments are just passed down to _odeint, which passes extra
    args down to the newton solver.
    """

    # Select the method and adaptor
    time_residual, time_adaptor, initialisation_actions = \
        _timestep_scheme_factory(method)

    # Check adaptivity arguments for consistency.
    if target_error is None and time_adaptor is not None:
        raise ValueError("Adaptive time stepping requires a target_error")
    if target_error is not None and time_adaptor is None:
        raise ValueError("Adaptive time stepping requires an adaptive method")


    ts = [0.0]  # List of times (floats)
    ys = [sp.array(y0, ndmin=1)]  # List of y vectors (ndarrays)

    # Now call the actual function to do the work
    return _odeint(func, ys, ts, dt, tmax, time_residual,
                   target_error, time_adaptor, initialisation_actions,
                   **kwargs)


def _odeint(func, ys, ts, dt, tmax, time_residual,
            target_error=None, time_adaptor=None,
            initialisation_actions=None, actions_after_timestep=None,
            **kwargs):
    """Underlying function for odeint.
    """

    if initialisation_actions is not None:
        ys, ts = initialisation_actions(func, ys, ts)

    # Main timestepping loop
    # ============================================================
    while ts[-1] < tmax:

        t_np1 = ts[-1] + dt

        # Fill in the residual for calculating dydt and the previous time
        # and y values ready for the Newton solver. Don't use a lambda
        # function because it confuses the profiler.
        def residual(y_np1):
            return time_residual(func, ts+[t_np1], ys+[y_np1])

        # Try to solve the system, using the previous y as an initial
        # guess. If it fails reduce dt and try again.
        try:
            y_np1 = newton(residual, ys[-1], **kwargs)
        except sp.optimize.nonlin.NoConvergence:
            dt = scale_timestep(dt, None, None, None,
                                scaling_function=failed_timestep_scaling)
            sys.stderr.write("Failed to converge, reducing time step.\n")
            continue

        # Execute any post-step actions requested (e.g. renormalisation,
        # simplified mid-point method update).
        if actions_after_timestep is not None:
            new_t_np1, new_y_np1 = actions_after_timestep(
                ts+[t_np1], ys+[y_np1])
            # Note: we store the results in new variables so that we can
            # easily discard this step if it fails.
        else:
            new_t_np1, new_y_np1 = t_np1, y_np1

        # Calculate the next value of dt if needed
        if time_adaptor is not None:
            try:
                dt = time_adaptor(ts+[new_t_np1], ys+[new_y_np1], target_error)

            # If the scaling factor is too small then don't store this
            # timestep, instead repeat it with the new step size.
            except FailedTimestepError, exception_data:
                sys.stderr.write('Rejected time step\n')
                dt = exception_data.new_dt
                continue

        # Update results storage (don't do this earlier in case the time
        # step fails).
        ys.append(new_y_np1)
        ts.append(new_t_np1)

    return ys, ts


def higher_order_explicit_start(order, func, ts, ys):
    starting_dt = 1e-6
    while len(ys) < order:
        ys, ts = _odeint_explicit(func, ts, ys, starting_dt,
                                  ts[-1] + starting_dt,
                                  emr_step)

    return ts, ys


def odeint_explicit(func, y0, dt, tmax, method='ab2', time_adaptor=None,
                    target_error=None, **kwargs):
    """Fairly naive implementation of constant step explicit time stepping
    for linear odes.
    """
    # Set up starting values
    ts = [0.0]
    ys = [sp.array([y0], ndmin=1)]

    if method == 'ab2':
        def wrapped_ab2(ts, ys, func):
            return ab2_step(ts[-1] - ts[-2], ys[-2], func(ts[-2], ys[-2]),
                            ts[-2] - ts[-3], func(ts[-3], ys[-3]))

        stepper = wrapped_ab2
        n_start = 2

    elif method == "ebdf2":
        def wrapped_ebdf2(ts, ys, func):
            return ebdf2_step(ts[-1] - ts[-2], ts[-2], func(ts[-2], ys[-2]),
                              ts[-2] - ts[-3], ys[-3])
        stepper = wrapped_ebdf2
        n_start = 2

    elif method == "ebdf3":
        def wrapped_ebdf3(ts, ys, func):
            return ebdf3_step(ts[-1] - ts[-2], ys[-2], func(ts[-2], ys[-2]),
                              ts[-2] - ts[-3], ys[-3],
                              ts[-3] - ts[-4], ys[-4])
        stepper = wrapped_ebdf3
        n_start = 3

    else:
        raise NotImplementedError("method "+method+" not implement (yet?)")


    # Generate enough values to start the main method (using emr)
    ts, ys = higher_order_explicit_start(n_start, func, ts, ys)

    # Call the real stepping function
    return _odeint_explicit(func, ts, ys, dt, tmax, stepper, time_adaptor)


def _odeint_explicit(func, ts, ys, dt, tmax,
                     stepper, target_error=None, time_adaptor=None):

    while ts[-1] < tmax:

        # Step (note: to be similar to implicit code pass dummy value of
        # ynp1 into stepper)
        t_np1 = ts[-1] + dt
        y_np1 = stepper(ts+[t_np1], ys+[None], func)

        # Calculate the next value of dt if needed
        if time_adaptor is not None:
            try:
                dt = time_adaptor(ts+[t_np1], ys+[y_np1], target_error)

            # If the scaling factor is too small then don't store this
            # timestep, instead repeat it with the new step size.
            except FailedTimestepError, exception_data:
                sys.stderr.write('Rejected time step\n')
                dt = exception_data.new_dt
                continue

        # Store values
        ys.append(y_np1)
        ts.append(t_np1)

    return ys, ts





# Newton solver and helpers
# ============================================================


def newton(residual, x0, jacobian_func=None, newton_tol=1e-8,
           solve_function=None,
           jacobian_fd_eps=1e-10, max_iter=20):
    """Find the minimum of the residual function using Newton's method.

    Optionally specify a Jacobian calculation function, a tolerance and/or
    a function to solve the linear system J.dx = r.

    If no Jacobian_Func is given the Jacobian is finite differenced.
    If no solve function is given then sp.linalg.solve is used.

    Norm for measuring residual is max(abs(..)).
    """
    if jacobian_func is None:
        jacobian_func = par(finite_diff_jacobian, residual, eps=jacobian_fd_eps)

    if solve_function is None:
        solve_function = sp.linalg.solve

    # Wrap the solve function to deal with non-matrix cases (i.e. when we
    # only have one degree of freedom and the "Jacobian" is just a number).
    def wrapped_solve(A, b):
        if len(b) == 1:
            return b/A[0][0]
        else:
            try:
                return solve_function(A, b)
            except scipy.linalg.LinAlgError:
                print "\n", A, b, "\n"
                raise

    # Call the real Newton solve function
    return _newton(residual, x0, jacobian_func, newton_tol,
                   wrapped_solve, max_iter)


def _newton(residual, x0, jacobian_func, newton_tol, solve_function, max_iter):
    """Core function of newton(...)."""

    if max_iter <= 0:
        raise sp.optimize.nonlin.NoConvergence

    r = residual(x0)

    # If max entry is below newton_tol then return
    if sp.amax(abs(r)) < newton_tol:
        return x0

    # Otherwise reduce residual using Newtons method + recurse
    else:
        J = jacobian_func(x0)
        dx = solve_function(J, r)
        return _newton(residual, x0 - dx, jacobian_func, newton_tol,
                       solve_function, max_iter - 1)


def finite_diff_jacobian(residual, x, eps):
    """Calculate the matrix of derivatives of the residual w.r.t. input
    values by finite differencing.
    """
    n = len(x)
    J = sp.empty((n, n))

    # For each entry in x
    for i in range(0, n):
        xtemp = x.copy() # Force a copy so that we don't modify x
        xtemp[i] += eps
        J[:,i] = (residual(xtemp) - residual(x))/eps

    return J



# Interpolation helpers
# ============================================================

def my_interpolate(ts, ys, n_interp, use_y_np1_in_interp=False):
    # Find the start and end of the slice of ts, ys that we want to use for
    # interpolation.
    start = -n_interp if use_y_np1_in_interp else -n_interp - 1
    end = None if use_y_np1_in_interp else -1

    # Nasty things could go wrong if you try to start adapting with not
    # enough points because [-a:-b] notation lets us go past the ends of
    # the list without throwing an error! Check it!
    assert len(ts[start:end]) == n_interp

    # Actually interpolate the values
    t_nph = (ts[-1] + ts[-2])/2
    t_nmh = (ts[-2] + ts[-3])/2
    interps = krogh_interpolate(
        ts[start:end], ys[start:end], [t_nmh, ts[-2], t_nph], der=[0, 1, 2])

    # Unpack (can't get "proper" unpacking to work)
    dy_nmh = interps[1][0]
    dy_n = interps[1][1]
    y_nph = interps[0][2]
    dy_nph = interps[1][2]
    ddy_nph = interps[2][2]

    return dy_nmh, y_nph, dy_nph, ddy_nph, dy_n


def imr_approximation_fake_interpolation(ts, ys):
    # Just use imr approximation for "interpolation"!

    dt_n = (ts[-1] + ts[-2])/2
    dt_nm1 = (ts[-2] + ts[-3])/2

    # Use imr average approximations
    y_nph = (ys[-1] + ys[-2])/2
    dy_nph = (ys[-1] - ys[-2])/dt_n
    dy_nmh = (ys[-2] - ys[-3])/dt_nm1

    # Finite diff it
    ddy_nph = (dy_nph - dy_nmh) / (dt_n/2 + dt_nm1/2)

    return dy_nmh, y_nph, dy_nph, ddy_nph, None


# Timestepper residual calculation functions
# ============================================================

def ab2_step(dt_n, y_n, dy_n, dt_nm1, dy_nm1):
    """Take a single step of the Adams-Bashforth 2 method.
    """
    dtr = dt_n / dt_nm1
    y_np1 = y_n + (dt_n/2)*((2 + dtr)*dy_n - dtr*dy_nm1)
    return y_np1


def ebdf2_step(dt_n, y_n, dy_n, dt_nm1, y_nm1):
    """Take a single step of the explicit midpoint rule.
    From G&S pg. 715 and Prinja's thesis pg.45.
    """
    dtr = dt_n / dt_nm1
    y_np1 = (1 - dtr**2)*y_n + (1 + dtr)*dt_n*dy_n + (dtr**2)*(y_nm1)
    return y_np1


def emr_step(ts, ys, func):
    dtn = ts[-1] - ts[-2]

    tn = ts[-2]
    yn = ys[-2]

    tnph = ts[-1] + dtn/2
    ynph = yn + (dtn/2)*func(tn, yn)

    ynp1 = yn + dtn * func(tnph, ynph)

    return ynp1


def ibdf2_step(dtn, yn, dynp1, dtnm1, ynm1):
    """Take an implicit (normal) bdf2 step, must provide the derivative or
    some approximation to it. For solves use residuals instead.

    From Gresho and Sani pg.715.
    """

    # ??ds possibly not the most stable formulation
    # numerically... differences of similar things!
    return yn + dtn*(dtn/(2*dtn + dtnm1)) *((yn - ynm1)/dtnm1) + dtn*(dtn + dtnm1)/(2*dtn + dtnm1)*dynp1

def ibdf3_step(dynp1, dtn, yn, dtnm1, ynm1, dtnm2, ynm2):
    return dynp1*(dtn**3*dtnm1**2*dtnm2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + dtn**3*dtnm1*dtnm2**2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + 2*dtn**2*dtnm1**3*dtnm2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + 3*dtn**2*dtnm1**2*dtnm2**2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + dtn**2*dtnm1*dtnm2**3/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + dtn*dtnm1**4*dtnm2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + 2*dtn*dtnm1**3*dtnm2**2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + dtn*dtnm1**2*dtnm2**3/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3)) + yn*(dtn**4*dtnm2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + 4*dtn**3*dtnm1*dtnm2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + 2*dtn**3*dtnm2**2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + 6*dtn**2*dtnm1**2*dtnm2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + 6*dtn**2*dtnm1*dtnm2**2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + dtn**2*dtnm2**3/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + 4*dtn*dtnm1**3*dtnm2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + 6*dtn*dtnm1**2*dtnm2**2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + 2*dtn*dtnm1*dtnm2**3/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + dtnm1**4*dtnm2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + 2*dtnm1**3*dtnm2**2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + dtnm1**2*dtnm2**3/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3)) + ynm1*(-dtn**4*dtnm1/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - dtn**4*dtnm2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - 2*dtn**3*dtnm1**2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - 4*dtn**3*dtnm1*dtnm2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - 2*dtn**3*dtnm2**2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - dtn**2*dtnm1**3/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - 3*dtn**2*dtnm1**2*dtnm2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - 3*dtn**2*dtnm1*dtnm2**2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - dtn**2*dtnm2**3/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3)) + ynm2*(dtn**4*dtnm1/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + 2*dtn**3*dtnm1**2/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + dtn**2*dtnm1**3/(3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + 4*dtn*dtnm1**3*dtnm2 + 6*dtn*dtnm1**2*dtnm2**2 + 2*dtn*dtnm1*dtnm2**3 + dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3))


def ebdf3_step_wrapper(ts, ys, dyn):
    """Get required values from ts, ys vectors and call ebdf3_step.
    """

    dtn = ts[-1] - ts[-2]
    dtnm1 = ts[-2] - ts[-3]
    dtnm2 = ts[-3] - ts[-4]

    yn = ys[-2]
    ynm1 = ys[-3]
    ynm2 = ys[-4]

    return ebdf3_step(dtn, yn, dyn, dtnm1, ynm1, dtnm2, ynm2)


def ebdf3_step(dtn, yn, dyn, dtnm1, ynm1, dtnm2, ynm2):
    """Calculate one step of "explicitBDF3", i.e. the third order analogue
    of explicit midpoint rule.

    Code is generated using sym-bdf3.py.
    """
    return -(dyn*(-dtn**3*dtnm1**2*dtnm2 - dtn**3*dtnm1*dtnm2**2 - 2*dtn**2*dtnm1**3*dtnm2 - 3*dtn**2*dtnm1**2*dtnm2**2 - dtn**2*dtnm1*dtnm2**3 - dtn*dtnm1**4*dtnm2 - 2*dtn*dtnm1**3*dtnm2**2 - dtn*dtnm1**2*dtnm2**3) + yn*(2*dtn**3*dtnm1*dtnm2 + dtn**3*dtnm2**2 + 3*dtn**2*dtnm1**2*dtnm2 + 3*dtn**2*dtnm1*dtnm2**2 + dtn**2*dtnm2**3 - dtnm1**4*dtnm2 - 2*dtnm1**3*dtnm2**2 - dtnm1**2*dtnm2**3) + ynm1*(-dtn**3*dtnm1**2 - 2*dtn**3*dtnm1*dtnm2 - dtn**3*dtnm2**2 - dtn**2*dtnm1**3 - 3*dtn**2*dtnm1**2*dtnm2 - 3*dtn**2*dtnm1*dtnm2**2 - dtn**2*dtnm2**3) + ynm2*(dtn**3*dtnm1**2 + dtn**2*dtnm1**3))/(dtnm1**4*dtnm2 + 2*dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3)


def ebdf3_dynm1_step(ts, ys, dynm1):
    dtn = ts[-1] - ts[-2]
    dtnm1 = ts[-2] - ts[-3]
    dtnm2 = ts[-3] - ts[-4]

    yn = ys[-2]
    ynm1 = ys[-3]
    ynm2 = ys[-4]

    return dynm1*(-dtn**3*dtnm1**2*dtnm2/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - dtn**3*dtnm1*dtnm2**2/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - 2*dtn**2*dtnm1**3*dtnm2/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - 3*dtn**2*dtnm1**2*dtnm2**2/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - dtn**2*dtnm1*dtnm2**3/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - dtn*dtnm1**4*dtnm2/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - 2*dtn*dtnm1**3*dtnm2**2/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - dtn*dtnm1**2*dtnm2**3/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3)) + yn*(dtn**3*dtnm2**2/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + 3*dtn**2*dtnm1*dtnm2**2/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + dtn**2*dtnm2**3/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + 3*dtn*dtnm1**2*dtnm2**2/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + 2*dtn*dtnm1*dtnm2**3/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + dtnm1**3*dtnm2**2/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + dtnm1**2*dtnm2**3/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3)) + ynm1*(dtn**3*dtnm1**2/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - dtn**3*dtnm2**2/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + 2*dtn**2*dtnm1**3/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - 3*dtn**2*dtnm1*dtnm2**2/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - dtn**2*dtnm2**3/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) + dtn*dtnm1**4/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - 3*dtn*dtnm1**2*dtnm2**2/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - 2*dtn*dtnm1*dtnm2**3/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3)) + ynm2*(-dtn**3*dtnm1**2/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - 2*dtn**2*dtnm1**3/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3) - dtn*dtnm1**4/(dtnm1**3*dtnm2**2 + dtnm1**2*dtnm2**3))


def imr_residual(base_residual, ts, ys):
    dt_n = ts[-1] - ts[-2]
    y_n = ys[-2]
    y_np1 = ys[-1]

    y_nph = sp.array((y_np1 + y_n) * 0.5)
    t_nph = (ts[-1] + ts[-2]) * 0.5
    dydt_nph = imr_dydt(ts, ys)

    return base_residual(t_nph, y_nph, dydt_nph)


def imr_dydt(ts, ys):
    """Get dy/dt at the midpoint as used by imr.
    """
    dt_n = ts[-1] - ts[-2]
    y_n = ys[-2]
    y_np1 = ys[-1]

    dydt = sp.array((y_np1 - y_n)/dt_n)
    return dydt


def interpolate_dyn(ts, ys):
    #??ds probably not accurate enough

    order = 3
    ts = ts[-1*order-1:]
    ys = ys[-1*order-1:]

    ynp1_list = ys[1:]
    yn_list = ys[:-1]
    dts = utils.ts2dts(ts)

    # double check steps match
    imr_ts = map(lambda tn, tnp1: (tn + tnp1)/2, ts[1:], ts[:-1])
    imr_dys = map(lambda dt, ynp1, yn: (ynp1 -yn)/dt,
                       dts, ynp1_list, yn_list)

    dyn = (sp.interpolate.barycentric_interpolate
           (imr_ts, imr_dys, ts[-2]))

    return dyn


def bdf1_residual(base_residual, ts, ys):
    dt_n = ts[-1] - ts[-2]
    y_n = ys[-2]
    y_np1 = ys[-1]

    dydt = (y_np1 - y_n) / dt_n
    return base_residual(ts[-1], y_np1, dydt)


def bdf_residual(base_residual, ts, ys, dydt_func):
    """ Calculate residual at latest time and y-value with a bdf
    approximation for the y derivative.
    """
    return base_residual(ts[-1], ys[-1], dydt_func(ts, ys))


def bdf2_dydt(ts, ys):
    """Get dy/dt at time ts[-1] (allowing for varying dt).
    Gresho & Sani, pg. 715"""
    dt_n = ts[-1] - ts[-2]
    dt_nm1 = ts[-2] - ts[-3]

    y_np1 = ys[-1]
    y_n = ys[-2]
    y_nm1 = ys[-3]

    # Copied from oomph-lib (algebraic rearrangement of G&S forumla).
    dydt = (((1.0/dt_n) + (1.0/(dt_n + dt_nm1))) * y_np1
            - ((dt_n + dt_nm1)/(dt_n * dt_nm1)) * y_n
            + (dt_n / ((dt_n + dt_nm1) * dt_nm1)) * y_nm1)

    return dydt


def bdf3_dydt(ts, ys):
    """Get dydt at time ts[-1] to O(dt^3).

    Code is generated using sym-bdf3.py.
    """

    dtn = ts[-1] - ts[-2]
    dtnm1 = ts[-2] - ts[-3]
    dtnm2 = ts[-3] - ts[-4]

    ynp1 = ys[-1]
    yn = ys[-2]
    ynm1 = ys[-3]
    ynm2 = ys[-4]

    return dtn*(dtn + dtnm1)*(-(-(ynm1 - ynm2)/dtnm2 + (yn - ynm1)/dtnm1)/(dtnm1 + dtnm2) + (-(yn - ynm1)/dtnm1 + (ynp1 - yn)/dtn)/(dtn + dtnm1))/(dtn + dtnm1 + dtnm2) + dtn*(-(yn - ynm1)/dtnm1 + (ynp1 - yn)/dtn)/(dtn + dtnm1) + (ynp1 - yn)/dtn


def bdf4_dydt(ts, ys):
    """Get dydt at time ts[-1] using bdf4 approximation.

    Code is generated using sym-bdf3.py.
    """

    dtn = ts[-1] - ts[-2]
    dtnm1 = ts[-2] - ts[-3]
    dtnm2 = ts[-3] - ts[-4]
    dtnm3 = ts[-4] - ts[-5]

    ynp1 = ys[-1]
    yn = ys[-2]
    ynm1 = ys[-3]
    ynm2 = ys[-4]
    ynm3 = ys[-5]


    return dtn*(dtn + dtnm1)*(-(-(ynm1 - ynm2)/dtnm2 + (yn - ynm1)/dtnm1)/(dtnm1 + dtnm2) + (-(yn - ynm1)/dtnm1 + (ynp1 - yn)/dtn)/(dtn + dtnm1))/(dtn + dtnm1 + dtnm2) + dtn*(dtn + dtnm1)*((-(-(ynm1 - ynm2)/dtnm2 + (yn - ynm1)/dtnm1)/(dtnm1 + dtnm2) + (-(yn - ynm1)/dtnm1 + (ynp1 - yn)/dtn)/(dtn + dtnm1))/(dtn + dtnm1 + dtnm2) - (-(-(ynm2 - ynm3)/dtnm3 + (ynm1 - ynm2)/dtnm2)/(dtnm2 + dtnm3) + (-(ynm1 - ynm2)/dtnm2 + (yn - ynm1)/dtnm1)/(dtnm1 + dtnm2))/(dtnm1 + dtnm2 + dtnm3))*(dtn + dtnm1 + dtnm2)/(dtn + dtnm1 + dtnm2 + dtnm3) + dtn*(-(yn - ynm1)/dtnm1 + (ynp1 - yn)/dtn)/(dtn + dtnm1) + (ynp1 - yn)/dtn


bdf2_residual = par(bdf_residual, dydt_func=bdf2_dydt)
bdf3_residual = par(bdf_residual, dydt_func=bdf3_dydt)
bdf4_residual = par(bdf_residual, dydt_func=bdf4_dydt)


# Assumption: we never actually undo a timestep (otherwise dys will become
# out of sync with ys).
class TrapezoidRuleResidual(object):
    """A class to calculate trapezoid rule residuals.

    We need a class because we need to store the past data. Other residual
    calculations do not.
    """

    def __init__(self):
        self.dys = []

    def calculate_new_dy_if_needed(self, ts, ys):
        """If dy_n/dt has not been calculated on this step then calculate
        it from previous values of y and dydt by inverting the trapezoid
        rule.
        """
        if len(self.dys) < len(ys) - 1:
            dt_nm1 = ts[-2] - ts[-3]
            dy_n = (2.0 / dt_nm1) * (ys[-2] - ys[-3]) - self.dys[-1]
            self.dys.append(dy_n)

    def _get_initial_dy(self, base_residual, ts, ys):
        """Calculate a step with imr to get dydt at y_n.
        """

        # We want to ignore the most recent two steps (the one being solved
        # for "now" outside this function and the one we are computing the
        # derivative at). We also want to ensure nothing is modified in the
        # solutions list:
        temp_ts = copy.deepcopy(ts[:-2])
        temp_ys = copy.deepcopy(ys[:-2])

        # Timestep should be double the timestep used for the previous
        # step, so that the imr is at y_n.
        dt_n = 2*(ts[-2] - ts[-3])

        # Calculate time step
        temp_ys, temp_ts = _odeint(base_residual, temp_ys, temp_ts, dt_n,
                                   temp_ts[-1] + dt_n, imr_residual)

        # Check that we got the right times: the midpoint should be at
        # the step before the most recent time.
        utils.assert_almost_equal((temp_ts[-1] + temp_ts[-2])/2, ts[-2])

        # Now invert imr to get the derivative
        dy_nph = (temp_ys[-1] - temp_ys[-2])/dt_n

        # Fill in dummys (as many as we have y values) followed by the
        # derivative we just calculated.
        self.dys = [float('nan')] * (len(ys)-1)
        self.dys[-1] = dy_nph

    def __call__(self, base_residual, ts, ys):
        if len(self.dys) == 0:
            self._get_initial_dy(base_residual, ts, ys)

        dt_n = ts[-1] - ts[-2]
        t_np1 = ts[-1]
        y_n = ys[-2]
        y_np1 = ys[-1]

        self.calculate_new_dy_if_needed(ts, ys)
        dy_n = self.dys[-1]

        dy_np1 = (2.0/dt_n) * (y_np1 - y_n) - dy_n
        return base_residual(t_np1, y_np1, dy_np1)

# Adaptive timestepping functions
# ============================================================


def default_dt_scaling(target_error, error_estimate, timestepper_order):
    """Standard way of rescaling the time step to attain the target error.
    Taken from Gresho and Sani (various places).
    """
    try:
        power = (1.0/(1.0 + timestepper_order))
        scaling_factor = (target_error/error_estimate)**power

    except ZeroDivisionError:
        scaling_factor = MAX_ALLOWED_DT_SCALING_FACTOR

    return scaling_factor


def failed_timestep_scaling(*_):
    """Return scaling factor for a failed time step, ignores all input
    arguments.
    """
    return TIMESTEP_FAILURE_DT_SCALING_FACTOR


def create_random_time_adaptor(base_dt,
                               min_scaling=TIMESTEP_FAILURE_DT_SCALING_FACTOR,
                               max_scaling=MAX_ALLOWED_DT_SCALING_FACTOR):
    """Create time adaptor which randomly changes the time step to some
    multiple of base_dt. Scaling factor is within the allowed range. For
    testing purposes.
    """
    def random_time_adaptor(*_):
        return base_dt * random.uniform(min_scaling, max_scaling)
    return random_time_adaptor


def scale_timestep(dt, target_error, error_norm, order,
                   scaling_function=default_dt_scaling):
    """Scale dt by a scaling factor. Mostly this function is needed to
    check that the scaling factor and new time step are within the
    allowable bounds.
    """

    # Calculate the scaling factor and the candidate for next step size.
    scaling_factor = scaling_function(target_error, error_norm, order)
    new_dt = scaling_factor * dt

    # If the error is too bad (i.e. scaling factor too small) reject the
    # step, unless we are already dealing with a rejected step;
    if scaling_factor < MIN_ALLOWED_DT_SCALING_FACTOR \
            and not scaling_function is failed_timestep_scaling:
        raise FailedTimestepError(new_dt)

    # or if the scaling factor is really large just use the max scaling.
    elif scaling_factor > MAX_ALLOWED_DT_SCALING_FACTOR:
        scaling_factor = MAX_ALLOWED_DT_SCALING_FACTOR

    # If the timestep would get too big then return the max time step;
    if new_dt > MAX_ALLOWED_TIMESTEP:
        return MAX_ALLOWED_TIMESTEP

    # or if the timestep would become too small then fail;
    elif new_dt < MIN_ALLOWED_TIMESTEP:
        error = "Tried to reduce dt to " + str(new_dt) +\
            " which is less than the minimum of " + str(MIN_ALLOWED_TIMESTEP)
        raise ConvergenceFailure(error)

    # otherwise scale the timestep normally.
    else:
        return new_dt


def general_time_adaptor(ts, ys, target_error, method_order, lte_calculator,
                         **kwargs):
    """General base function for time adaptivity function.

    Partially evaluate with a method order and an lte_calculator to create
    a complete time adaptor function.

    Other args are passed down to the lte calculator.
    """

    # Get the local truncation error estimator
    lte_est = lte_calculator(ts, ys, **kwargs)

    # Get the 2-norm
    error_norm = sp.linalg.norm(sp.array(lte_est, ndmin=1), 2)

    # Return the scaled timestep (with lots of checks).
    return scale_timestep(ts[-1] - ts[-2], target_error, error_norm,
                          method_order)


def bdf2_mp_prinja_lte_estimate(ts, ys):
    """Estimate LTE using combination of bdf2 and explicit midpoint. From
    Prinja's thesis.
    """

    # Get local values (makes maths more readable)
    dt_n = ts[-1] - ts[-2]
    dt_nm1 = ts[-2] - ts[-3]
    dtr = dt_n / dt_nm1
    dtrinv = 1.0 / dtr

    y_np1 = ys[-1]
    y_n = ys[-2]
    y_nm1 = ys[-3]

    # Invert bdf2 to get derivative
    dy_n = bdf2_dydt(ts[:-1], ys[:-1])

    # Calculate predictor value (variable dt explicit mid point rule)
    y_np1_EMR = ebdf2_step(dt_n, y_n, dy_n, dt_nm1, y_nm1)

    error_weight = (dt_nm1 + dt_n) / (3*dt_n + 2*dt_nm1)

    # Calculate truncation error -- oomph-lib
    error = (y_np1 - y_np1_EMR) * error_weight

    return error


def bdf2_mp_gs_lte_estimate(ts, ys):
    """Estimate LTE using combination of bdf2 and explicit midpoint. From
    oomph-lib and G&S.
    """

    # Get local values (makes maths more readable)
    dt_n = ts[-1] - ts[-2]
    dt_nm1 = ts[-2] - ts[-3]
    dtr = dt_n / dt_nm1
    dtrinv = 1.0 / dtr

    y_np1 = ys[-1]
    y_n = ys[-2]
    y_nm1 = ys[-3]

    # Invert bdf2 to get predictor data (using the exact same function as
    # was used in the residual calculation).
    dy_n = bdf2_dydt(ts[:-1], ys[:-1])

    # Calculate predictor value (variable dt explicit mid point rule)
    y_np1_EMR = ebdf2_step(dt_n, y_n, dy_n, dt_nm1, y_nm1)

    error_weight = ((1.0 + dtrinv)**2) / \
        (1.0 + 3.0*dtrinv + 4.0 * dtrinv**2
         + 2.0 * dtrinv**3)

    # Calculate truncation error -- oomph-lib
    error = (y_np1 - y_np1_EMR) * error_weight

    return error


# ??Ds use prinja's
bdf2_mp_lte_estimate = bdf2_mp_prinja_lte_estimate


def tr_ab_lte_estimate(ts, ys, dydt_func):
    dt_n = ts[-1] - ts[-2]
    dt_nm1 = ts[-2] - ts[-3]
    dtrinv = dt_nm1 / dt_n

    y_np1 = ys[-1]
    y_n = ys[-2]
    y_nm1 = ys[-3]

    dy_n = dydt_func(ts[-2], y_n)
    dy_nm1 = dydt_func(ts[-3], y_nm1)

    # Predict with AB2
    y_np1_AB2 = ab2_step(dt_n, y_n, dy_n, dt_nm1, dy_nm1)

    # Estimate LTE
    lte_est = (y_np1 - y_np1_AB2) / (3*(1 + dtrinv))
    return lte_est


def imr_jacobian_ab_time_adaptor(ts, ys, target_error, dfdy_function=None):
    """
    See notes: 7/2/2013 for the algebra on calculating the AB2
    solution. See "mathematica_adaptive_midpoint.m for algebra to get a
    local truncation error out of all this.
    """

    dt_n = ts[-1] - ts[-2]
    dt_nm1 = ts[-2] - ts[-3]
    y_np1_MP = ys[-1]
    y_nm1_MP = ys[-3]

    t_nph = (ts[-1] + ts[-2])/2

    dy_nmh, y_nph, dy_nph, ddy_nph = imr_approximation_fake_interpolation(
        ts, ys)

    # Get explicit adams-bashforth 2 solution (variable timestep -- uses
    # steps at n + 1/2 and n - 1/2).
    # ============================================================
    # Calculate the corresponding fictional timestep sizes
    AB_dt_n = 0.5*dt_n
    AB_dt_nm1 = 0.5*dt_n + 0.5*dt_nm1

    # Calculate using AB2 variable timestep formula
    y_np1_AB2 = ab2_step(AB_dt_n, y_nph, dy_nph,
                         AB_dt_nm1, dy_nmh)

    # Choose an estimate for the derivatives of y at time t_{n+0.5}
    # ============================================================

    # df/dy term (note: this should be a square matrix of size len(y))
    if dfdy_function is not None:
        dfdy_nph = dfdy_function(t_nph, y_nph, dy_nph)
    else:
        # Can't calculate so just ignore it
        dfdy_nph = sp.zeros((len(ys[-1]), len(ys[-1])))

    # Get the truncation error and scaling factor
    # ============================================================
    # As derived in notes from 21/2/13
    a = 4 / (1 + 3*(dt_nm1/dt_n))
    dddy_term = (y_np1_AB2 - y_np1_MP) * a
    dfdy_term = (dt_n**3 * (1+a)/8) * dfdy_nph.dot(ddy_nph)

    error = dddy_term - dfdy_term

    # Get a norm and scaling factor
    error_norm = sp.linalg.norm(sp.array(error, ndmin=1), 2)

    return scale_timestep(dt_n, target_error, error_norm, 3)



def imr_ab_lte_estimate(ts, ys, interpolator=my_interpolate,
                             ab_start_point='t_n',
                             explicit_derivative=None,
                             fudge_factor=1.0):
    """ See notes: 19-20/3/2013 for algebra and explanations.
    """

    # Notation:
    # _nph denotes exact values at time t_nph
    # _mid denotes approximations using (y_np1 + yn)/2
    # Set up some variables
    # ============================================================
    dt_n = ts[-1] - ts[-2]
    dt_nm1 = ts[-2] - ts[-3]
    dtr = dt_n/dt_nm1

    y_np1_MP = ys[-1]
    y_n_MP = ys[-2]
    y_nm1_MP = ys[-3]

    t_n = ts[-2]
    t_nm1 = ts[-3]
    t_nph = (ts[-1] + ts[-2])/2
    t_nmh = (ts[-2] + ts[-3])/2

    y_nmid = (y_np1_MP + y_n_MP)/2
    dy_nmid = (y_np1_MP - y_n_MP)/dt_n
    dy_nm1_mid = (ys[-3] - ys[-2])/dt_nm1

    # Interpolate exact value and derivatives at t_nph
    # ============================================================

    if explicit_derivative is None:
        dy_nmh, y_nph, dy_nph, _, dy_n = interpolator(ts, ys)

    # we have a useable explicit derivative function, use it to calculate
    # some derivatives. This WONT WORK with some of the weirder
    # settings.... Should fix it.. ??ds
    else:
        assert(ab_start_point=='t_n')
        # assert(interpolator==par(my_interpolate, n_interp=2))

        # dy_nph = explicit_derivative(t_nph)
        # dy_nmh = explicit_derivative(t_nmh)

        dy_nm1 = explicit_derivative(t_nm1, y_nm1_MP)
        dy_n = explicit_derivative(t_n, y_n_MP)
        dy_np1 = explicit_derivative(ts[-1], y_np1_MP)

        # Just interpolate value + derivative at t_nph
        interps = krogh_interpolate([ts[-4], t_nm1, t_nm1, t_n, t_n,], # input ts
                                    [ys[-4], y_nm1_MP, dy_nm1, y_n_MP, dy_n,], # inputs ys
                                    [t_nph], der=[0, 1]) # outputs wanted

        y_nph = interps[0][0]
        dy_nph = interps[1][0]


    # Calculate this part of the truncation error (see notes)
    # ymid_estimation_error = dt_n*dy_nph + y_n_MP - y_np1_MP
    ymid_estimation_error = dt_n*(dy_nph - dy_nmid)


    # Use an AB2 predictor to eliminate the dddy_nph term
    # ============================================================
    AB_dt_n = 0.5*dt_n

    if ab_start_point == 't_n':
        AB_dt_nm1 = 0.5*dt_n
        AB_dy_start = dy_n

    elif ab_start_point == 't_nmh':
        AB_dt_nm1 = 0.5*dt_n + 0.5*dt_nm1
        AB_dy_start = dy_nmh

    else:
        err = "Don't recognise ab_start_point value "+str(ab_start_point)
        raise ValueError(err)


    y_np1_AB2 = ab2_step(AB_dt_n, y_nph, dy_nph,
                         AB_dt_nm1, AB_dy_start)


    # Get the truncation error, scaling factor and new time step size
    # ============================================================

    # Calculate with a choice of start points (see notes)
    if ab_start_point == 't_nmh':
        p = -4 / (1 + 3/dtr)
        imr_lte = ymid_estimation_error + \
          p * (y_np1_AB2 - y_np1_MP - ymid_estimation_error)

    elif ab_start_point == 't_n':
        imr_lte = 4*(y_np1_MP - y_np1_AB2) + 5*ymid_estimation_error

    else:
        err = "Don't recognise ab_start_point value "+str(ab_start_point)
        raise ValueError(err)

    return imr_lte * fudge_factor


def ebdf3_lte_estimate(ts, ys, dydt_func=None):
    """Estimate lte for any second order integration method by comparing
    with the third order explicit bdf method.
    """

    # Use BDF approximation to dyn if no function given
    if dydt_func is None:
        # dyn = bdf3_dydt(ts[:-1], ys[:-1])
        dyn = interpolate_dyn(ts, ys)
    else:
        dyn = dydt_func(ts[-2], ys[-2])

    y_np1_EBDF3 = ebdf3_step_wrapper(ts, ys, dyn)

    return ys[-1] - y_np1_EBDF3


def ebdf3_dynm1_lte_estimate(ts, ys, dydt_func=None):
    """Estimate lte for with the third order explicit bdf method w/
    derivative point at y_{n-1}. Useful for imr?
    """

    # Use BDF approximation to dyn if no function given
    if dydt_func is None:
        # assuming imr is the real timestepper
        dynm1 = imr_dydt(ts[:-1], ys[:-1]) - imr_dydt(ts[:-2], ys[:-2])
    else:
        dynm1 = dydt_func(ts[-3], ys[-3])

        # # debugging ??ds
        # dtn = ts[-1] - ts[-2]
        # err_margin = 25* dtn**2
        # est_dynm1 = (imr_dydt(ts[:-1], ys[:-1]) - imr_dydt(ts[:-2], ys[:-2]))/2
        # if not utils.almost_equal(dynm1, est_dynm1, err_margin):
        #     print
        #     print dynm1
        #     print est_dynm1
        #     print "diff =", abs(dynm1 - est_dynm1), ">", err_margin

    y_np1_EBDF3 = ebdf3_dynm1_step(ts, ys, dynm1)

    return ys[-1] - y_np1_EBDF3


def imr_fe_ab_time_adaptor(ts, ys, target_error,
                                interpolator=my_interpolate):
    """ See notes: 19-20/3/2013 for algebra and explanations.
    """

    # Notation:
    # _nph denotes exact values at time t_nph
    # _mid denotes approximations using (y_np1 + yn)/2

    # Set up some variables
    # ============================================================

    dt_n = ts[-1] - ts[-2]
    dt_nm1 = ts[-2] - ts[-3]

    y_np1_MP = ys[-1]
    y_n_MP = ys[-2]
    y_nm1_MP = ys[-3]

    t_nph = (ts[-1] + ts[-2])/2
    t_nmh = (ts[-2] + ts[-3])/2

    y_nmid = (y_np1_MP + y_n_MP)/2
    dy_nmid = (y_np1_MP - y_n_MP)/dt_n
    dy_nm1_mid = (ys[-3] - ys[-2])/dt_nm1

    # Interpolate value at t_nph
    # ============================================================

    dy_nmh, y_nph, dy_nph, ddy_nph, _ = interpolator(ts, ys)

    # Use a forward Euler predictor to eliminate the Jacobian term
    # ============================================================
    # Forward Euler with exact initial y and approximate imr
    # derivative.
    y_np1_FE = y_nph + (dt_n/2) * dy_nmid

    # Use an AB2 predictor to eliminate the dddy_nph term
    # ============================================================
    AB_dt_n = 0.5*dt_n
    AB_dt_nm1 = 0.5*dt_n + 0.5*dt_nm1

    y_np1_AB2 = ab2_step(AB_dt_n, y_nph, dy_nph,
                         AB_dt_nm1, dy_nmh)

    # Get the truncation error, scaling factor and new time step size
    # ============================================================

    # Some useful values
    omega_n = dt_n / (3*dt_nm1)
    y_np1_FEc = (y_np1_FE + ((dt_n**2)/4) * ddy_nph)

    # Calculate the lte (see notes for why this equation)
    imr_lte = (omega_n * (y_np1_MP - y_np1_AB2)
                    + ((omega_n - 1)/2) * (y_np1_FEc - y_np1_MP))

    # Get a norm
    error_norm = sp.linalg.norm(sp.array(imr_lte, ndmin=1), 2)

    # Return the scaled timestep (with lots of checks).
    return scale_timestep(dt_n, target_error, error_norm, 3)


# Scheme from friday 13th-ish
# ============================================================

def f13_f_hat_imr(ts, ys_est):
    """Extract derivative approximation used by imr from
    y-values output by imr.
    """
    dtn = ts[-1] - ts[-2]
    return (ys_est[-1] - ys_est[-2])/dtn


def f13_lte_est(ts, ys, F_func):

    f_hat_nph = f13_f_hat_imr(ts, ys)
    f_hat_nmh = f13_f_hat_imr(ts[:-1], ys[:-1])

    dtn = ts[-1] - ts[-2]
    dtnm1 = ts[-2] - ts[-3]

    ynp1_MP = ys[-1]
    yn = ys[-2]
    ynm1 = ys[-3]

    # ??ds rough approximation to ynph used here... should be ok maybe...
    Fnph = F_func(ts[-2] + dtn/2, (ynp1_MP + yn)/2)

    assert Fnph == 0

    # (rough) Approximations!
    ddyn_approx = (f_hat_nph - (dtn/dtnm1) * f_hat_nmh)
    print ddyn_approx

    Fdotddyn_approx = sp.dot(Fnph, ddyn_approx)

    dyn_approx = ((1/2)*(f_hat_nph + (dtn/dtnm1) * f_hat_nmh)
                  - (dtn**2/8)*Fdotddyn_approx)

    ynp1_P = ebdf2_step(dtn, yn, dyn_approx, dtnm1, ynm1)

    print dtn, ynp1_P, ynp1_MP

    divfac = (1/4) * (dtn/6*dtnm1)
    dddynph_approx = ((1/(dtn**3 * divfac)) * (ynp1_MP - ynp1_P)
                      - (1/(8*divfac))*Fdotddyn_approx)

    LTEn = (1/10000)*(dtn**3/24) * (dddynph_approx - 3*Fdotddyn_approx)

    return LTEn



# Testing
# ============================================================
import matplotlib.pyplot as plt
from simpleode.core.example_residuals import *


def check_problem(method, residual, exact, tol=1e-4, tmax=2.0):
    """Helper function to run odeint with a specified method for a problem
    and check that the solution matches.
    """

    ys, ts = odeint(residual, [exact(0.0)], tmax, dt=1e-6,
                    method=method, target_error=tol)

    # Total error should be bounded by roughly n_steps * LTE
    overall_tol = len(ys) * tol * 10
    utils.assert_list_almost_equal(ys, map(exact, ts), overall_tol)

    return ts, ys


def test_bad_timestep_handling():
    """ Check that rejecting timesteps works.
    """
    tmax = 0.4
    tol = 1e-4

    def list_cummulative_sums(values, start):
        temp = [start]
        for v in values:
            temp.append(temp[-1] + v)
        return temp

    dts = [1e-6, 1e-6, 1e-6, (1. - 0.01)*tmax]
    initial_ts = list_cummulative_sums(dts[:-1], 0.)
    initial_ys = [sp.array(exp3_exact(t), ndmin=1) for t in initial_ts]

    adaptor = par(general_time_adaptor, lte_calculator=bdf2_mp_lte_estimate,
                  method_order=2)

    ys, ts = _odeint(exp3_residual, initial_ys, initial_ts, dts[-1], tmax,
                     bdf2_residual, tol, adaptor)

    # plt.plot(ts,ys)
    # plt.plot(ts[1:], utils.ts2dts(ts))
    # plt.plot(ts, map(exp3_exact, ts))
    # plt.show()

    overall_tol = len(ys) * tol * 2  # 2 is a fudge factor...
    utils.assert_list_almost_equal(ys, map(exp3_exact, ts), overall_tol)



def test_ab2():

    def check_explicit_stepper(stepper, exact_symb):

        exact, residual, dys, J = utils.symb2functions(exact_symb)

        base_dt = 1e-3
        ys, ts = odeint_explicit(dys[1], exact(0.0), base_dt, 1.0, stepper,
                                 time_adaptor=create_random_time_adaptor(base_dt))

        exact_ys = map(exact, ts)
        utils.assert_list_almost_equal(ys, exact_ys, 1e-4)


    t = sympy.symbols('t')
    functions = [2*t**2,
                 sympy.exp(t),
                 3*sympy.exp(-t)
                 ]

    steppers = ['ab2',
                # 'ebdf2',
                # 'ebdf3'
                ]

    for func in functions:
        for stepper in steppers:
            yield check_explicit_stepper, stepper, func


def test_dydt_calcs():

    def check_dydt_calcs(dydt_calculator, order, dt, dydt_exact, y_exact):
        """Check that derivative approximations are roughly as accurate as
        expected for a few functions.
        """

        ts = sp.arange(0, 0.5, dt)
        exact_ys = map(y_exact, ts)
        exact_dys = map(dydt_exact, ts, exact_ys)

        est_dys = map(dydt_calculator, utils.partial_lists(ts, 5),
                      utils.partial_lists(exact_ys, 5))

        utils.assert_list_almost_equal(est_dys, exact_dys[4:], 10*(dt**order))

    fs = [(poly_dydt, poly_exact),
          (exp_dydt, exp_exact),
          (exp_of_poly_dydt, exp_of_poly_exact),
          ]

    dydt_calculators = [(bdf2_dydt, 2),
                        (bdf3_dydt, 3),
                        (imr_dydt, 1),
                        ]
    dts = [0.1, 0.01, 0.001]

    for dt in dts:
        for dydt_exact, y_exact in fs:
            for dydt_calculator, order in dydt_calculators:
                yield check_dydt_calcs, dydt_calculator, order, dt, dydt_exact, y_exact


def test_exp_timesteppers():

    # Auxilary checking function
    def check_exp_timestepper(method, tol):
        def residual(t, y, dydt):
            return y - dydt
        tmax = 1.0
        dt = 0.001
        ys, ts = odeint(exp_residual, [exp(0.0)], tmax, dt=dt,
                        method=method)

        # plt.plot(ts,ys)
        # plt.plot(ts, map(exp,ts), '--r')
        # plt.show()
        utils.assert_almost_equal(ys[-1], exp(tmax), tol)

    # List of test parameters
    methods = [('bdf2', 1e-5),
               ('bdf1', 1e-2),  # First order method...
               ('imr', 1e-5),
               ('trapezoid', 1e-5),
               ]

    # Generate tests
    for meth, tol in methods:
        yield check_exp_timestepper, meth, tol


def test_vector_timesteppers():

    # Auxilary checking function
    def check_vector_timestepper(method, tol):
        def residual(t, y, dydt):
            return sp.array([-1.0 * sin(t), y[1]]) - dydt
        tmax = 1.0
        ys, ts = odeint(residual, [cos(0.0), exp(0.0)], tmax, dt=0.001,
                        method=method)

        utils.assert_almost_equal(ys[-1][0], cos(tmax), tol[0])
        utils.assert_almost_equal(ys[-1][1], exp(tmax), tol[1])

    # List of test parameters
    methods = [('bdf2', [1e-4, 1e-4]),
               ('bdf1', [1e-2, 1e-2]),  # First order methods suck...
               ('imr', [1e-4, 1e-4]),
               ('trapezoid', [1e-4, 1e-4]),
               ]

    # Generate tests
    for meth, tol in methods:
        yield check_vector_timestepper, meth, tol


def test_adaptive_dt():

    methods = [('bdf2 mp', 1e-4),
               # ('imr fe ab', 1e-4),
               # ('imr ab', 1e-4),
               ('imr ebdf3', 1e-5), # test with and without explicit f
               ({'label':'imr ebdf3'}, 1e-5),
               ({'label':'tr ab'}, 1e-4),
               ]

    functions = [(exp_residual, exp, exp_dydt),
                 # (exp_of_minus_t_residual, exp_of_minus_t_exact, exp_of_minus_t_dydt),
                 (poly_residual, poly_exact, poly_dydt),
                 (exp_of_poly_residual, exp_of_poly_exact, exp_of_poly_dydt)
                 ]

    for meth, tol in methods:
        for residual, exact, dydt in functions:

            # If we can, then put the dydt function name into the dict,
            # most methods shouldn't need this.
            try:
                meth['dydt_func'] = dydt
            except TypeError:
                pass

            yield check_problem, meth, residual, exact, tol


def test_sharp_dt_change():

    # Parameters, don't fiddle with these too much or it might miss the
    # step altogether...
    alpha = 20
    step_time = 0.4
    tmax = 2.5
    tol = 1e-4

    # Set up functions
    residual = par(tanh_residual, alpha=alpha, step_time=step_time)
    exact = par(tanh_exact, alpha=alpha, step_time=step_time)

    # Run it
    return check_problem('imr ab', residual, exact, tol=tol)


# def test_with_stiff_problem():
#     """Check that imr fe ab works well for stiff problem (i.e. has a
#     non-insane number of time steps).
#     """
#     # Slow test!

#     mu = 1000
#     residual = par(van_der_pol_residual, mu=mu)

#     ys, ts = odeint(residual, [2.0, 0], 1000.0, dt=1e-6,
#                     method='imr ab', target_error=1e-3)

#     print len(ts)
#     plt.plot(ts, [y[0] for y in ys])
#     plt.plot(ts[1:], utils.ts2dts(ts))
#     plt.show()

#     n_steps = len(ts)
#     assert n_steps < 5000

def test_newton():
    def check_newton(residual, exact):
        solution = newton(residual, sp.array([1.0]*len(exact)))
        utils.assert_list_almost_equal(solution, exact)

    tests = [(lambda x: sp.array([x**2 - 2]), sp.array([sp.sqrt(2)])),
             (lambda x: sp.array([x[0]**2 - 2, x[1] - x[0] - 1]),
              sp.array([sp.sqrt(2), sp.sqrt(2) + 1])),
             ]

    for res, exact in tests:
        check_newton(res, exact)
