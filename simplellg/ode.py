from __future__ import division

from math import sin, cos, tan, log, atan2, acos, pi, sqrt, exp
import scipy as sp
import scipy.integrate
from scipy.optimize import newton_krylov
from scipy.linalg import norm
import scipy.optimize.nonlin
import functools as ft
from functools import partial as par
import itertools as it
import copy
from scipy.interpolate import krogh_interpolate

import simplellg.utils as utils

# PARAMETERS
MAX_ALLOWED_DT_SCALING_FACTOR = 4.0
MIN_ALLOWED_DT_SCALING_FACTOR = 0.9
TIMESTEP_FAILURE_DT_SCALING_FACTOR = 0.5

MIN_ALLOWED_TIMESTEP = 1e-8
MAX_ALLOWED_TIMESTEP = 1e8


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


class FailedTimestepError(Exception):
    def __init__(self, new_dt):
        self.new_dt = new_dt

    def __str__(self):
        return "Exception: timestep failed, next timestep should be "\
            + repr(self.new_dt)


def _timestep_scheme_dispatcher(label):
    """Pick the functions needed for this method. Returns functions for a
    time residual, a timestep adaptor and any intialisation actions needed.
    """
    label = label.lower()

    if label == 'bdf2':
        return bdf2_residual, None, par(higher_order_start, 2)

    elif label == 'bdf2 ab':
        return bdf2_residual, bdf2_ab_time_adaptor,\
            par(higher_order_start, 2)

    elif label == 'bdf1':
        return bdf1_residual, None, None

    elif label == 'midpoint':
        return midpoint_residual, None, None

    elif label == 'midpoint ab':
        n_start = 4
        adaptor = par(midpoint_ab_time_adaptor,
                      interpolator=krogh_interpolate)
        return midpoint_residual, adaptor, par(higher_order_start, n_start)

    elif label == 'midpoint fe ab':
        n_start = 4
        adaptor = par(midpoint_fe_ab_time_adaptor,
                      interpolator=Interpolator(n_start))
        return midpoint_residual, adaptor, par(higher_order_start, n_start)

    elif label == 'trapezoid':
        # TR is actually self starting but due to technicalities with
        # getting derivatives of y from implicit formulas we need an extra
        # starting point.
        return TrapezoidRuleResidual(), None, par(higher_order_start, 2)

    else:
        message = "Method '"+label+"' not recognised."
        raise ValueError(message)


def higher_order_start(order, func, ys, ts):
    """ Run a few steps of midpoint method with a very small timestep.
    Useful for generating extra initial data for multi-step methods.
    """
    starting_dt = 1e-6
    while len(ys) < order:
        ys, ts = _odeint(func, ys, ts, starting_dt, ts[-1] + starting_dt,
                         midpoint_residual)
    return ys, ts


def odeint(func, y0, tmax, dt,
           method='bdf2',
           target_error=None,
           actions_after_timestep=None,
           **kwargs):

    # Select the method and adaptor
    time_residual, time_adaptor, initialisation_actions = \
        _timestep_scheme_dispatcher(method)

    ts = [0.0]  # List of times (floats)
    ys = [sp.array(y0, ndmin=1)]  # List of y vectors (ndarrays)

    # Now call the actual function to do the work
    return _odeint(func, ys, ts, dt, tmax, time_residual,
                   target_error, time_adaptor, initialisation_actions,
                   actions_after_timestep, **kwargs)


def _odeint(func, ys, ts, dt, tmax, time_residual,
            target_error=None, time_adaptor=None,
            initialisation_actions=None, actions_after_timestep=None):

    if initialisation_actions is not None:
        ys, ts = initialisation_actions(func, ys, ts)

    # Main timestepping loop
    # ============================================================
    while ts[-1] < tmax:

        t_np1 = ts[-1] + dt

        # Fill in the residual for calculating dydt and the previous time
        # and y values ready for the Newton solver.
        residual = lambda y_np1: time_residual(func, ts+[t_np1], ys+[y_np1])

        # Try to solve the system, using the previous y as an initial
        # guess. If it fails reduce dt and try again.
        try:
            y_np1 = newton_krylov(residual, ys[-1], method='gmres')
        except sp.optimize.nonlin.NoConvergence:
            dt = _scale_timestep(dt, TIMESTEP_FAILURE_DT_SCALING_FACTOR, True)
            print "Failed to converge, reducing time step."
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
            except FailedTimestepError, exception:
                dt = exception.new_dt
                continue

        # Update results storage (don't do this earlier in case the time
        # step fails).
        ys.append(new_y_np1)
        ts.append(new_t_np1)

    return ys, ts


# Timestepper residual calculation functions
# ============================================================

def midpoint_residual(base_residual, ts, ys):
    dt_n = ts[-1] - ts[-2]
    y_n = ys[-2]
    y_np1 = ys[-1]

    y_nph = (y_np1 + y_n) * 0.5
    t_nph = (ts[-1] + ts[-2]) * 0.5
    dydt_nph = midpoint_dydt(ts, ys)

    return base_residual(t_nph, y_nph, dydt_nph)


def midpoint_dydt(ts, ys):
    """Get dy/dt at the midpoint,

    time = (ts[-1] + ts[-2])/ 2.
    """
    dt_n = ts[-1] - ts[-2]
    y_n = ys[-2]
    y_np1 = ys[-1]

    dydt = (y_np1 - y_n)/dt_n
    return dydt


def bdf1_residual(base_residual, ts, ys):
    dt_n = ts[-1] - ts[-2]
    y_n = ys[-2]
    y_np1 = ys[-1]

    dydt = (y_np1 - y_n) / dt_n
    return base_residual(ts[-1], y_np1, dydt)


def bdf2_residual(base_residual, ts, ys):
    return base_residual(ts[-1], ys[-1], bdf2_dydt(ts, ys))


def bdf2_dydt(ts, ys):
    """Get dy/dt at time ts[-1]."""
    dt_n = ts[-1] - ts[-2]
    dt_nm1 = ts[-2] - ts[-3]

    y_np1 = ys[-1]
    y_n = ys[-2]
    y_nm1 = ys[-3]

    a = dt_n/((2*dt_n + dt_nm1)*dt_nm1)
    prefactor = (2*dt_n + dt_nm1)/(dt_n + dt_nm1)
    alphas = [1.0/dt_n, -1.0/dt_n - a, a]

    dydt = prefactor * sp.dot(alphas, [y_np1, y_n, y_nm1])
    return dydt


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
        """Calculate a step with midpoint to get dydt at y_n.
        """

        # We want to ignore the most recent two steps (the one being solved
        # for "now" outside this function and the one we are computing the
        # derivative at). We also want to ensure nothing is modified in the
        # solutions list:
        temp_ts = copy.deepcopy(ts[:-2])
        temp_ys = copy.deepcopy(ys[:-2])

        # Timestep should be double the timestep used for the previous
        # step, so that the midpoint is at y_n.
        dt_n = 2*(ts[-2] - ts[-3])

        # Calculate time step
        temp_ys, temp_ts = _odeint(base_residual, temp_ys, temp_ts, dt_n,
                                   temp_ts[-1] + dt_n, midpoint_residual)

        # Check that we got the right times: the midpoint should be at
        # the step before the most recent time.
        utils.assertAlmostEqual((temp_ts[-1] + temp_ts[-2])/2, ts[-2])

        # Now invert midpoint to get the derivative
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


def _scale_timestep(dt, scaling_factor, failed_step_flag=False):
    """ Scale dt by a scaling factor. Mostly this function is only needed
    for error checking.
    """

    # If the error is too bad (scaling factor too small) then reject the
    # step.
    if scaling_factor < MIN_ALLOWED_DT_SCALING_FACTOR \
            and not failed_step_flag:
        raise FailedTimestepError(scaling_factor * dt)

    # If the scaling factor is really large just use the max
    if scaling_factor > MAX_ALLOWED_DT_SCALING_FACTOR:
        scaling_factor = MAX_ALLOWED_DT_SCALING_FACTOR

    # If the timestep would still get too big then return the max
    if scaling_factor * dt > MAX_ALLOWED_TIMESTEP:
        return MAX_ALLOWED_TIMESTEP

    # If the timestep would become too small then fail
    elif scaling_factor * dt < MIN_ALLOWED_TIMESTEP:
        error = "Tried to reduce dt to " + str(scaling_factor * dt) +\
            " which is less than the minimum of " + str(MIN_ALLOWED_TIMESTEP)
        raise ValueError(error)

    # Otherwise scale the timestep normally
    else:
        return scaling_factor * dt


def ab2_step(dt_n, y_n, dy_n, dt_nm1, dy_nm1):
    """ Calculate the solution at time n.

    From: my code... ??ds
    """
    dtr = dt_n / dt_nm1
    y_np1 = y_n + 0.5*dt_n*((2 + dtr)*dy_n - dtr*dy_nm1)
    return y_np1


def bdf2_lte(dt_n, ys, dys, d2ys, d3ys):
    """ From Prinja's thesis.
    """
    d3ydt_n = d3ys[-1]
    return (2.0/9) * d3ydt_n * dt_n**3


def trapezoid_lte(dt_n, ys, dys, d2ys, d3ys):
    """ Calculated myself, notes: 15/2/13.
    """
    return (1.0/12) * dt_n**3 * d3ys[-1]


def midpoint_lte(dt_n, ys, dys, d2ys, d3ys, dfdy):
    """ notes: 20/2/13
    """
    if dfdy is None:
        dfdy = 0.

    # dfdy should be able to be a function but not implelmented yet

    return ((dt_n**3)/24) * (d3ys[-1] - sp.dot(dfdy, d2ys[-1]))


def bdf2_ab_time_adaptor(ts, ys, target_error, dfdy_function=None):
    """ Calculate a new timestep size based on estimating the error from
    the previous timestep. Weighting numbers stolen from oopmh-lib.
    """

    if dfdy_function is not None:
        print "Warning: ignoring dfdy_function in bdf2 error estimator."

    # Get local values (makes maths more readable)
    dt_n = ts[-1] - ts[-2]
    dt_nm1 = ts[-2] - ts[-3]
    dtr = dt_n / dt_nm1

    y_np1 = ys[-1]
    y_n = ys[-2]
    y_nm1 = ys[-3]

    # Fudge the first step
    if len(ys) < 4:
        return dt_n

    # Invert bdf to get predictor data (using the exact same function as
    # was used in the residual calculation).
    dy_n = bdf2_dydt(ts, ys)

    # Calculate predictor value (variable dt explicit mid point rule)
    y_np1_EMP = y_n + (1 + dtr)*dt_n*dy_n - (dtr**2)*y_n + (dtr**2)*y_nm1

    # Calculate truncation error
    error = (y_np1 - y_np1_EMP) * ((1 + dt_nm1/dt_n)**2) /\
        (1 + 3*(dt_nm1/dt_n) + 4*(dt_nm1/dt_n)**2 + 2*(dt_nm1/dt_n)**3)

    error_norm = sp.linalg.norm(error, 2)

    scaling_factor = (target_error / error_norm)**(1.0/3)
    return _scale_timestep(dt_n, scaling_factor)


class Interpolator(object):
    """ """

    def __init__(self, n_points=4):

        # Number of points to store
        self.n_points = n_points

        # Underlying interpolation function (don't use the object version
        # of it because we want to use new points each time).
        self.underlying_interpolator = krogh_interpolate

    def __call__(self, ts, ys, output_points):

        # Range of t and y values to use
        a, b = -self.n_points-1, -1

        # Interpolate
        interpolated_values = \
            self.underlying_interpolator(ts[a:b], ys[a:b],
                                         output_points, der=[0, 1, 2])

        # Unpack derivatives
        y_nph, dy_nph, ddy_nph = list(interpolated_values)

        # Store the dy value for use in the next time step
        self.dy_nmh = dy_nph

        # Return requested values
        return y_nph, dy_nph, ddy_nph


def midpoint_ab_time_adaptor(ts, ys, target_error, interpolator):
    """

    See notes: 19-20/3/2013 for algebra and explanations.
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

    t_n = ts[-2]
    t_nph = (ts[-1] + ts[-2])/2
    # t_nmh = (ts[-2] + ts[-3])/2

    y_nmid = (y_np1_MP + y_n_MP)/2
    dy_nmid = (y_np1_MP - y_n_MP)/dt_n

    # ??ds
    exact = exp

    # Interpolate exact value and derivatives at t_nph
    # ============================================================
    n_interpolation_points = 6
    a = -n_interpolation_points - 1
    b = -1
    interps = interpolator(ts[a:b], ys[a:b], [t_n, t_nph], der=[0, 1])

    # Unpack (can't get "proper" unpacking to work)
    dy_n = interps[0][1]
    y_nph = interps[1][0]
    dy_nph = interps[1][1]

    # # print "***", dt_n**4, abs(dy_n - exp(ts[-2])), abs(y_nph - exp(t_nph)),
    # # print abs(dy_nph - exp(t_nph))
    # #??ds
    # dy_n = sp.array([exact(t_n)])
    # y_nph = sp.array([exact(t_nph)])
    # dy_nph = sp.array([exact(t_nph)])

    #??ds
    dy_nmh = (y_n_MP - y_nm1_MP)/dt_n

    y_np1_AB = ab2_step(dt_n/2, y_nph, dy_nph, dt_n/2 + dt_nm1/2, dy_nmh)
    a = dt_n*dy_nph + y_n_MP - y_np1_MP

    # Get the truncation error, scaling factor and new time step size
    # ============================================================

    # Calculate (see notes)
    # midpoint_lte = -y_np1_MP - 4*y_nph + dt_n*dy_n + 2*dt_n*dy_nph + 5*y_n_MP
    midpoint_lte = 4*(y_np1_MP - y_np1_AB) + 5*a

    # print -y_np1_MP, -4*y_nph, dt_n*dy_n, 2*dt_n*dy_nph, 5*y_n_MP
    print dt_n, midpoint_lte, (ys[-1] - exact(ts[-1]))/len(ys)

    # Get a norm
    error_norm = sp.linalg.norm(sp.array(midpoint_lte, ndmin=1), 2)

    # Guard against division by zero
    if error_norm < 1e-12:
        scaling_factor = MAX_ALLOWED_DT_SCALING_FACTOR

    # Calculate scaling factor
    else:
        scaling_factor = ((target_error / error_norm)**0.3333)

    # Return the scaled timestep (function does lots of checks).
    return _scale_timestep(dt_n, scaling_factor)


def midpoint_fe_ab_time_adaptor(ts, ys, target_error, interpolator):
    """

    See notes: 19-20/3/2013 for algebra and explanations.
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

    # Get the past value from the interpolator (since we already have it no
    # need to interpolate it again). This must be done before the next
    # interpolation (or the [-1] would become [-2]).
    try:
        dy_nmh = interpolator.dy_nmh

    # If it was out of range then we don't have a value, approximate by
    # dy_mid_nmh (bit of a fudge...)
    except AttributeError:
        dy_nmh = (ys[-2] + ys[-3])/2

    # Interpolate exact value and derivatives at t_nph
    y_nph, dy_nph, ddy_nph = interpolator(ts, ys, t_nph)

    # Use a forward Euler predictor to eliminate the Jacobian term
    # ============================================================

    # Forward Euler with exact initial y and approximate midpoint
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
    midpoint_lte = (omega_n * (y_np1_MP - y_np1_AB2)
                    + ((omega_n - 1)/2) * (y_np1_FEc - y_np1_MP))

    # Get a norm
    error_norm = sp.linalg.norm(sp.array(midpoint_lte, ndmin=1), 2)

    # Guard against division by zero
    if error_norm < 1e-12:
        scaling_factor = MAX_ALLOWED_DT_SCALING_FACTOR

    # Calculate scaling factor
    else:
        scaling_factor = ((target_error / error_norm)**0.3333)

    # Return the scaled timestep (function does lots of checks).
    return _scale_timestep(dt_n, scaling_factor)


# Testing
# ============================================================
import matplotlib.pyplot as plt
from example_residuals import *


def check_problem(method, residual, exact, tol=1e-3, tmax=5.0):
    """Helper function to run odeint with a specified method for a problem
    and check that the solution matches.
    """

    ys, ts = odeint(residual, [exact(0.0)], tmax, dt=1e-6,
                    method=method, target_error=tol)

    # Total error should be bounded by roughly n_steps * LTE
    overall_tol = len(ys) * tol * 10
    utils.assertListAlmostEqual(ys, map(exact, ts), overall_tol)

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

    dts = [1e-6, 1e-6, (1. - 0.01)*tmax]
    initial_ts = list_cummulative_sums(dts[:-1], 0.)
    initial_ys = [sp.array(exp3_exact(t), ndmin=1) for t in initial_ts]

    ys, ts = _odeint(exp3_residual, initial_ys, initial_ts, dts[-1], tmax,
                     bdf2_residual, tol, bdf2_ab_time_adaptor)

    print map(lambda x, y: abs(x-y), ys, map(exp3_exact, ts))
    print ts

    utils.assertAlmostEqual(ys[-1], exp3_exact(tmax), 10*tol)


def test_ab2():
    def dydt(t, y):
        return y
    tmax = 1.0

    # Oscillate dt to check variable timestep maths is ok.
    dt_base = 1e-4
    input_dts = it.cycle([dt_base, 15*dt_base])

    # Starting values
    ts = [0.0, 1e-6]
    ys = map(exp, ts)
    dts = [1e-6]

    while ts[-1] < tmax:
        dtprev = dts[-1]
        dt = input_dts.next()

        y_np1 = ab2_step(dt, ys[-1], dydt(ts[-1], ys[-1]),
                         dtprev, dydt(ts[-2], ys[-2]))
        ys.append(y_np1)
        ts.append(ts[-1] + dt)
        dts.append(dt)

    # plt.plot(ts, ys, 'x', ts, map(exp, ts))
    # plt.show()

    utils.assertAlmostEqual(ys[-1], exp(ts[-1]), 1e-5)


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
        utils.assertAlmostEqual(ys[-1], exp(tmax), tol)

    # List of test parameters
    methods = [('bdf2', 1e-5),
               ('bdf1', 1e-2),  # First order method...
               ('midpoint', 1e-5),
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

        utils.assertAlmostEqual(ys[-1][0], cos(tmax), tol[0])
        utils.assertAlmostEqual(ys[-1][1], exp(tmax), tol[1])

    # List of test parameters
    methods = [('bdf2', [1e-4, 1e-4]),
               ('bdf1', [1e-2, 1e-2]),  # First order methods suck...
               ('midpoint', [1e-4, 1e-4]),
               ('trapezoid', [1e-4, 1e-4]),
               ]

    # Generate tests
    for meth, tol in methods:
        yield check_vector_timestepper, meth, tol


def test_adaptive_dt():

    methods = [('bdf2 ab', 1e-3, 382),
               ('midpoint fe ab', 1e-3, 51),
               ('midpoint ab', 1e-3, 51),
               ]

    functions = [(exp_residual, exp),
                 (exp_of_minus_t_residual, lambda x: exp(-1*x)),
                 (poly_residual, lambda t: t**4 + t**2),
                 (exp_of_poly_residual, lambda t: exp(t - t**3))
                 ]

    for meth, tol, allowed_steps in methods:
        for residual, exact in functions:
            yield check_problem, meth, residual, exact, tol


def test_local_truncation_error():

    tests = [(TrapezoidRuleResidual(), trapezoid_lte),
             (bdf2_residual, bdf2_lte),
             (midpoint_residual, par(midpoint_lte, dfdy=0.)),
             #(bdf1_residual, bdf1_lte),
             ]
    # We can't write a comparable midpoint lte calculation function because
    # we calculate it using values at the midpoint between time steps
    # (c.f. other lte functions which use values at time steps).

    # Auxilary function
    def check_local_truncation_error(method_residual, error_function):
        """Test that a single timestep of the method has local truncation
        error below that given by the appropriate function.
        """
        residual = exp_residual
        value = exp
        d1y = exp
        d2y = exp
        d3y = exp

        tstart = 2.0

        # Can't do smaller than ~1e-4 because in second order methods lte
        # is ~dt^3 and (1e-4)^3 = 1e-12 which is approaching numerical
        # error.
        for dt in [1e-2, 1e-3, 5e-4]:

            # Additional points for higher order methods
            ts = [tstart - 1e-6, tstart]
            ys = map(exp, ts)
            tmax = tstart + dt

            ys, ts = _odeint(residual, ys, ts, dt,
                             tmax, method_residual)

            lte_analytic = error_function(dt, map(value, ts), map(d1y, ts),
                                          map(d2y, ts), map(d3y, ts))

            # Check
            actual_error = ys[-1] - value(tmax)
            print "actual_error =", actual_error
            print "calculated error =", lte_analytic
            print

            # assert(actual_error < lte_analytic)
            utils.assertAlmostEqual(
                actual_error, lte_analytic, abs(2*lte_analytic))
            # ??ds Not sure exactly how close estimate will be...

    for meth_res, tol_func in tests:
        yield check_local_truncation_error, meth_res, tol_func


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
    return check_problem('midpoint fe ab', residual, exact, tol=tol)

# def test_with_stiff_problem():
#     """Check that midpoint fe ab works well for stiff problem (i.e. has a
#     non-insane number of time steps).
#     """
#     # Slow test!

#     mu = 1000
#     residual = par(van_der_pol_residual, mu=mu)

#     ys, ts = odeint(residual, [2.0, 0], 1000.0, dt=1e-6,
#                     method='bdf2 ab', target_error=1e-3)

#     print len(ts)
#     plt.plot(ts, [y[0] for y in ys])
#     plt.plot(ts[1:], utils.ts2dts(ts))
#     plt.show()

#     n_steps = len(ts)
#     assert n_steps < 5000
