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
MAX_ALLOWED_DT_SCALING_FACTOR = 3.0
MIN_ALLOWED_DT_SCALING_FACTOR = 0.75
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


def _timestep_scheme_factory(label):
    """Pick the functions needed for this method. Returns functions for a
    time residual, a timestep adaptor and any intialisation actions needed.
    """
    label = label.lower()

    if label == 'bdf2':
        return bdf2_residual, None, par(higher_order_start, 2)

    elif label == 'bdf2 mp':
        return bdf2_residual, bdf2_mp_time_adaptor,\
            par(higher_order_start, 2)

    elif label == 'bdf1':
        return bdf1_residual, None, None

    elif label == 'midpoint':
        return midpoint_residual, None, None

    elif label == 'midpoint ab':
        n_start = 6
        adaptor = par(midpoint_ab_time_adaptor,
                      interpolator=krogh_interpolate,
                      n_interp=n_start)
        return midpoint_residual, adaptor, par(higher_order_start, n_start)

    elif label == 'midpoint fe ab':
        n_start = 6
        adaptor = par(midpoint_fe_ab_time_adaptor,
                      interpolator=krogh_interpolate,
                      n_interp=n_start)
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
        _timestep_scheme_factory(method)

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
            dt = scale_timestep(dt, _, _, failed_timestep_scaling)
            sys.stderr.write("Failed to converge, reducing time step.")
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
                dt = exception_data.new_dt
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
    """ Calculate residual at latest time and y-value with the bdf2
    approximation for the y derivative.
    """
    return base_residual(ts[-1], ys[-1], bdf2_dydt(ts, ys))


def bdf2_dydt(ts, ys):
    """Get dy/dt at time ts[-1] (allowing for varying dt).
    Gresho & Sani, pg. 715"""
    dt_n = ts[-1] - ts[-2]
    dt_nm1 = ts[-2] - ts[-3]

    y_np1 = ys[-1]
    y_n = ys[-2]
    y_nm1 = ys[-3]

    # Copied from oomph-lib (algebraic rearrangement of G&S forumla).
    weights = [1.0/dt_n + 1.0/(dt_n + dt_nm1),
               - (dt_n + dt_nm1)/(dt_n * dt_nm1),
               dt_n / ((dt_n + dt_nm1) * dt_nm1)]

    dydt = sp.dot(weights, [y_np1, y_n, y_nm1])

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
        utils.assert_almost_equal((temp_ts[-1] + temp_ts[-2])/2, ts[-2])

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


def default_dt_scaling(target_error, error_estimate, timestepper_order):
    """Standard way of rescaling the time step to attain the target error.
    Taken from Gresho and Sani (various places).
    """
    try:
        scaling_factor = (target_error/error_estimate)**(1.0/timestepper_order)

    except ZeroDivisionError:
        scaling_factor = MAX_ALLOWED_DT_SCALING_FACTOR

    return scaling_factor


def failed_timestep_scaling(_):
    """Return scaling factor for a failed time step, ignores all input
    arguments.
    """
    return TIMESTEP_FAILURE_DT_SCALING_FACTOR


def scale_timestep(dt, target_error, error_norm, order,
                   scaling_function=default_dt_scaling):
    """Scale dt by a scaling factor. Mostly this function is needed to
    check that the scaling factor and new time step are within the
    allowable bounds.
    """

    # Calculate the scaling factor
    scaling_factor = scaling_function(target_error, error_norm, order)

    # If the error is too bad (i.e. scaling factor too small) reject the
    # step, unless we are already dealing with a rejected step;
    if scaling_factor < MIN_ALLOWED_DT_SCALING_FACTOR \
            and not scaling_function is failed_timestep_scaling:
        raise FailedTimestepError(scaling_factor * dt)

    # or if the scaling factor is really large just use the max scaling.
    elif scaling_factor > MAX_ALLOWED_DT_SCALING_FACTOR:
        scaling_factor = MAX_ALLOWED_DT_SCALING_FACTOR

    # If the timestep would get too big then return the max time step;
    if scaling_factor * dt > MAX_ALLOWED_TIMESTEP:
        return MAX_ALLOWED_TIMESTEP

    # or if the timestep would become too small then fail;
    elif scaling_factor * dt < MIN_ALLOWED_TIMESTEP:
        error = "Tried to reduce dt to " + str(scaling_factor * dt) +\
            " which is less than the minimum of " + str(MIN_ALLOWED_TIMESTEP)
        raise ValueError(error)

    # otherwise scale the timestep normally.
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


def bdf2_mp_time_adaptor(ts, ys, target_error):
    """ Calculate a new timestep size based on estimating the error from
    the previous timestep. Weighting numbers stolen from oopmh-lib.
    """

    # Get local values (makes maths more readable)
    dt_n = ts[-1] - ts[-2]
    dt_nm1 = ts[-2] - ts[-3]
    dtr = dt_n / dt_nm1

    y_np1 = ys[-1]
    y_n = ys[-2]
    y_nm1 = ys[-3]

    # Invert bdf2 to get predictor data (using the exact same function as
    # was used in the residual calculation).
    dy_n = bdf2_dydt(ts, ys)

    # # Calculate predictor value (variable dt explicit mid point rule) --G&S version
    # y_np1_EMP = y_n + (1 + dtr)*dt_n*dy_n - (dtr**2)*y_n + (dtr**2)*y_nm1

    # Calculate predictor value (variable dt explicit mid point rule) --oomph-
    # lib version
    y_np1_EMP =  (1.0 - dtr**2) * y_n\
        + (dtr**2) * y_nm1 \
        + (1.0 + dtr)*dt_n * dy_n

    # # Calculate truncation error -- G&S
    # error = (y_np1 - y_np1_EMP) * ((1 + dt_nm1/dt_n)**2) /\
    #     (1 + 3*(dt_nm1/dt_n) + 4*(dt_nm1/dt_n)**2 + 2*(dt_nm1/dt_n)**3)

    dtrinv = 1.0 / dtr
    error_weight = ((1.0 + dtrinv)**2) / \
        (1.0 + 3.0*dtrinv + 4.0 * dtrinv**2
         + 2.0 * dtrinv**3)

    # Calculate truncation error -- oomph-lib
    error = (y_np1 - y_np1_EMP) * error_weight
    error_norm = sp.linalg.norm(error, 2)

    return scale_timestep(dt_n, target_error, error_norm, 3)


def my_interpolate(ts, ys, interpolator, n_interp, use_y_np1_in_interp):
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
    interps = interpolator(
        ts[start:end], ys[start:end], [t_nmh, t_nph], der=[0, 1, 2])

    # Unpack (can't get "proper" unpacking to work)
    dy_nmh = interps[1][0]
    y_nph = interps[0][1]
    dy_nph = interps[1][1]
    ddy_nph = interps[2][1]

    return dy_nmh, y_nph, dy_nph, ddy_nph

# class midpoint_ab_time_adaptor(object):
#     """ Utility storage class for passing variables between function calls."""
#     def __call__(*args):
#         return midpoint_ab_time_adaptor(*args, T_nm1=self.Tnm1)


def midpoint_ab_time_adaptor(ts, ys, target_error, interpolator,
                             n_interp=4,
                             use_y_np1_in_interp=True):
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
    t_nph = (ts[-1] + ts[-2])/2
    t_nmh = (ts[-2] + ts[-3])/2

    y_nmid = (y_np1_MP + y_n_MP)/2
    dy_nmid = (y_np1_MP - y_n_MP)/dt_n
    dy_nm1_mid = (ys[-3] - ys[-2])/dt_nm1

    # Interpolate exact value and derivatives at t_nph
    # ============================================================
    dy_nmh, y_nph, dy_nph, _ = my_interpolate(ts, ys, interpolator,
                                              n_interp, use_y_np1_in_interp)

    # Calculate this part of the truncation error (see notes)
    ymid_estimation_error = dt_n*dy_nph + y_n_MP - y_np1_MP

    # Use an AB2 predictor to eliminate the dddy_nph term
    # ============================================================
    AB_dt_n = 0.5*dt_n
    AB_dt_nm1 = 0.5*dt_n + 0.5*dt_nm1

    y_np1_AB2 = ab2_step(AB_dt_n, y_nph, dy_nph,
                         AB_dt_nm1, dy_nmh)

    # Get the truncation error, scaling factor and new time step size
    # ============================================================
    # Calculate (see notes)
    # midpoint_lte = 4*(y_np1_MP - y_np1_AB2) + 5*ymid_estimation_error
    midpoint_lte = ymid_estimation_error + \
        4 * (3.0/dtr - 1) * (y_np1_AB2 - y_np1_MP - ymid_estimation_error)

    # print exp(t_nmh) - dy_nmh, exp(t_nph) - y_nph,exp(t_nph) -  dy_nph, dt_n
    # print ymid_estimation_error, y_np1_AB2 - y_np1_MP

    # if len(ts) > 100: assert False

    # Get a norm
    error_norm = sp.linalg.norm(sp.array(midpoint_lte, ndmin=1), 2)

    # Return the scaled timestep (with lots of checks).
    return scale_timestep(dt_n, target_error, error_norm, 3)


def midpoint_fe_ab_time_adaptor(ts, ys, target_error, interpolator,
                                n_interp=4,
                                use_y_np1_in_interp=False):
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

    dy_nmh, y_nph, dy_nph, ddy_nph = \
        my_interpolate(ts, ys, interpolator, n_interp, use_y_np1_in_interp)

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

    # Return the scaled timestep (with lots of checks).
    return scale_timestep(dt_n, target_error, error_norm, 3)


# Testing
# ============================================================
import matplotlib.pyplot as plt
from example_residuals import *


def check_problem(method, residual, exact, tol=1e-4, tmax=2.0):
    """Helper function to run odeint with a specified method for a problem
    and check that the solution matches.
    """

    ys, ts = odeint(residual, [exact(0.0)], tmax, dt=1e-6,
                    method=method, target_error=tol)

    # Total error should be bounded by roughly n_steps * LTE
    overall_tol = len(ys) * tol * 10
    utils.assert_list_almost_equal(ys, map(exact, ts), overall_tol)

    # if 'midpoint ab' in method:
    #     plt.plot(ts, ys)
    #     plt.plot(ts[1:], utils.ts2dts(ts))
    #     plt.title(method + str(residual) + str(exact))
    #     ys, ts = odeint(residual, [exact(0.0)], tmax, dt=1e-6,
    #                     method='bdf2 mp', target_error=tol)
    #     plt.plot(ts[1:], utils.ts2dts(ts))
    #     plt.show()

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
                     bdf2_residual, tol, bdf2_mp_time_adaptor)

    # plt.plot(ts,ys)
    # plt.plot(ts[1:], utils.ts2dts(ts))
    # plt.plot(ts, map(exp3_exact, ts))
    # plt.show()

    overall_tol = len(ys) * tol * 1.2  # 1.2 is a fudge factor...
    utils.assert_list_almost_equal(ys, map(exp3_exact, ts), overall_tol)


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

    utils.assert_almost_equal(ys[-1], exp(ts[-1]), 1e-5)


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

        utils.assert_almost_equal(ys[-1][0], cos(tmax), tol[0])
        utils.assert_almost_equal(ys[-1][1], exp(tmax), tol[1])

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

    methods = [('bdf2 mp', 1e-4),
               ('midpoint fe ab', 1e-4),
               ('midpoint ab', 1e-4),
               ]

    functions = [(exp_residual, exp),
                 (exp_of_minus_t_residual, exp_of_minus_t_exact),
                 (poly_residual, poly_exact),
                 (exp_of_poly_residual, exp_of_poly_exact)
                 ]

    for meth, tol in methods:
        for residual, exact in functions:
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
    return check_problem('midpoint ab', residual, exact, tol=tol)

# def test_with_stiff_problem():
#     """Check that midpoint fe ab works well for stiff problem (i.e. has a
#     non-insane number of time steps).
#     """
#     # Slow test!

#     mu = 1000
#     residual = par(van_der_pol_residual, mu=mu)

#     ys, ts = odeint(residual, [2.0, 0], 1000.0, dt=1e-6,
#                     method='midpoint ab', target_error=1e-3)

#     print len(ts)
#     plt.plot(ts, [y[0] for y in ys])
#     plt.plot(ts[1:], utils.ts2dts(ts))
#     plt.show()

#     n_steps = len(ts)
#     assert n_steps < 5000
