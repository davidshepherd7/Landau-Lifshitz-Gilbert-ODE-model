from __future__ import division

from math import sin, cos, tan, log, atan2, acos, pi, sqrt, exp
import scipy as sp
import scipy.integrate
from scipy.optimize import newton_krylov
import scipy.optimize.nonlin
import functools as ft
import itertools as it
import copy

import simplellg.utils as utils

# PARAMETERS
MAX_ALLOWED_DT_SCALING_FACTOR = 4.0
MIN_ALLOWED_DT_SCALING_FACTOR = 0.8
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
        return bdf2_residual, None, ft.partial(higher_order_start, 2)

    elif label == 'bdf2 ab':
        return bdf2_residual, bdf2_ab_time_adaptor,\
          ft.partial(higher_order_start, 2)

    elif label == 'bdf1':
        return bdf1_residual, None, None

    elif label == 'midpoint':
        return midpoint_residual, None, None

    elif label == 'midpoint ab':
        return midpoint_residual, midpoint_ab_time_adaptor, \
          ft.partial(higher_order_start,2)

    elif label == 'trapezoid':
        # TR is actually self starting but due to technicalities with
        # getting derivatives of y from implicit formulas we need an extra
        # starting point.
        return TrapezoidRuleResidual(), None, ft.partial(higher_order_start, 2)

    else:
        message = "Method '"+label+"' not recognised."
        raise ValueError(message)


def higher_order_start(order, func, ys, ts, dt):
    """ Run a few steps of midpoint method, useful for generating extra
    initial data for multi-step methods.
    """
    while len(ys) < order:
        ys, ts = _odeint(func, ys, ts, dt, ts[-1] + dt,
                         midpoint_residual)
    return ys, ts


def odeint(func, y0, tmax, dt,
           method = 'bdf2',
           target_error = None,
           actions_after_timestep = None):

    # Select the method and adaptor
    time_residual, time_adaptor, initialisation_actions = \
      _timestep_scheme_dispatcher(method)

    ts = [0.0] # List of times (floats)
    ys = [sp.array(y0, ndmin=1)] # List of y vectors (ndarrays)

    # Now call the actual function to do the work
    return _odeint(func, ys, ts, dt, tmax, time_residual,
                   target_error, time_adaptor, initialisation_actions,
                   actions_after_timestep)


def _odeint(func, ys, ts, dt, tmax, time_residual,
            target_error = None, time_adaptor = None,
            initialisation_actions = None, actions_after_timestep = None):

    if initialisation_actions is not None:
        ys, ts = initialisation_actions(func, ys, ts, dt)


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
            new_t_np1, new_y_np1 = actions_after_timestep(ts+[t_np1], ys+[y_np1])
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
    dt = ts[-1] - ts[-2]
    y_n = ys[-2]
    y_np1 = ys[-1]

    y_nph = (y_np1 + y_n) * 0.5
    tmid = (ts[-1] + ts[-2]) * 0.5
    dydt = (y_np1 - y_n)/dt

    return base_residual(tmid, y_nph, dydt)


def bdf1_residual(base_residual, ts, ys):
    dt = ts[-1] - ts[-2]
    y_n = ys[-2]
    y_np1 = ys[-1]

    dydt = (y_np1 - y_n) / dt
    return base_residual(ts[-1], y_np1, dydt)


def bdf2_residual(base_residual, ts, ys):
    dt = ts[-1] - ts[-2]
    dtprev = ts[-2] - ts[-3]

    y_np1 = ys[-1]
    y_n = ys[-2]
    y_nm1 = ys[-3]

    a = (2*dt + dtprev)*dtprev
    prefactor = (2*dt + dtprev)/(dt + dtprev)
    alphas = [1.0/dt, -1.0/dt - dt/a, dt/a]

    dydt = prefactor * sp.dot(alphas, [y_np1, y_n, y_nm1])
    return base_residual(ts[-1], y_np1, dydt)


# Assumption: we never actually undo a timestep (otherwise dys will become
# out of sync with ys).
class TrapezoidRuleResidual(object):

    def __init__(self):
        self.dys = []


    def calculate_new_dy_if_needed(self, ts, ys):
        """If dy_n/dt has not been calculated on this step then calculate
        it from previous values of y and dydt by inverting the trapezoid
        rule.
        """
        if len(self.dys) < len(ys) - 1:
            dtprev = ts[-2] - ts[-3]
            dy_n = (2.0 / dtprev) * (ys[-2] - ys[-3]) - self.dys[-1]
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
        dt = 2*(ts[-2] - ts[-3])

        # Calculate time step
        temp_ys, temp_ts = _odeint(base_residual, temp_ys, temp_ts, dt,
                                   temp_ts[-1] + dt, midpoint_residual)

        # Check that we got the right times: the midpoint should be at
        # the step before the most recent time.
        utils.assertAlmostEqual((temp_ts[-1] + temp_ts[-2])/2, ts[-2])

        # Now invert midpoint to get the derivative
        dy_nph = (temp_ys[-1] - temp_ys[-2])/dt

        # Fill in dummys (as many as we have y values) followed by the
        # derivative we just calculated.
        self.dys = [float('nan')] * (len(ys)-1)
        self.dys[-1] = dy_nph


    def __call__(self, base_residual, ts, ys):
        if len(self.dys) == 0:
            self._get_initial_dy(base_residual, ts, ys)

        dt = ts[-1] - ts[-2]
        t_np1 = ts[-1]
        y_n = ys[-2]
        y_np1 = ys[-1]

        self.calculate_new_dy_if_needed(ts, ys)
        dy_n = self.dys[-1]

        dy_np1 = (2.0/dt) * (y_np1 - y_n) - dy_n
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
        error = "Tried to reduce dt to "+ str(scaling_factor * dt) +\
          " which is less than the minimum of "+ str(MIN_ALLOWED_TIMESTEP)
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


def bdf2_lte(dt, ys, dys, d2ys, d3ys):
    """ From Prinja's thesis.
    """
    d3ydt_n = d3ys[-1]
    return (2.0/9) * d3ydt_n * dt**3


def trapezoid_lte(dt, ys, dys, d2ys, d3ys):
    """ Calculated myself, notes: 15/2/13.
    """
    return (1.0/12) * dt**3 * d3ys[-1]


def bdf2_ab_time_adaptor(ts, ys, target_error):
    """ Calculate a new timestep size based on estimating the error from
    the previous timestep. Weighting numbers stolen from oopmh-lib.
    """

    dt = ts[-1] - ts[-2]

    # Fudge the first step
    if len(ys) < 4: return dt

    dtprev = ts[-2] - ts[-3]
    dtr = dt / dtprev

    # oomph-lib based version:

    # # Set up weights
    # weights = [0.0, 1 - dtr**2, dtr**2, (1 + dtr) * dt]
    # error_weight = (1 + dtr)**2 / \
    #   (1.0 + 3*dtr + 4*(dtr**2) + 2*(dtr**3))

    # # Calculate error estimate
    # predictor_solutions = sp.dot(weights, [ys[-1], ys[-2], ys[-3], ys[-4]])
    # errors_est = error_weight * (ys[-1] - predictor_solutions)

    # # Get a norm if we have an array, otherwise just take the absolute
    # # value
    # error_norm = sp.linalg.norm(errors_est, 2)

    # My implementation directly from G&S:

    y_nm2 = ys[-4]
    y_nm1 = ys[-3]
    y_n = ys[-2]
    y_np1 = ys[-1]

    # Invert bdf to get predictor data
    dy_n = (3*y_n - 4*y_nm1 + y_nm2)/ (2 * dt) #??ds wrong according to milan

    # # Inverting bdf2 from gresho and sani (3.16-247)
    # dt_nm1 = dtprev
    # dt_nm2 = ts[-3] - ts[-4]
    # dy_n = (1/((dt_nm1 + dt_nm2)/(2*dt_nm1 + dt_nm2))) * \
    #   ( (y_n - y_nm1)/dt_nm1 - ((y_nm1 - y_nm2)/dt_nm2) * (dt_nm1/ (2*dt_nm1 + dt_nm2)))

    # Calculate predictor value (variable dt explicit mid point rule)
    y_np1_EMP = y_n + (1 + dtr)*dt*dy_n - (dtr**2)*y_n + (dtr**2)*y_nm1

    # Calculate truncation error

    error = (y_np1 - y_np1_EMP) * ((1 + dtprev/dt)**2) /\
             (1 + 3*(dtprev/dt) + 4*(dtprev/dt)**2  + 2*(dtprev/dt)**3)
    error_norm = sp.linalg.norm(error, 2)


    scaling_factor = (target_error / error_norm)**(1.0/3)
    return _scale_timestep(dt, scaling_factor)


def midpoint_ab_time_adaptor(ts, ys, target_error):
    """

    See notes: 7/2/2013 for the algebra on calculating the AB2
    solution. See "mathematica_adaptive_midpoint.m for algebra to get a
    local truncation error out of all this.
    """

    dt = ts[-1] - ts[-2]
    dtprev = ts[-2] - ts[-3]
    y_np1_MP = ys[-1]
    y_nm1_MP = ys[-3]


    # Get explicit adams-bashforth 2 solution (variable timestep -- uses
    # steps at n + 1/2 and n - 1/2).
    # ============================================================

    # Get y derivatives at previous midpoints (this gives no additional
    # loss of accuracy because we are just inverting the midpoint method to
    # get back the derivative).
    dy_nph = (ys[-1] - ys[-2])/dt
    dy_nmh = (ys[-2] - ys[-3])/dtprev

    # Approximate y at step n+1/2 by averaging.
    y_nph = (ys[-1] + ys[-2]) * 0.5 # (ys[-1] + ys[-2])/2

    # Calculate the corresponding fictional timestep sizes
    AB_dt = 0.5*dt
    AB_dtprev = 0.5*dt + 0.5*dtprev

    # Calculate using AB2 variable timestep formula
    y_np1_AB2 = ab2_step(AB_dt, y_nph, dy_nph,
                        AB_dtprev, dy_nmh)


    # Choose an estimate for the derivatives of y at time t_{n+0.5}
    # ============================================================

    # # Finite difference dy/dt from neighbouring y values
    # dy_n = (y_np1_MP - y_nm1_MP)/(dt + dtprev)

    # # Average of midpoints. Accuracy should be O(h^2)?
    # dy_n = (dy_nph + dy_nmh) / 2

    # Ignore d2y/dt2 (conservative estimate).
    ddy_nph = 0.

    # # Finite difference dy2/dt2 from neighbouring dy values. Not sure on
    # # the accuracy here...
    # ddy_n = (dy_nph - dy_nmh) / (dt/2 + dtprev/2)

    # dfdy term: can't calculate properly since it is a matrix so just
    # ignore it for now.
    dfdy_nph = 0.


    # Get the truncation error and scaling factor
    # ============================================================

    # As derived in notes from 21/2/13
    a = 4 / (1 + 3*(dtprev/dt))
    error = (y_np1_AB2 - y_np1_MP) * a - (dt**3 * (1+a)/8) * ddy_nph * dfdy_nph

    # print (y_np1_AB2 - y_np1_MP) * a
    # print dt**3 , (1+a)/8 , ddy_nph , dfdy_nph


    # Get a norm and scaling factor
    error_norm = sp.linalg.norm(sp.array(error, ndmin=1), 2)

    if error_norm < 1e-12:
        scaling_factor = MAX_ALLOWED_DT_SCALING_FACTOR
    else:
        scaling_factor = ((target_error / error_norm)**0.3333)

    return _scale_timestep(dt, scaling_factor)


# Testing
# ============================================================
import matplotlib.pyplot as plt

# Some residuals for testing with

def exp_residual(ts, y, dydt):
    """y = exp(t) """
    return y - dydt


def exp_of_minus_t_residual(ts, y, dydt):
    """ y = exp(-t) """
    return y + dydt


def poly_residual(t, y, dydt):
    """ y = t**4 + t**2 """
    return 4*t**3 + 2*t - dydt


def exp_of_poly_residual(t, y, dydt):
    """ y = exp(t - t**3)"""
    return y*(1 - 3*t**2) - dydt


def test_bad_timestep_handling():
    """ Check that rejecting timesteps works.
    """
    tmax = 0.001
    tol = 1e-5

    def list_cummulative_sums(values, start):
        temp = [start]
        for v in values:
            temp.append(temp[-1] + v)
        return temp

    dts = [1e-6, 1e-6, 1.0]
    initial_ts = list_cummulative_sums(dts[:-1], 0.)
    initial_ys = [sp.array(exp(t), ndmin=1) for t in initial_ts]
    ys, ts = _odeint(exp_residual, initial_ys, initial_ts, dts[-1], tmax,
                     bdf2_residual, tol, bdf2_ab_time_adaptor)
    utils.assertAlmostEqual(ys[-1], exp(tmax), 10 * tol)


def test_ab2():
    def dydt(t, y): return y
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
        def residual(t, y, dydt): return y - dydt
        tmax = 1.0
        dt = 0.001
        ys, ts = odeint(exp_residual, [exp(0.0)], tmax, dt = dt,
                         method = method)

        # plt.plot(ts,ys)
        # plt.plot(ts, map(exp,ts), '--r')
        # plt.show()
        utils.assertAlmostEqual(ys[-1], exp(tmax), tol)

    # List of test parameters
    methods = [('bdf2', 1e-5),
               ('bdf1', 1e-2), # First order method...
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
        ys, ts = odeint(residual, [cos(0.0), exp(0.0)], tmax, dt = 0.001,
                             method = method)

        utils.assertAlmostEqual(ys[-1][0], cos(tmax), tol[0])
        utils.assertAlmostEqual(ys[-1][1], exp(tmax), tol[1])

    # List of test parameters
    methods = [('bdf2', [1e-4, 1e-4]),
               ('bdf1', [1e-2, 1e-2]), # First order methods suck...
               ('midpoint', [1e-4, 1e-4]),
               ('trapezoid', [1e-4, 1e-4]),
               ]

    # Generate tests
    for meth, tol in methods:
        yield check_vector_timestepper, meth, tol


def test_adaptive_dt():

    # Aux checking function
    def check_adaptive_dt(method, tol, steps, residual, exact):
        tmax = 6.0
        ys, ts = odeint(residual, [exact(0.0)], tmax, dt = 1e-3,
                        method = method, target_error = tol)

        # dts = utils.ts2dts(ts)
        # plt.plot(ts, ys, 'x', ts, map(exact, ts))
        # dts = utils.ts2dts(ts)
        # plt.title(str(residual))

        # plt.figure()
        # plt.plot(ts[:-1],dts)
        # plt.show()

        # Total error is bounded by roughly n_steps * LTE, add a "fudge
        # factor" of 10 because LTE is only an estimate.
        overall_tol = len(ys) * tol * 10

        print "nsteps =", len(ys)
        print "error = ", abs(ys[-1][0] - exact(tmax))
        print "tol = ", overall_tol

        utils.assertAlmostEqual(ys[-1][0], exact(tmax), overall_tol)
        #utils.assertAlmostEqual(len(ys), steps, steps * 0.1)

    methods = [#('bdf2 ab', 1e-3, 382),
               ('midpoint ab', 1e-6, 51),
               ]
    functions = [# (exp_residual, exp),
                 # (exp_of_minus_t_residual, lambda x: exp(-1*x)),
                 # (poly_residual, lambda t: t**4 + t**2),
                 (exp_of_poly_residual, lambda t: exp(t - t**3))
                 ]

    for meth, requested_tol, allowed_steps in methods:
        for residual, exact in functions:
            yield check_adaptive_dt, meth, requested_tol, allowed_steps, \
              residual, exact



def test_local_truncation_error():

    tests = [(TrapezoidRuleResidual(), trapezoid_lte),
             (bdf2_residual, bdf2_lte),
              #(midpoint_residual, midpoint_lte),
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

            #assert(actual_error < lte_analytic)
            utils.assertAlmostEqual(actual_error, lte_analytic, abs(2*lte_analytic))
            # ??ds Not sure exactly how close estimate will be...

    for meth_res, tol_func in tests:
        yield check_local_truncation_error, meth_res, tol_func
