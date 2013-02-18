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
MIN_ALLOWED_DT_SCALING_FACTOR = 0.9
TIMESTEP_FAILURE_DT_SCALING_FACTOR = 0.5

MIN_ALLOWED_TIMESTEP = 1e-8
MAX_ALLOWED_TIMESTEP = 1e8


# TODO

# Fix data flow in _odeint--having updated and non-updated lists around is
# strange...


# Data storage notes
# ============================================================

# Y values and time values are stored in lists throughout (for easy
# appending).

# The final value in each of these lists (accessed with [-1]) is the most
# recent.

# Almost always the most recent value in the list is the current
# guess/result for ynp1/tnp1 (i.e. y_{n+1}, t_{n+1}, i.e. the value being
# calculated), the previous is yn, etc.

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

        tnp1 = ts[-1] + dt

        # Fill in the residual for calculating dydt and the previous time
        # and y values ready for the Newton solver.
        residual = lambda ynp1: time_residual(func, ts+[tnp1], ys+[ynp1])

        # Try to solve the system, using the previous y as an initial
        # guess. If it fails reduce dt and try again.
        try:
            ynp1 = newton_krylov(residual, ys[-1], f_tol=1e-8, method='gmres')
        except sp.optimize.nonlin.NoConvergence:
            dt = _scale_timestep(dt, TIMESTEP_FAILURE_DT_SCALING_FACTOR, True)
            continue

        # Execute any post-step actions requested (e.g. renormalisation,
        # simplified mid-point method update).
        uts, uys = ts+[ts[-1]+dt], ys+[ynp1]
        if actions_after_timestep is not None:
            uts, uys = actions_after_timestep(uts, uys)

        # Calculate the next value of dt if needed
        if time_adaptor is not None:
            try:
                dt = time_adaptor(uts, uys, target_error)

            # If the scaling factor is too small then don't store this
            # timestep, instead repeat it with the new step size.
            except FailedTimestepError, exception:
                dt = exception.new_dt
                continue

        # Update results storage (don't do this earlier in case the time
        # step fails).
        ys.append(uys[-1])
        ts.append(uts[-1])

    return ys, ts


# Timestepper residual calculation functions
# ============================================================

def midpoint_residual(base_residual, ts, ys):
    dt = ts[-1] - ts[-2]
    yn = ys[-2]
    ynp1 = ys[-1]

    ymid = (ynp1 + yn) * 0.5
    tmid = ts[-2] + (dt*0.5)
    dydt = (ynp1 - yn)/dt
    return base_residual(tmid, ymid, dydt)


def bdf1_residual(base_residual, ts, ys):
    dt = ts[-1] - ts[-2]
    yn = ys[-2]
    ynp1 = ys[-1]

    dydt = (ynp1 - yn) / dt
    return base_residual(ts[-1], ynp1, dydt)


def bdf2_residual(base_residual, ts, ys):
    dt = ts[-1] - ts[-2]
    dtprev = ts[-2] - ts[-3]

    ynp1 = ys[-1]
    yn = ys[-2]
    ynm1 = ys[-3]

    a = (2*dt + dtprev)*dtprev
    prefactor = (2*dt + dtprev)/(dt + dtprev)
    alphas = [1.0/dt, -1.0/dt - dt/a, dt/a]

    dydt = prefactor * sp.dot(alphas, [ynp1, yn, ynm1])
    return base_residual(ts[-1], ynp1, dydt)


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
            dyn = (2.0 / dtprev) * (ys[-2] - ys[-3]) - self.dys[-1]
            self.dys.append(dyn)


    def _get_initial_dy(self, base_residual, ts, ys):
        """Calculate a step with midpoint to get dydt at yn.
        """

        # We want to ignore the most recent two steps (the one being solved
        # for "now" outside this function and the one we are computing the
        # derivative at). We also want to ensure nothing is modified in the
        # solutions list:
        temp_ts = copy.deepcopy(ts[:-2])
        temp_ys = copy.deepcopy(ys[:-2])

        # Timestep should be double the timestep used for the previous
        # step, so that the midpoint is at yn.
        dt = 2*(ts[-2] - ts[-3])

        # Calculate time step
        temp_ys, temp_ts = _odeint(base_residual, temp_ys, temp_ts, dt,
                                   temp_ts[-1] + dt, midpoint_residual)

        # Check that we got the right times: the midpoint should be at
        # the step before the most recent time.
        utils.assertAlmostEqual((temp_ts[-1] + temp_ts[-2])/2, ts[-2])

        # Now invert midpoint to get the derivative
        ydot_nphalf = (temp_ys[-1] - temp_ys[-2])/dt

        # Fill in dummys (as many as we have y values) followed by the
        # derivative we just calculated.
        self.dys = [float('nan')] * (len(ys)-1)
        self.dys[-1] = ydot_nphalf


    def __call__(self, base_residual, ts, ys):
        if len(self.dys) == 0:
            self._get_initial_dy(base_residual, ts, ys)

        dt = ts[-1] - ts[-2]
        tnp1 = ts[-1]
        yn = ys[-2]
        ynp1 = ys[-1]

        self.calculate_new_dy_if_needed(ts, ys)
        dyn = self.dys[-1]

        dynp1 = (2.0/dt) * (ynp1 - yn) - dyn
        return base_residual(tnp1, ynp1, dynp1)

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


def ab2_step(dtn, yn, dyn, dtnm1, dynm1):
    """ Calculate the solution at time n.

    From: my code... ??ds
    """

    dtr = (dtn*1.0)/dtnm1
    ynp1 = yn + 0.5*dtn*((2 + dtr)*dyn -  dtr*dynm1)
    return ynp1


def bdf2_lte(dt, ys, dys, d2ys, d3ys):
    """ From Prinja's thesis.
    """
    d3ydt_n = d3ys[-1]
    return (2.0/9) * d3ydt_n * dt**3


def trapezoid_lte(dt, ys, dys, d2ys, d3ys):
    """ Calculated myself, notes: 15/2/13.
    """
    return (1.0/12) * dt**3 * d3ys[-1]


def midpoint_lte(dt, ys, dys, d2ys, d3ys):
    """ From Milan
    """
    # Milan's:
    return (d3ys[-1] * dys[-1] - 2*(d2ys[-1]**2)) * dt**3 / 24

    # # Other guy
    # return (d3ys[-1] * dys[-1] - 2*(d2ys[-1]**2)) * dys[-1] * dt**3 / 24


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

    ynm2 = ys[-4]
    ynm1 = ys[-3]
    yn = ys[-2]
    ynp1 = ys[-1]

    # Invert bdf to get predictor data
    dyn = (3*yn - 4*ynm1 + ynm2)/ (2 * dt) #??ds wrong according to milan

    # # Inverting bdf2 from gresho and sani (3.16-247)
    # dtnm1 = dtprev
    # dtnm2 = ts[-3] - ts[-4]
    # dyn = (1/((dtnm1 + dtnm2)/(2*dtnm1 + dtnm2))) * \
    #   ( (yn - ynm1)/dtnm1 - ((ynm1 - ynm2)/dtnm2) * (dtnm1/ (2*dtnm1 + dtnm2)))

    # Calculate predictor value (variable dt explicit mid point rule)
    ynp1_EMP = yn + (1 + dtr)*dt*dyn - (dtr**2)*yn + (dtr**2)*ynm1

    # Calculate truncation error

    error = (ynp1 - ynp1_EMP) * ((1 + dtprev/dt)**2) /\
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
    ynp1_MP = ys[-1]
    ynm1_MP = ys[-3]


    # Get explicit adams-bashforth 2 solution (variable timestep -- uses
    # steps at n + 1/2 and n - 1/2)
    # ============================================================

    # Get y derivatives at previous midpoints (this gives no additional
    # loss of accuracy because we are just inverting the midpoint method to
    # get back the derivative used).
    ydot_nphalf = (ys[-1] - ys[-2])/dt
    ydot_nmhalf = (ys[-2] - ys[-3])/dtprev

    # Approximate y at time n+1/2 by averaging (same as is used in
    # midpoint).
    y_nphalf = (ys[-1] + ys[-2])/2

    # Calculate the corresponding fictional timestep sizes
    AB_dt = 0.5*dt
    AB_dtprev = 0.5*dt + 0.5*dtprev

    # Calculate using AB2 variable timestep formula
    ynp1_AB2 = ab2_step(AB_dt, y_nphalf, ydot_nphalf,
                        AB_dtprev, ydot_nmhalf)


    # Choose an estimate for the time derivatives of y at time tn
    # ============================================================

    # # Finite difference dy/dt from neighbouring y values
    # ydotn = (ynp1_MP - ynm1_MP)/(dt + dtprev)

    # Average of midpoints. Accuracy should be O(h^2)?
    ydotn = (ydot_nphalf + ydot_nmhalf) / 2

    # Ignore d2y/dt2 (conservative estimate?).
    ydotdotn = 0.0

    # # Finite difference dy2/dt2 from neighbouring dy values. Not sure on
    # # the accuracy here...
    # ydotdotn = (ydot_nphalf - ydot_nmhalf) / (dt/2 + dtprev/2)

    # Get the truncation error and scaling factor
    # ============================================================

    error = (-6.0 * (ynp1_AB2 - ynp1_MP) * ydotn + 5.0 * dt**3 * ydotdotn**2 ) \
        / (6.0 * ( -10.0 + ydotn))

    # Get a norm if we have an array, otherwise just take the absolute
    # value
    try: error_norm = sp.linalg.norm(error, 2)
    except ValueError: error_norm = abs(error)

    scaling_factor = ((target_error / error_norm)**0.33333)
    #??ds what should the power here be?
    # 0.5 works much better than 1/3, but Milan says it should always be
    # 1/3 because it's a second order method...

    print scaling_factor, dt

    return _scale_timestep(dt, scaling_factor)


# Testing
# ============================================================
import matplotlib.pyplot as plt


def test_bad_timestep_handling():
    """ Check that rejecting timesteps works.
    """
    def residual(t, y, dydt): return y - dydt
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
    ys, ts = _odeint(residual, initial_ys, initial_ts, dts[-1], tmax,
                     bdf2_residual, tol, bdf2_ab_time_adaptor)
    utils.assertAlmostEqual(ys[-1], exp(tmax), 10 * tol)


def test_ab2():
    def dydt(t, y): return y
    tmax = 1.0

    # Oscillate dt a little to check variable timestep maths is ok.
    dt_base = 1e-4
    input_dts = it.cycle([dt_base, 5*dt_base])

    # Starting values
    ts = [0.0, 1e-6]
    ys = map(exp, ts)
    dts = [1e-6]

    while ts[-1] < tmax:
        dtprev = dts[-1]
        dt = input_dts.next()

        ynp1 = ab2_step(dt, ys[-1], dydt(ts[-1], ys[-1]),
                                 dtprev, dydt(ts[-2], ys[-2]))
        ys.append(ynp1)
        ts.append(ts[-1]+dt)
        dts.append(dt)

    utils.assertAlmostEqual(ys[-1], exp(ts[-1]), 1e-5)


def test_exp_timesteppers():

    # Auxilary checking function
    def check_exp_timestepper(method, tol):
        def residual(t, y, dydt): return y - dydt
        tmax = 1.0
        dt = 0.001
        ys, ts = odeint(residual, [exp(0.0)], tmax, dt = dt,
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
    def check_adaptive_dt(method, tol, steps):
        def residual(ts, ys, dydt): return ys - dydt
        tmax = 1.0
        ys, ts = odeint(residual, [exp(0.0)], tmax, dt = 1e-6,
                        method = method, target_error = tol)

        # dts = utils.ts2dts(ts)
        # plt.plot(ts,ys,'--', ts, map(exp, ts))
        # dts = utils.ts2dts(ts)
        # plt.figure()
        # plt.plot(ts[:-1],dts)
        # plt.show()

        print "nsteps =", len(ys)

        utils.assertAlmostEqual(ys[-1][0], exp(tmax), tol * 5)
        utils.assertAlmostEqual(len(ys), steps, steps * 0.1)

    methods = [('bdf2 ab', 1e-3, 382),
               ('midpoint ab', 1e-3, 105),
               ]
    for meth, requested_tol, allowed_steps in methods:
        yield check_adaptive_dt, meth, requested_tol, allowed_steps


def test_local_truncation_error():
    tests = [(TrapezoidRuleResidual(), trapezoid_lte),
             (bdf2_residual, bdf2_lte),
             (midpoint_residual, midpoint_lte),
              #(bdf1_residual, bdf1_lte),
              ]

    # Auxilary function
    def check_local_truncation_error(method_residual, error_function):
        """Test that a single timestep of the method has local truncation
        error below that given by the appropriate function.
        """
        # define the function we are approximating (exponential)
        def residual(t, y, dydt): return y - dydt
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
