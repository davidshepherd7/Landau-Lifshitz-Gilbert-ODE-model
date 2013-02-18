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
        return midpoint_residual, midpoint_ab_time_adaptor, ft.partial(higher_order_start,2)

    else:
        s = "Method '"+label+"' not recognised."
        raise ValueError(s)


def higher_order_start(order, func, ys, ts, dts):
    """ Run a few steps of midpoint method, useful for generating extra
    initial data for multi-step methods.
    """
    while len(ys) < order:
        ys, ts, dts = _odeint(func, ys, ts, dts+[dts[-1]],\
                              ts[-1] + dts[-1], midpoint_residual)
    return ys, ts, dts


def odeint(func, y0, tmax, dt,
           method = 'bdf2',
           target_error = None,
           actions_after_timestep = None):

    # Select the method and adaptor
    time_residual, time_adaptor, initialisation_actions = \
      _timestep_scheme_dispatcher(method)

    ts = [0.0] # List of times (floats)
    ys = [sp.array(y0, ndmin=1)] # List of y vectors (ndarrays)
    dts = [dt] # List of time steps (floats)

    # Now call the actual function to do the work
    return _odeint(func, ys, ts, dts, tmax, time_residual,
                   target_error, time_adaptor, initialisation_actions,
                   actions_after_timestep)


def _odeint(func, ys, ts, dts, tmax, time_residual,
            target_error = None, time_adaptor = None,
            initialisation_actions = None, actions_after_timestep = None):

    if initialisation_actions is not None:
        ys, ts, dts = initialisation_actions(func, ys, ts, dts)

    # Get the dt requested and remove it from the list (in case it fails).
    dt = dts[-1]
    dts = dts[:-1]

    # Main timestepping loop
    # ============================================================
    while ts[-1] < tmax:

        # Fill in the residual for calculating dydt and the known values:
        # t, dt, previous y values (in reverse order, done as a slice)
        # ready for the Newton solver.
        residual = ft.partial(time_residual, func, ts[-1], dts+[dt], ys)

        # Try to solve the system, using the previous y as an initial
        # guess. If it fails reduce dt and try again.
        try: ynp1 = newton_krylov(residual, ys[-1])
        except sp.optimize.nonlin.NoConvergence:
            dt = _scale_timestep(dt, TIMESTEP_FAILURE_DT_SCALING_FACTOR, True)
            continue

        # Execute any post-step actions requested (e.g. renormalisation,
        # simplified mid-point method update).
        uts, uys, udts = ts+[ts[-1]+dt], ys+[ynp1], dts+[dt]
        if actions_after_timestep is not None:
            uts, uys, udts = actions_after_timestep(uts, uys, udts)

        # Calculate the next value of dt if needed
        if time_adaptor is not None:
            try: dt = time_adaptor(uts, uys, target_error)

            # If the scaling factor is too small then don't store this
            # timestep, instead repeat it with the new step size.
            except FailedTimestepError, e:
                dt = e.new_dt
                continue

        # Update results storage (don't do this earlier in case the time step fails).
        ys.append(uys[-1])
        ts.append(uts[-1])
        dts.append(udts[-1])

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


def trapezoid_rule_residual(base_residual, t, dts, ys, ynp1):


    return base_residual()

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
    """

    From Prinja's thesis.
    """
    d3ydt_n = d3ys[-1]
    return (2.0/9) * d3ydt_n * dt**3

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

def midpoint_lte(dt, ys, dys, d2ys, d3ys):
    """


    """


def midpoint_ab_time_adaptor(ts, ys, target_error):
    """

    See notes: 7/2/2013 for the algebra on calculating the AB2
    solution. See "mathematica_adaptive_midpoint.m for algebra to get a
    local truncation error out of all this.
    """

    dt = ts[-1] - ts[-2]
    dtprev = ts[-2] - ts[-3]
    ynp1_MP = ys[-1]


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

    # # Ignore d2y/dt2 (conservative estimate?).
    # ydotdotn = 0.0

    # Finite difference dy2/dt2 from neighbouring dy values. Not sure on
    # the accuracy here...
    ydotdotn = (ydot_nphalf - ydot_nmhalf) / (dt/2 + dtprev/2)

    # Get the truncation error
    # ============================================================

    error = (-6.0 * (ynp1_AB2 - ynp1_MP) * ydotn + 5.0 * dt**3 * ydotdotn**2 ) \
        / (6.0 * ( -10.0 + ydotn))

    # Compute + use scaling factor
    # ============================================================

    # Get a norm if we have an array, otherwise just take the absolute
    # value
    try: error_norm = sp.linalg.norm(error, 2)
    except ValueError: error_norm = abs(error)

    scaling_factor = ((target_error / error_norm)**0.5)
    #??ds what should the power here be?

    # 0.5 works much better than 1/3, but Milan says it should always be
    # 1/3 because it's a second order method...

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
    ts = list_cummulative_sums(dts[:-1], 0.)
    ys = map(exp, ts)
    ys = map(lambda x: sp.array(exp(x), ndmin=1), ts)
    ys, ts, dts = _odeint(residual, ys, ts, dts, tmax,
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
        ys, ts, dts = odeint(residual, [cos(0.0), exp(0.0)], tmax, dt = 0.001,
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
        ys, ts, dts = odeint(residual, [exp(0.0)], tmax, dt = 1e-6,
                             method = method, target_error = tol)

        # plt.plot(ts,ys,'--', ts, map(exp, ts))
        # dts = utils.dts_from_ts(ts)
        # plt.figure()
        # plt.plot(ts[:-1],dts)
        # plt.show()

        print "nsteps =", len(ys)

        utils.assertAlmostEqual(ys[-1][0], exp(tmax), requested_tol * 5)
        utils.assertAlmostEqual(len(ys), steps, steps * 0.1)

    methods = [('bdf2 ab', 1e-3, 382),
               ('midpoint ab', 1e-3, 105),
               ]
    for meth, requested_tol, allowed_steps in methods:
        yield check_adaptive_dt, meth, requested_tol, allowed_steps

def test_local_truncation_error():
    """Test that a single timestep has local truncation error below that
    given by the appropriate function.
    """

    tests = [ ##(midpoint_residual, midpoint_lte),
    #(trapezoid_rule_residual, trapezoid_rule_lte),
    (bdf2_residual, bdf2_lte),
    #(bdf1_residual, bdf1_lte),
    ]

    # Auxilary function
    def check_local_truncation_error(method_residual, error_function):

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
            dts = [1e-6]
            tmax = tstart + dt

            ys, ts, dts = _odeint(residual, ys, ts, dts+[dt],
                                 tmax, method_residual)

            lte_analytic = error_function(dt, map(value, ts), map(d1y, ts),
                                          map(d2y,ts), map(d3y,ts))

            # Check
            actual_error = ys[-1] - value(tmax)
            print "actual_error =", actual_error
            print "calculated error =", lte_analytic
            print

            #assert(actual_error < lte_analytic)
            utils.assertAlmostEqual(actual_error, lte_analytic, lte_analytic*10)
            # ??ds Not sure exactly how close estimate will be...

    for meth_res, tol_func in tests:
        yield check_local_truncation_error, meth_res, tol_func
