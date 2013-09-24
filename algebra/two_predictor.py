
from __future__ import division
from __future__ import absolute_import

import sympy
import scipy.misc
import sys
import itertools as it
import math

from sympy import Rational as sRat
from sympy.simplify.cse_main import cse


from pprint import pprint as pp
from operator import mul
from functools import partial as par
# import sympy.simplify.simplify as simpl

import simpleode.core.utils as utils
import simpleode.core.ode as ode


# Error calculations
# ============================================================
def imr_lte(dtn, dddynph, Fddynph):
    """From my derivations
    """
    return dtn**3*dddynph/24 + dtn*imr_f_approximation_error(dtn, Fddynph)


def bdf2_lte(dtn, dtnm1, dddyn):
    """Gresho and Sani pg.715, sign inverted
    """
    return -((dtn + dtnm1)**2/(dtn*(2*dtn + dtnm1))) * dtn**3 * dddyn/6


def bdf3_lte(*_):
    """Gresho and Sani pg.715, sign inverted
    """
    return 0 # No error to O(dtn**4) (third order)


def ebdf2_lte(dtn, dtnm1, dddyn):
    """Gresho and Sani pg.715, sign inverted, dtn/dtnm1 corrected as in Prinja
    """
    return (1 + dtnm1/dtn) * dtn**3 * dddyn / 6


def ebdf3_lte(*_):
    return 0 # no error to O(dtn**4) (third order)


def ab2_lte(dtn, dtnm1, dddyn):
    return (2 + 3*dtnm1/dtn) * dtn**3 * dddyn/12


def tr_lte(dtn, dddyn):
    return -dtn**3 * dddyn/12


def imr_f_approximation_error(dtn, Fddynph):
    """My derivations:
    y'_imr = y'(t_{n+1/2}) + dtn**2 * Fddynph/8 + O(dtn**3)

    =>  y'(t_{n+1/2}) - y'_imr = -dtn**2 * Fddynph/8 + O(dtn**3)

    """
    return -dtn**2*Fddynph/8


# Helper functions
# ============================================================
def constify_step(expr):
    return expr.subs([(dts[1], dts[0]), (dts[2], dts[0]), (dts[3], dts[0])])


def cse_print(expr):
    cses, simple_expr = cse(expr)
    print
    print sympy.pretty(simple_expr)
    pp(cses)
    print


def system2matrix(system, variables):
    """Create a matrix from a system of equations (assuming it's linear!).
    """
    A = sympy.Matrix([[None]*len(system)]*len(variables))
    for i, eqn in enumerate(system):
        for j, var in enumerate(variables):
            # Create a dict where all vars except this one are zero, this
            # one is one.
            subs_dict = dict([(v, 1 if v == var else 0) for v in variables])
            A[i, j] = eqn.subs(subs_dict)

    return A


def is_rational(x):
    try:
        # If it's a sympy object this is all we need
        return x.is_rational
    except AttributeError:
        # Otherwise only integer-like objects are rational
        return int(x) == x


def rational_as_mixed(x):
    """Convert a rational number to (integer_part, remainder_as_fraction)
    """
    assert is_rational(x) # Don't want to accidentally end up with floats
                          # in here as very long rationals!
    x_int = int(x)
    return x_int, sRat(x - x_int)


def sum_dts(a, b):
    """Get t_a - t_b in terms of dts.

    e.g.
    a = 0, b = 2:
    t_0 - t_2 = t_{n+1} - t_{n-1} = dt_{n+1} + dt_n = dts[0] + dts[1]

    a = 2, b = 0:
    t_2 - t_0 = - dts[0] - dts[1]
    """

    # Doesn't work for negative a, b
    assert a >= 0 and b >= 0

    # if a and b are in the "wrong" order then it's just the negative of
    # the sum with them in the "right" order.
    if a > b:
        return -1 * sum_dts(b, a)

    # Deal with non-integer a
    a_int, a_frac = rational_as_mixed(a)
    if a_frac != 0:
        result = -a_frac * dts[a_int] + sum_dts(a_int, b)
        return result

    b_int, b_frac = rational_as_mixed(b)
    if b_frac != 0:
        return b_frac * dts[b_int] + sum_dts(a, b_int)

    return sum(dts[a:b])


def bdf2_step_to_midpoint(ts, ys, dyn_func):

    dtn = (ts[-1] - ts[-2])/2
    dtnm1 = ts[-2] - ts[-3]

    dynp1 = dyn_func(ts, ys)
    yn = ys[-2]
    ynm1 = ys[-3]

    return ode.ibdf2_step(dtn, yn, dynp1, dtnm1, ynm1)


def bdf3_step_to_midpoint(ts, ys, dyn_func):

    dtn = (ts[-1] - ts[-2])/2
    dtnm1 = ts[-2] - ts[-3]
    dtnm2 = ts[-3] - ts[-4]

    dynp1 = dyn_func(ts, ys)
    yn = ys[-2]
    ynm1 = ys[-3]
    ynm2 = ys[-4]

    return ode.ibdf3_step(dynp1, dtn, yn, dtnm1, ynm1, dtnm2, ynm2)


# Define symbol names
# ============================================================
dts = sympy.symbols('Delta0:9', Real=True)
dddynph = sympy.symbols("y'''_h", Real=True)
Fddynph = sympy.symbols("F.y''_h", Real=True)
y_np1_exact = sympy.symbols('y_0', Real=True)
y_np1_imr, y_np1_p2, y_np1_p1 = sympy.symbols('y_0_imr y_0_p2 y_0_p1',
                                              Real=True)


# Calculate full errors
# ============================================================
def generate_p_dt_func(n_hist, symbolic_func):
    """Helper function: convert a symbolic function for predictor dt into a
    python function f(ts) = predictor_dt.
    """
    f = sympy.lambdify(dts[:n_hist-1], symbolic_func)
    def p_dt(ts):
        # Get last n_hist-1 dts in order: n, nm1, nm2 etc.
        dts = utils.ts2dts(ts[-n_hist:])[::-1]
        print dts
        return f(*dts)
    return p_dt


def use_exact(n):
    def f(ts, ys):
        return ys[n]
    return f

    time_points, p_name = scheme

    p_dtn = sum_dts(time_points[0], time_points[1])
    p_dtnm1 = sum_dts(time_points[1], time_points[2])

    # To cancel errors we need the final step to be at t_np1
    assert time_points[0] == 0

    # The whole point of this stuff is to resuse y'_nph from midpoint so
    # the one before final step must be at t_nph. Actually might want to
    # change this later...
    assert time_points[1] == sRat(1,2)


    # Construct errors and func for dyn estimate
    # ============================================================

    if dyn_estimate == "imr":
        dyn_error = imr_f_approximation_error(dts[0], Fddynph)

        dynm1_error = imr_f_approximation_error(dts[1], Fddynph)
        # + higher order terms due to expanding F, ddy from nmh to nph,
        # luckily for us these end up in O(dtn**4).

        dyn_func = ode.imr_dydt


    else:
        raise ValueError("Unrecognised dyn_estimate name " + dyn_estimate)


    # Construct errors and func for yn estimate
    # ============================================================

    if yn_estimate == "bdf2":
        # Error on y_nph as calculated by BDF2 using imr approximation for
        # dy_nph, tnp1 = tnph, tn = tn, tnm1 = tnm1
        yn_error = (
            # Natural bdf2 lte:
            bdf2_lte(dts[0]*sRat(1,2), dts[1], dddynph)
            # Error due to imr approximation to derivative:
            + ode.ibdf2_step(dts[0]*sRat(1,2), 0,  dyn_error, dts[1], 0))

        # Function to estimate ynph
        yn_func = bdf2_step_to_midpoint

    elif yn_estimate == "bdf3":
        # Error on y_nph as calculated by BDF2 using imr approximation for
        # dy_nph, tnp1 = tnph, tn = tn, tnm1 = tnm1, tnm2 = tnm2
        yn_error = (
            # Natural bdf2 lte:
            0
            # Error due to imr approximation to derivative:
            + ode.ibdf3_step(dyn_error, dts[0]*sRat(1,2), 0,  dts[1], 0, dts[2], 0))

        # Function to estimate ynph
        yn_func = bdf3_step_to_midpoint

    else:
        raise ValueError("Unrecognised yn_estimate name " + yn_estimate)


    # Construct errors and function for the predictor
    # ============================================================

    if p_name == "ebdf2" or p_name == "wrong step ebdf2":

        if p_name == "wrong step ebdf2":
            temp_p_dtnm1 = p_dtnm1 + dts[0]
        else:
            temp_p_dtnm1 = p_dtnm1

        y_np1_p_expr = y_np1_exact - (
            # Natural ebdf2 lte:
            ebdf2_lte(p_dtn, temp_p_dtnm1, dddynph)
            # error due to imr approximation to derivative:
            + ode.ebdf2_step(p_dtn, 0, dyn_error, p_dtnm1, 0)
            # error due to bdf2 approximation to midpoint: y
            + ode.ebdf2_step(p_dtn, yn_error, 0, p_dtnm1, 0))

        def predictor_func(ts, ys):
            dtn = (ts[-1] - ts[-2])/2
            dtnm1 = dtn + ts[-2] - ts[-(time_points[2]+1)]

            ynm1 = ys[-(time_points[2]+1)]
            dyn = dyn_func(ts, ys)
            yn = yn_func(ts, ys)
            return ode.ebdf2_step(dtn, yn, dyn, dtnm1, ynm1)


    elif p_name == "ab2":

        if dyn_estimate == "imr":
            assert time_points[2] == sRat(3,2)

        # Use the same estimate for dyn and dynm1 (except at different
        # points)
        y_np1_p_expr = y_np1_exact - (
            # Natural ab2 lte:
            ab2_lte(p_dtn, p_dtnm1, dddynph)
            # error due to approximation to derivative at tn
            + ode.ab2_step(p_dtn, 0, dyn_error, p_dtnm1, 0)
            # error due to approximation to derivative at tnm1
            + ode.ab2_step(p_dtn, 0, 0, p_dtnm1, dynm1_error)
            # error due to approximation to midpoint: y
            + ode.ab2_step(p_dtn, yn_error, 0, p_dtnm1, 0))


        def predictor_func(ts, ys):
            dtn = (ts[-1] - ts[-2])/2
            assert time_points[2] == sRat(3,2)
            dtnm1 = (ts[-1] - ts[-2])/2 + (ts[-2] - ts[-3])/2

            dyn = dyn_func(ts, ys)
            dynm1 = dyn_func(ts[:-1], ys[:-1])
            yn = yn_func(ts, ys)
            return ode.ab2_step(dtn, yn, dyn, dtnm1, dynm1)

    # elif p_name == "ebdf3":

    # p_dtnm2 = sum_dts(time_points[2], time_points[3])

    #     y_np1_p_expr = y_np1_exact - (
    #         # Natural ebdf3 lte (=0)
    #         ebdf3_lte(p_dtn, p_dtnm1, p_dtnm2, dddynph)
    #         # Error due to approximation to derivative at tn
    #         + ode.ebdf3_step(p_dtn, 0, dyn_error, p_dtnm1, 0, p_dtnm2, 0)

    else:
        raise ValueError("Unrecognised predictor name " + p_name)

    return predictor_func, y_np1_p_expr


def generate_predictor_pair_scheme(p1_scheme, p2_scheme, yn_estimate,
                                  dyn_estimate):
    """Generate two-predictor lte system of equations and predictor step
    functions.
    """

    # Generate the two schemes
    p1_func, y_np1_p1_expr = generate_predictor_scheme(p1_scheme, yn_estimate,
                                                       dyn_estimate)
    p2_func, y_np1_p2_expr = generate_predictor_scheme(p2_scheme, yn_estimate,
                                                       dyn_estimate)

    # LTE for IMR: just natural lte:
    y_np1_imr_expr = y_np1_exact - imr_lte(dts[0], dddynph, Fddynph)

    # Return error expressions and stepper functions
    return (y_np1_p1_expr, y_np1_p2_expr, y_np1_imr_expr), (p1_func, p2_func)


def generate_predictor_pair_lte_est(p1, p2, ynph_approximation,
                                    dynph_approximation):

    lte_equations, (p1_func, p2_func) \
      = generate_predictor_pair_scheme(p1, p2, ynph_approximation,
                                       dynph_approximation)

    A = system2matrix(lte_equations, [dddynph, Fddynph, y_np1_exact])
    x = A.inv()

    # Look at some things for the constant step case:
    cse_print(constify_step(A))
    cse_print(constify_step(x))
    print sympy.pretty(constify_step(A).det())

    # We can get nice expressions by factorising things (row 2 dotted with
    # [predictor values] gives us y_np1_exact):
    exact_ynp1_symb = sum([y_est * xi.factor() for xi, y_est in
                           zip(x.row(2), [y_np1_p1, y_np1_p2, y_np1_imr])])

    exact_ynp1_func = sympy.lambdify((dts[0], dts[1], dts[2], dts[3],
                                      y_np1_p1, y_np1_p2, y_np1_imr),
                                      exact_ynp1_symb)


    # Debugging:
    dddy_symb = sum([y_est * xi.factor() for xi, y_est in
                    zip(x.row(0), [y_np1_p1, y_np1_p2, y_np1_imr])])
    Fddy_symb = sum([y_est * xi.factor() for xi, y_est in
                    zip(x.row(1), [y_np1_p1, y_np1_p2, y_np1_imr])])
    dddy_func = sympy.lambdify((dts[0], dts[1], dts[2], dts[3],
                                      y_np1_p1, y_np1_p2, y_np1_imr),
                                      dddy_symb)
    Fddy_func = sympy.lambdify((dts[0], dts[1], dts[2], dts[3],
                                      y_np1_p1, y_np1_p2, y_np1_imr),
                                      Fddy_symb)

    print sympy.pretty(constify_step(dddy_symb))
    print sympy.pretty(constify_step(Fddy_symb))
    print sympy.pretty(constify_step(exact_ynp1_symb))

    def lte_est(ts, ys):

        # Compute predicted values
        y_np1_p1 = p1_func(ts, ys)
        y_np1_p2 = p2_func(ts, ys)

        y_np1_imr = ys[-1]
        dtn = ts[-1] - ts[-2]
        dtnm1 = ts[-2] - ts[-3]
        dtnm2 = ts[-3] - ts[-4]
        dtnm3 = ts[-4] - ts[-5]

        # Calculate the exact value (to O(dtn**4))
        y_np1_exact = exact_ynp1_func(dtn, dtnm1, dtnm2, dtnm3,
                                      y_np1_p1, y_np1_p2, y_np1_imr)


        dddy_est = dddy_func(dtn, dtnm1, dtnm2, dtnm3, y_np1_p1, y_np1_p2, y_np1_imr)
        Fddy_est = Fddy_func(dtn, dtnm1, dtnm2, dtnm3, y_np1_p1, y_np1_p2, y_np1_imr)
        print
        print "%0.16f"%y_np1_imr, "%0.16f"%y_np1_p1, "%0.16f"%y_np1_p2
        print "abs(y_np1_imr - y_np1_p1) =", abs(y_np1_imr - y_np1_p1)
        print "abs(y_np1_p2 - y_np1_imr) =", abs(y_np1_p2 - y_np1_imr)
        print "dddy_est =", dddy_est
        print "Fddy_est =", Fddy_est
        print "y_np1_exact - y_np1_imr =", y_np1_exact - y_np1_imr
        print "-dtn**3 * dddy_est/24 =", -dtn**3 * dddy_est/24

        # Compare with IMR value to get truncation error
        return y_np1_exact - y_np1_imr
        # ??ds

        # return dtn**3


    return lte_est


# import simpleode.core.example_residuals as er
# import scipy as sp
# from matplotlib.pyplot import show as pltshow
# from matplotlib.pyplot import subplots

# def main():

#     # Function to integrate

#     # residual = er.exp_residual
#     # exact = er.exp_exact

#     residual = par(er.damped_oscillation_residual, 1, 0)
#     exact = par(er.damped_oscillation_exact, 1, 0)

#     # method
#     lte_est = generate_predictor_pair((0, sRat(1,2), 2, "ebdf2"),
#                                       (0, sRat(1,2), sRat(3,2), "ab2"),
#                                       "bdf3",
#                                       "midpoint")
#     my_adaptor = par(ode.general_time_adaptor, lte_calculator=lte_est,
#                       method_order=2)
#     init_actions = par(ode.higher_order_start, 6)

#     # Do it
#     ts, ys = ode._odeint(residual, [sp.array(exact(0.0), ndmin=1)], [0.0],
#                          1e-4, 5.0, ode.midpoint_residual,
#                          1e-6, my_adaptor, init_actions)

#     # Plot

#     # Get errors + exact solution
#     exacts = map(exact, ts)
#     errors = [sp.linalg.norm(y - ex, 2) for y, ex in zip(ys, exacts)]


#     fig, axes = subplots(4, 1, sharex=True)
#     dt_axis=axes[1]
#     result_axis=axes[0]
#     exact_axis=axes[2]
#     error_axis=axes[3]
#     method_name = "w18 lte est imr"
#     if exact_axis is not None:
#         exact_axis.plot(ts, exacts, label=method_name)
#         exact_axis.set_xlabel('$t$')
#         exact_axis.set_ylabel('$y(t)$')

#     if error_axis is not None:
#         error_axis.plot(ts, errors, label=method_name)
#         error_axis.set_xlabel('$t$')
#         error_axis.set_ylabel('$||y(t) - y_n||_2$')

#     if dt_axis is not None:
#         dt_axis.plot(ts[1:], utils.ts2dts(ts),
#                      label=method_name)
#         dt_axis.set_xlabel('$t$')
#         dt_axis.set_ylabel('$\Delta_n$')


#     if result_axis is not None:
#         result_axis.plot(ts, ys, label=method_name)
#         result_axis.set_xlabel('$t$')
#         result_axis.set_ylabel('$y_n$')

#     pltshow()

# if __name__ == '__main__':
#     sys.exit(main())


# Tests
# ============================================================

import simpleode.core.example_residuals as er
import scipy as sp
import operator as op

# def check_dddy_estimates(exact_symb):
#     dt = 5e-2

#     # Derive the required functions/derivatives:
#     exact = sympy.lambdify(sympy.symbols('t'), exact_symb)

#     dy_symb = sympy.diff(exact_symb, sympy.symbols('t'), 1).subs(exact_symb, sympy.symbols('y'))
#     residual_symb = sympy.symbols('Dy') - dy_symb
#     residual = sympy.lambdify((sympy.symbols('t'), sympy.symbols('y'), sympy.symbols('Dy')),
#                               residual_symb)

#     dfdy_symb = sympy.diff(dy_symb, sympy.symbols('y'))
#     ddy_symb = sympy.diff(exact_symb, sympy.symbols('t'), 2)
#     Fdoty = sympy.lambdify((sympy.symbols('t'), sympy.symbols('y')),
#                            dfdy_symb * ddy_symb)
#     exact_dddy_symb = sympy.diff(exact_symb, sympy.symbols('t'), 3)
#     exact_dddy = sympy.lambdify(sympy.symbols('t'), exact_dddy_symb)

#     print dfdy_symb, ddy_symb


#     # Solve with imr
#     ts, est_ys = ode.odeint(residual, exact(0.0), dt=dt,
#                             tmax=3.0, method='imr',
#                             newton_tol=1e-10, jacobian_fd_eps=1e-12)

#     # Construct predictors
#     p1_steps = (0, sRat(1,2), 3)
#     p2_steps = (0, sRat(1,2), 4)
#     lte_equations, (p1_func, p2_func) = general_two_predictor(p1_steps, p2_steps)

#     # Compare estimates of values with actual values
#     n = 5
#     for par_ts, par_ys in zip(utils.partial_lists(ts, n), utils.partial_lists(est_ys, n)):
#         y_np1_p1 = p1_func(par_ts, par_ys)
#         y_np1_p2 = p2_func(par_ts, par_ys)

#         dtn = par_ts[-1] - par_ts[-2]
#         dtnm1 = par_ts[-2] - par_ts[-3]
#         y_np1_imr = par_ys[-1]

#         # dddy_est = t17_dddy_est(dtn, dtnm1, y_np1_p1, y_np1_p2, y_np1_imr)

#         print dtnm1, dtn
#         print "%0.16f" % y_np1_imr, "%0.16f" % y_np1_p1, "%0.16f" % y_np1_p2
#         print
#         print abs(y_np1_imr - y_np1_p2)
#         assert abs(y_np1_imr - y_np1_p2) > 1e-8
#         assert abs(y_np1_imr - y_np1_p1) > 1e-8

#         # Fddy_est = t17_Fddy_est(dtn, dtnm1, y_np1_p1, y_np1_p2, y_np1_imr)

#         dddy = exact_dddy(par_ts[-1])
#         Fddy = Fdoty(par_ts[-1], par_ys[-1])

#         # utils.assert_almost_equal(dddy_est, , min(1e-6, 30* dt**4))
#         # utils.assert_almost_equal(Fddy_est, Fddy, min(1e-6, 30* dt**4))


#     # check we actually did something!
#     assert utils.partial_lists(ts, n) != []


# def test_dddy_estimates():

#         #
#     t = sympy.symbols('t')

#     equations = [
#         3*t**3,
#         sympy.tanh(t),
#         sympy.exp(-t)
#         ]


#     for exact_symb in equations:
#          yield check_dddy_estimates, exact_symb


def test_generate_predictor_scheme_a_bit():

    # Check the we generate imr's lte if we plug in the right times and
    # approximations to ebdf2 (~explicit midpoint rule).
    _, p_lte = generate_predictor_scheme(([0, sRat(1,2), 1], "ebdf2"),
                                              "bdf2", "imr")

    _, p2_lte = generate_predictor_scheme(([0, sRat(1,2), 1], "ebdf2"),
                                              "bdf3", "imr")

    utils.assert_sym_eq(y_np1_exact - imr_lte(dts[0], dddynph, Fddynph), p_lte)
    utils.assert_sym_eq(y_np1_exact - imr_lte(dts[0], dddynph, Fddynph), p2_lte)



def test_sum_dts():

    # Check a simple fractional case
    utils.assert_sym_eq(sum_dts(sRat(1,2), 1), dts[0]/2)

    # Check two numbers the same gives zero always
    utils.assert_sym_eq(sum_dts(1, 1), 0)
    utils.assert_sym_eq(sum_dts(0, 0), 0)
    utils.assert_sym_eq(sum_dts(sRat(1,2), sRat(1,2)), 0)


    def check_sum_dts(a, b):

        # Check we can swap the sign
        utils.assert_sym_eq(sum_dts(a, b), -sum_dts(b, a))

        # Starting half a step earlier
        utils.assert_sym_eq(sum_dts(a, b + sRat(1,2)),
                            sRat(1,2)*dts[int(b)] + sum_dts(a, b))

        # Check that we can split it up
        utils.assert_sym_eq(sum_dts(a, b),
                            sum_dts(a, b-1) + sum_dts(b-1, b))

    # Make sure b>=1 or the last check will fail due to negative b.
    cases = [(0, 1),
             (5, 8),
             (sRat(1,2), 3),
             (sRat(2,2), 3),
             (sRat(3,2), 3),
             (0, sRat(9,7)),
             (sRat(3/4), sRat(9,7)),
             ]

    for a, b in cases:
        yield check_sum_dts, a, b


def test_ltes():

    import numpy
    numpy.seterr(all='raise', divide='raise', over=None, under=None, invalid=None)

    def check_lte(method_residual, lte, exact_symb, base_dt, implicit):

        exact, residual, dys, J = utils.symb2functions(exact_symb)
        dddy = dys[3]
        Fddy = lambda t, y: J(t, y) * dys[2](t, y)

        newton_tol = 1e-10

        # tmax varies with dt so that we can vary dt over orders of
        # magnitude.
        tmax = 50*dt

        # Run ode solver
        if implicit:
            ts, ys = ode._odeint(
                residual, [0.0], [sp.array([exact(0.0)], ndmin=1)],
                dt, tmax, method_residual,
                target_error=None,
                time_adaptor=ode.create_random_time_adaptor(base_dt),
                initialisation_actions=par(ode.higher_order_start, 3),
                newton_tol=newton_tol)

        else:
            ts, ys = ode.odeint_explicit(
                dys[1], exact(0.0), base_dt, tmax, method_residual,
                time_adaptor=ode.create_random_time_adaptor(base_dt))

        dts = utils.ts2dts(ts)

        # Check it's accurate-ish
        exact_ys = map(exact, ts)
        errors = map(op.sub, exact_ys, ys)
        utils.assert_list_almost_zero(errors, 1e-3)

        # Calculate ltes by two methods. Note that we need to drop a few
        # values because exact calculation (may) need a few dts. Could be
        # dodgy: exact dddys might not correspond to dddy in experiment if
        # done over long time and we've wandered away from the solution.
        exact_dddys = map(dddy, ts, ys)
        exact_Fddys = map(Fddy, ts, ys)
        exact_ltes = map(lte, dts[2:], dts[1:-1], dts[:-2],
                         exact_dddys[3:], exact_Fddys[3:])
        error_diff_ltes = map(op.sub, errors[1:], errors[:-1])[2:]

        # Print for debugging when something goes wrong
        print exact_ltes
        print error_diff_ltes

        # Probably the best test is that they give the same order of
        # magnitude and the same sign... Can't test much more than that
        # because we have no idea what the constant in front of the dt**4
        # term is. Effective zero (noise level) is either dt**4 or newton
        # tol, whichever is larger.
        z = 50 * max(dt**4, newton_tol)
        map(par(utils.assert_same_sign, fp_zero=z),
            exact_ltes, error_diff_ltes)
        map(par(utils.assert_same_order_of_magnitude, fp_zero=z),
            exact_ltes, error_diff_ltes)

        # For checking imr in more detail on J!=0 cases
        # if method_residual is ode.imr_residual:
        #     if J(1,2) != 0:
        #         assert False


    t = sympy.symbols('t')
    functions = [2*t**2,
                 t**3 + 3*t**4,
                 sympy.exp(t),
                 3*sympy.exp(-t),
                 sympy.sin(t),
                 sympy.sin(t)**2 + sympy.cos(t)**2
                 ]

    methods = [
        (ode.imr_residual, True,
         lambda dtn, _, _1, dddyn, Fddy: imr_lte(dtn, dddyn, Fddy)
         ),

        (ode.bdf2_residual, True,
         lambda dtn, dtnm1, _, dddyn, _1: bdf2_lte(dtn, dtnm1, dddyn)
         ),

        (ode.bdf3_residual, True,
         lambda dtn, dtnm1, dtnm2, dddyn, _: bdf3_lte(dtn, dtnm1, dtnm2, dddyn)
         ),

        ('ab2', False,
         lambda dtn, dtnm1, _, dddyn, _1: ab2_lte(dtn, dtnm1, dddyn)
         ),

         # ('ebdf2', False,
         # lambda dtn, dtnm1, _, dddyn, _1: ebdf2_lte(dtn, dtnm1, dddyn)
         # ),

         # ('ebdf3', False,
         # lambda *_: ebdf3_lte()
         # ),

        ]

    # Seems to work from 1e-2 down until newton method stops converging due
    # to FD'ed Jacobian. Just do a middling value so that we can wobble the
    # step size around lots without problems.
    dts = [1e-3]

    for exact_symb in functions:
        for method_residual, implicit, lte in methods:
            for dt in dts:
                yield check_lte, method_residual, lte, exact_symb, dt, implicit


def test_tr_ab2_scheme_generation():
    """Make sure tr-ab can be derived using same methodology as I'm using
    (also checks lte expressions).
    """

    dddy = sympy.symbols("y'''")
    y_np1_tr = sympy.symbols("y_{n+1}_tr")

    y_np1_ab2_expr = y_np1_exact - ab2_lte(dts[0], dts[1], dddy)
    y_np1_tr_expr = y_np1_exact - tr_lte(dts[0], dddy)

    A = system2matrix([y_np1_ab2_expr, y_np1_tr_expr], [dddy, y_np1_exact])
    x = A.inv()

    exact_ynp1_symb = sum([y_est * xi.factor() for xi, y_est in
                        zip(x.row(1), [y_np1_p1, y_np1_tr])])

    utils.assert_sym_eq(exact_ynp1_symb - y_np1_tr,
                        (y_np1_p1 - y_np1_tr)/(3*(1 + dts[1]/dts[0])))


def test_bdf2_ebdf2_scheme_generation():
    """Make sure adaptive bdf2 can be derived using same methodology as I'm
    using (also checks lte expressions).
    """

    dddy = sympy.symbols("y'''")
    y_np1_bdf2 = sympy.symbols("y_{n+1}_bdf2")

    y_np1_ebdf2_expr = y_np1_exact - ebdf2_lte(dts[0], dts[1], dddy)
    y_np1_bdf2_expr = y_np1_exact - bdf2_lte(dts[0], dts[1], dddy)

    A = system2matrix([y_np1_ebdf2_expr, y_np1_bdf2_expr], [dddy, y_np1_exact])
    x = A.inv()

    exact_ynp1_symb = sum([y_est * xi.factor() for xi, y_est in
                        zip(x.row(1), [y_np1_p1, y_np1_bdf2])])

    answer = -(dts[1] + dts[0])*(y_np1_bdf2 - y_np1_p1)/(3*dts[0] +2*dts[1])

    utils.assert_sym_eq(exact_ynp1_symb - y_np1_bdf2,
                        answer)


def test_generate_predictor_dt_func():

    symbs = [dts[0], dts[1], dts[2], dts[0] + dts[1],
             dts[0]/2 + dts[1]/2]

    t = sympy.symbols('t')

    fake_ts = utils.dts2ts(t, dts[::-1])

    for symb in symbs:
        print fake_ts
        yield utils.assert_sym_eq, symb, generate_p_dt_func(3, symb)(fake_ts)

#
