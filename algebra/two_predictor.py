
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
    return dtn**3*dddynph/24 - dtn**3*Fddynph/8


def bdf2_lte(dtn, dtnm1, dddyn):
    """Gresho and Sani pg.715, sign inverted
    """
    return -((dtn + dtnm1)**2/(dtn*(2*dtn + dtnm1))) * dtn**3 * dddyn/6


def bdf3_lte(dtn, dtnm1, dtnm2, dddyn):
    """Gresho and Sani pg.715, sign inverted
    """
    return 0 # No error to O(dtn**4) (third order)


def ebdf2_lte(dtn, dtnm1, dddyn):
    """Gresho and Sani pg.715, sign inverted, dtn/dtnm1 corrected as in Prinja
    """
    return (1 + dtnm1/dtn) * dtn**3 * dddyn / 6


def ebdf3_lte(dtn, dtnm1, dtnm2, dddyn):
    return 0 # no error to O(dtn**4) (third order)


def ab2_lte(dtn, dtnm1, dddyn):
    return (2 + 3*dtnm1/dtn) * dtn**3 * dddyn/12


def midpoint_f_approximation_error(dtn, Fddynph):
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
        return -a_frac * dts[a_int] + sum_dts(a_int, b)

    b_int, b_frac = rational_as_mixed(b)
    if b_frac != 0:
        return b_frac * dts[b_int] + sum_dts(a, b_int)

    return sum(dts[a:b])


def bdf2_step_to_midpoint(ts, ys):

    dtn = (ts[-1] - ts[-2])/2
    dtnm1 = ts[-2] - ts[-3]

    dynp1 = ode.midpoint_dydt(ts, ys)
    yn = ys[-2]
    ynm1 = ys[-3]

    return ode.ibdf2_step(dtn, yn, dynp1, dtnm1, ynm1)


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
def generate_predictor_scheme(scheme, yn_estimate, dyn_estimate):

    p_t0, p_t1, p_t2, p_name = scheme

    p_dtn = sum_dts(p_t0, p_t1)
    p_dtnm1 = sum_dts(p_t1, p_t2)

    # Construct errors and func for dyn estimate
    # ============================================================

    if dyn_estimate == "midpoint":
        assert p_t0 == 0
        assert p_t1 == sRat(1,2)
        dyn_error = midpoint_f_approximation_error(dts[0], Fddynph)
        dynm1_error = midpoint_f_approximation_error(dts[1], Fddynph)
        # + higher order terms due to expanding F, ddy from nmh to nph,
        # luckily for us these end up in O(dtn**4).

        dyn_func = ode.midpoint_dydt

    else:
        raise ValueError("Unrecognised dyn_estimate name " + dyn_estimate)


    # Construct errors and func for yn estimate
    # ============================================================

    if yn_estimate == "bdf2":
        # Error on y_nph as calculated by BDF2 using midpoint approximation for
        # dy_nph, tnp1 = tnph, tn = tn, tnm1 = tnm1
        yn_error = (
            # Natural bdf2 lte:
            bdf2_lte(dts[0]*sRat(1,2), dts[1], dddynph)
            # Error due to imr approximation to derivative:
            + ode.ibdf2_step(dts[0]*sRat(1,2), 0,  dyn_error, dts[1], 0))

        # Function to estimate ynph
        yn_func = bdf2_step_to_midpoint

    elif yn_estimate == "bdf3":
        # Error on y_nph as calculated by BDF2 using midpoint approximation for
        # dy_nph, tnp1 = tnph, tn = tn, tnm1 = tnm1, tnm2 = tnm2
        yn_error = (
            # Natural bdf2 lte:
            0
            # Error due to imr approximation to derivative:
            + ode.ibdf3_step(dyn_error, dts[0]*sRat(1,2), 0,  dts[1], 0, dts[2], 0))

        # Function to estimate ynph
        yn_func = None
    else:
        raise ValueError("Unrecognised yn_estimate name " + yn_estimate)


    # Construct errors and function for the predictor
    # ============================================================

    if p_name == "ebdf2":
        y_np1_p_expr = y_np1_exact - (
            # Natural ebdf2 lte:
            ebdf2_lte(p_dtn, p_dtnm1, dddynph)
            # error due to imr approximation to derivative:
            + ode.ebdf2_step(p_dtn, 0, dyn_error, p_dtnm1, 0)
            # error due to bdf2 approximation to midpoint: y
            + ode.ebdf2_step(p_dtn, yn_error, 0, p_dtnm1, 0))

        def p_func():

            # Only works for tn = t{n+1/2}
            assert p_t0 == 0
            assert p_t1 == sRat(1,2)
            dtn = (ts[-1] - ts[-2])/2
            dtnm1 = dtn + ts[-2] - ts[-(p_t2+1)]
            #

            ynm1 = ys[-(p_t2+1)]
            dyn = dyn_func(ts, ys)
            yn = yn_func(ts, ys)
            return ode.ebdf2_step(dtn, yn, dyn, dtnm1, ynm1)


    elif p_name == "ab2":

        if dyn_estimate == "midpoint":
            assert p_t2 == sRat(3,2)

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


        def p_func():

            # Only works for tn = t{n+1/2}
            assert p_t0 == 0
            assert p_t1 == sRat(1,2)
            dtn = (ts[-1] - ts[-2])/2
            dtnm1 = dtn + ts[-2] - ts[-(p_t2+1)]
            #

            dyn = dyn_func(ts, ys)
            dynm1 = dyn_func(ts[:-1], ys[:-1])
            yn = yn_func(ts, ys)
            return ode.ab2_step(dtn, yn, dyn, dtnm1, dynm1)

    else:
        raise ValueError("Unrecognised predictor name " + p_name)

    return p_func, y_np1_p_expr


def generate_two_predictor_scheme(p1_scheme, p2_scheme, yn_estimate="bdf2",
                                  dyn_estimate="midpoint"):
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


def main():

    p1 = (0, sRat(1,2), 2, "ebdf2")
    p2 = (0, sRat(1,2), sRat(3,2), "ab2")
    lte_equations, (p1_func, p2_func) \
      = generate_two_predictor_scheme(p1, p2, "bdf2", "midpoint")

    A = system2matrix(lte_equations, [dddynph, Fddynph, y_np1_exact])
    x = A.inv()

    # We can get nice expressions by factorising things (row 2 dotted with
    # [predictor values] gives us y_np1_exact, I think):
    print sympy.pretty([xi.factor() for xi in x.row(2)])

    # Look at some things for the constant step case:
    cse_print(constify_step(x))
    print sympy.pretty(constify_step(A).det())



if __name__ == '__main__':
    sys.exit(main())


# Tests
# ============================================================

import simpleode.core.example_residuals as er
import scipy as sp

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


#     # Solve with midpoint
#     est_ys, ts = ode.odeint(residual, exact(0.0), dt=dt,
#                             tmax=3.0, method='midpoint',
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


def test_dddy_estimates():

        #
    t = sympy.symbols('t')

    equations = [
        3*t**3,
        sympy.tanh(t),
        sympy.exp(-t)
        ]


    for exact_symb in equations:
         yield check_dddy_estimates, exact_symb


def test_general_two_predictor_imr_a_bit():
    # Check the we generate imr's lte if we plug in the right times.
    p1_steps = (0, sRat(1,2), 1)
    p2_steps = (0, sRat(1,2), 1)
    lte_equations, (p1_func, p2_func) = general_two_predictor(p1_steps, p2_steps)

    utils.assert_sym_eq(lte_equations[0], lte_equations[2])
    utils.assert_sym_eq(lte_equations[1], lte_equations[2])


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



#
