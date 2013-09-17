
from __future__ import division
from __future__ import absolute_import

import sympy
import scipy.misc
import sys
import itertools as it

from sympy import Rational as sRat
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


def ebdf2_lte(dtn, dtnm1, dddyn):
    """Gresho and Sani pg.715, sign inverted, dtn/dtnm1 corrected as in Prinja
    """
    return (1 + dtnm1/dtn) * dtn**3 * dddyn / 6


def ebdf3_lte(dtn, dtnm1, dtnm2, dddyn):
    return 0 # no error to O(dtn**4) (third order)


def midpoint_f_approximation_error(dtn, Fddynph):
    return dtn**2*Fddynph/8


# Helper functions
# ============================================================
def full_pretty(expr):
    print sympy.pretty(sympy.collect(expr.expand(), [dddynph, Fddynph, y_np1_exact]))


def get_a(expr):
    """Get coeff of dddynph from expression.
    """
    return expr.subs([(y_np1_exact, 0), (dddynph,1), (Fddynph,0)])


def get_b(expr):
    """Get coeff of F.y''_h from expression.
    """
    return expr.subs([(y_np1_exact, 0), (dddynph,0), (Fddynph,1)])


def get_coeffs(expr):
    y_np1_imr, y_np1_p2, y_np1_p3 = sympy.symbols('y_0_mp y_0_p2 y_0_p3', Real=True)
    imr_coeff = expr.subs([(y_np1_imr, 1), (y_np1_p2,0), (y_np1_p3,0)])
    p2_coeff = expr.subs([(y_np1_imr,0), (y_np1_p2,1), (y_np1_p3,0)])
    p3_coeff = expr.subs([(y_np1_imr,0), (y_np1_p2,0), (y_np1_p3,1)])

    return imr_coeff, p2_coeff, p3_coeff


def constify_step(expr):
    return expr.subs([(dts[1], dts[0])])


# Define symbol names
# ============================================================
dts = sympy.symbols('Delta0:9', Real=True)
dddynph = sympy.symbols("y'''_h", Real=True)
Fddynph = sympy.symbols("F.y''_h", Real=True)
y_np1_exact = sympy.symbols('y_0', Real=True)
y_np1_imr, y_np1_p2, y_np1_p3 = sympy.symbols('y_0_mp y_0_p2 y_0_p3', Real=True)


# Calculate full errors
# ============================================================

def m16_scheme():
    """Generate dictionary of coeffs for the scheme.
    """

    y_nph_mp_err = midpoint_f_approximation_error(dts[0], Fddynph)

    # Error on y_nph as calculated by BDF2 using midpoint approximation for
    # dy_nph, tnp1 = tnph, tn = tn, tnm1 = tnm1
    y_nph_err = (
        # Natural bdf2 lte:
        bdf2_lte(dts[0]/2, dts[1], dddynph)
        # Error due to imr approximation to derivative:
        + ode.ibdf2_step(dts[0]/2, dts[1], 0, 0, y_nph_mp_err))


    # Error on y_np1_P2 (as calculated using y_nph from bdf2 and y'_nph
    # from imr in eBDF2 at time points tnm1 = tn, tn=tnph ,tnp1 = tnp1).
    y_np1_p2_expr = y_np1_exact - (
        # Natural ebdf2 lte:
        ebdf2_lte(dts[0]/2, dts[0]/2, dddynph)
        # error due to imr approximation to derivative:
        + ode.ebdf2_step(dts[0]/2, 0, y_nph_mp_err, dts[0]/2, 0)
        # error due to bdf2 approximation to midpoint:
        + ode.ebdf2_step(dts[0]/2, y_nph_err, 0, dts[0]/2, 0))

    # Error on y_np1_P3 (as calculated using y_nph from bdf2 and y'_nph
    # from imr in eBDF3 at time points tnm2 = tnm1, tnm1 = tn, tn = tnph, tnp1 = tnp1).
    y_np1_p3_expr = y_np1_exact - (
        # Natural ebdf3 lte:
        ebdf3_lte(dts[0]/2, dts[0]/2, dts[1], dddynph)
        # error due to imr approximation to derivative:
        + ode.ebdf3_step(dts[0]/2, 0, y_nph_mp_err, dts[0]/2, 0, dts[1], 0)
        # error due to bdf2 approximation to midpoint: y
        + ode.ebdf3_step(dts[0]/2, y_nph_err, 0, dts[0]/2, 0, dts[1], 0))

    y_np1_imr_expr = y_np1_exact - (
        # Just natural lte here:
        imr_lte(dts[0], dddynph, Fddynph))

    coeff_dict = dict(
        a_p2=get_a(y_np1_p2_expr),
        b_p2=get_b(y_np1_p2_expr),
        a_p3=get_a(y_np1_p3_expr),
        b_p3=get_b(y_np1_p3_expr),
        a_imr=get_a(y_np1_imr_expr),
        b_imr=get_b(y_np1_imr_expr))

    return coeff_dict


def t17_scheme():

    y_nph_mp_err = midpoint_f_approximation_error(dts[0], Fddynph)

    # Error on y_nph as calculated by BDF2 using midpoint approximation for
    # dy_nph, tnp1 = tnph, tn = tn, tnm1 = tnm1
    y_nph_err = (
        # Natural bdf2 lte:
        bdf2_lte(dts[0]/2, dts[1], dddynph)
        # Error due to imr approximation to derivative:
        + ode.ibdf2_step(dts[0]/2, dts[1], 0, 0, y_nph_mp_err))

    # Error on y_np1_P2 (as calculated using y_nph from bdf2 and y'_nph
    # from imr in eBDF2 at time points tnm1 = tn, tn=tnph ,tnp1 = tnp1).
    y_np1_p2_expr = y_np1_exact - (
        # Natural ebdf2 lte:
        ebdf2_lte(dts[0]/2, dts[0]/2, dddynph)
        # error due to imr approximation to derivative:
        + ode.ebdf2_step(dts[0]/2, 0, y_nph_mp_err, dts[0]/2, 0)
        # error due to bdf2 approximation to midpoint:
        + ode.ebdf2_step(dts[0]/2, y_nph_err, 0, dts[0]/2, 0))

    # Error on y_np1_P3 (as calculated using y_nph from bdf2 and y'_nph
    # from imr in eBDF2 at time points tnm1 = tm1, tn=tnph ,tnp1 = tnp1).
    y_np1_p3_expr = y_np1_exact - (
        # Natural ebdf2 lte:
        ebdf2_lte(dts[0]/2, dts[0]/2 + dts[1], dddynph)
        # error due to imr approximation to derivative:
        + ode.ebdf2_step(dts[0]/2 + dts[1], 0, y_nph_mp_err, dts[0]/2, 0)
        # error due to bdf2 approximation to midpoint:
        + ode.ebdf2_step(dts[0]/2 + dts[1], y_nph_err, 0, dts[0]/2, 0))

    y_np1_imr_expr = y_np1_exact - (
        # Just natural lte here:
        imr_lte(dts[0], dddynph, Fddynph))

    coeff_dict = dict(
        a_p2=get_a(y_np1_p2_expr),
        b_p2=get_b(y_np1_p2_expr),
        a_p3=get_a(y_np1_p3_expr),
        b_p3=get_b(y_np1_p3_expr),
        a_imr=get_a(y_np1_imr_expr),
        b_imr=get_b(y_np1_imr_expr))

    return coeff_dict


def full_method(error_calculator):

    # Do the calculation in two parts so that sympy can handle it easier:
    # 1) Calculate lte expressions for each integrator and break into a set
    # of coefficients for each term (y''', f.y'' and y_{n+1}) in each lte.
    # 2) Invert the matrix of (single symbol) coefficients, with some
    # coefficients filled in.
    # 3) Subs in the full coeff expressions into the now-inverted system.

    coeff_dict = error_calculator()

    # Create and invert matrix of coefficients of y''', f.y'' and y_{n+1}.
    a_p2, b_p2, a_p3, b_p3, a_imr, b_imr = sympy.symbols('a_p2, b_p2, a_p3, b_p3, a_imr, b_imr')
    A = sympy.Matrix([[a_p2, b_p2, 1],
                      [a_p3, b_p3, 1],
                      [a_imr, b_imr, 1]])
    b = sympy.Matrix([y_np1_p2, y_np1_p3, y_np1_imr])
    x = A.LUsolve(b)

    # substitue in the coeff values
    full_x = [xi.subs(coeff_dict) for xi in x]
    full_x_coeffs = [get_coeffs(xi) for xi in full_x]

    return full_x_coeffs


def main():


    full_x_coeffs = full_method(t17_scheme)

    # # Print with constant step sizes
    # coeffs_flat = list(it.chain(*full_x_coeffs))
    # print sympy.pretty([constify_step(xi).simplify() for xi in coeffs_flat])

    coeffs_flat = list(it.chain(*full_x_coeffs))
    print [xi.simplify() for xi in coeffs_flat]



if __name__ == '__main__':
    sys.exit(main())


# Tests
# ============================================================

def regression_test_t17():
    """Check that the method hasn't changed while I fiddle with stuff.
    """
    old = [-12/(dts[0]*(dts[0]**2 + 2*dts[0]*dts[1] + 2*dts[1]**2)), 12*(dts[0] + dts[1])/(dts[0]*dts[1]*(dts[0]**2 + 2*dts[0]*dts[1] + 2*dts[1]**2)), -12/(dts[1]*(dts[0]**2 + 2*dts[0]*dts[1] + 2*dts[1]**2)), 4/dts[0]**3, -4/dts[0]**3, 0, dts[1]*(dts[0] + dts[1])/(dts[0]**2 + 2*dts[0]*dts[1] + 2*dts[1]**2), (dts[0]**3 + 2*dts[0]**2*dts[1] + 2*dts[0]*dts[1]**2 + 2*dts[1]**3)/(2*dts[1]*(dts[0]**2 + 2*dts[0]*dts[1] + 2*dts[1]**2)), -dts[0]**3/(2*dts[1]*(dts[0]**2 + 2*dts[0]*dts[1] + 2*dts[1]**2))]

    new_coeffs_flat = list(it.chain(*full_method(t17_scheme)))
    for old, new in zip(old, new_coeffs_flat):
        utils.assert_sym_eq(old, new.simplify())
