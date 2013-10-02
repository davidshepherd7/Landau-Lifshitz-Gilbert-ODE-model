
from __future__ import division
from __future__ import absolute_import

import sympy
import scipy.misc
import sys
import itertools as it
import math
import collections

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
    return 0  # No error to O(dtn**4) (third order)


def ebdf2_lte(dtn, dtnm1, dddyn):
    """Gresho and Sani pg.715, sign inverted, dtn/dtnm1 corrected as in Prinja
    """
    return (1 + dtnm1/dtn) * dtn**3 * dddyn / 6


def ebdf3_lte(*_):
    return 0  # no error to O(dtn**4) (third order)


def ab2_lte(dtn, dtnm1, dddyn):
    return ((2 + 3*(dtnm1/dtn)) * dtn**3 * dddyn)/12


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
    return expr.subs([(Sdts[1], Sdts[0]),
                    (Sdts[2], Sdts[0]), (Sdts[3], Sdts[0]),
                    (Sdts[4], Sdts[0])])


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
        # Otherwise assume only integers are rational (not other way to
        # represent rational numbers afaik).
        return is_integer(x)


def is_integer(x):
    """Check if x is an integer by comparing it with floor(x). This should work
    well with floats, sympy rationals etc.
    """
    return math.floor(x) == x


def is_half_integer(x):
    return x == (math.floor(x) + float(sRat(1, 2)))


def rational_as_mixed(x):
    """Convert a rational number to (integer_part, remainder_as_fraction)
    """
    assert is_rational(x)  # Don't want to accidentally end up with floats
                          # in here as very long rationals!
    x_int = int(x)
    return x_int, sRat(x - x_int)


def sum_dts(a, b):
    """Get t_a - t_b in terms of dts.

    e.g.
    a = 0, b = 2:
    t_0 - t_2 = t_{n+1} - t_{n-1} = dt_{n+1} + dt_n = Sdts[0] + Sdts[1]

    a = 2, b = 0:
    t_2 - t_0 = - Sdts[0] - Sdts[1]
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
        result = -a_frac * Sdts[a_int] + sum_dts(a_int, b)
        return result

    b_int, b_frac = rational_as_mixed(b)
    if b_frac != 0:
        return b_frac * Sdts[b_int] + sum_dts(a, b_int)

    return sum(Sdts[a:b])


# Define symbol names
# ============================================================
Sdts = sympy.symbols('Delta0:9')
Sys = sympy.symbols('y0:9')
Sdys = sympy.symbols('Dy0:9')
St = sympy.symbols('t')
Sdddynph = sympy.symbols("y'''_h")
SFddynph = sympy.symbols("F.y''_h")
y_np1_exact = sympy.symbols('y_0')
y_np1_imr, y_np1_p2, y_np1_p1 = sympy.symbols('y_0_imr y_0_p2 y_0_p1')


# Calculate full errors
# ============================================================
def generate_p_dt_func(symbolic_func):
    """Helper function: convert a symbolic function for predictor dt into a
    python function f(ts) = predictor_dt.
    """
    f = sympy.lambdify(Sdts[:5], symbolic_func)

    def p_dt(ts):
        assert len(ts) >= 6
        # Get last 5 dts in order: n, nm1, nm2 etc.
        dts = utils.ts2dts(ts[-6:])[::-1]
        return f(*dts)  # plug them into the symbolic function
    return p_dt


PTInfo = collections.namedtuple('PTInfo', ['time', 'y_est', 'dy_est'])


def bdf2_imr_ptinfos(pt):
    # assert pt == sRat(1,2)
    return [PTInfo(pt.time, None, "imr"),
            PTInfo(pt.time + sRat(1, 2) + 1, "corr_val", None),
            PTInfo(pt.time + sRat(1, 2) + 2, "corr_val", None)]


def bdf3_imr_ptinfos(pt):
    # assert pt == sRat(1,2)
    return [PTInfo(pt.time, None, "imr"),
            PTInfo(pt.time + sRat(1, 2), "corr_val", None),
            PTInfo(pt.time + sRat(1, 2) + 1, "corr_val", None),
            PTInfo(pt.time + sRat(1, 2) + 2, "corr_val", None)]


def t_at_time_point(ts, time_point):
    """Get the value of time at a "time point" (i.e. the i in n+1-i). Deal with
    non integer points by linear interpolation.
    """
    assert time_point >= 0

    if is_integer(time_point):
        return ts[-(time_point+1)]

    # Otherwise linearly interpolate
    else:
        pa = int(math.floor(time_point))
        pb = int(math.ceil(time_point))
        frac = time_point - pa
        ta = t_at_time_point(ts, pa)
        tb = t_at_time_point(ts, pb)
        return ta + frac*(tb - ta)


def ptinfo2yerr(pt):
    """Construct the sympy expression giving the error of the y approximation
    requested.
    """

    # Use (implicit) bdf2 with the derivative at tnp1 given by implicit
    # midpoint rule. Only works at half time steps.
    if pt.y_est == "bdf2 imr":
        _, y_np1_bdf2 = generate_predictor_scheme(
            bdf2_imr_ptinfos(pt), "ibdf2")
        y_error = -y_np1_bdf2.subs(y_np1_exact, 0)

    # Use (implicit) bdf3 with the derivative at tnp1 given by implicit
    # midpoint rule. Only works at half time steps.
    elif pt.y_est == "bdf3 imr":
        _, y_np1_bdf3 = generate_predictor_scheme(
            bdf3_imr_ptinfos(pt), "ibdf3")
        y_error = -y_np1_bdf3.subs(y_np1_exact, 0)

    # Just use corrector value at this point, counts as exact for lte.
    elif pt.y_est == "corr_val":
        y_error = 0

    elif pt.y_est == "exact":
        y_error = 0

    # None: don't use this value
    elif pt.y_est is None:
        y_error = None

    else:
        raise ValueError("Unrecognised y_est name " + str(pt.y_est))

    return y_error


def ptinfo2yfunc(pt, y_of_t_func=None):
    """Construct a python function to calculate the approximation to y
    requested.
    """

    # Use (implicit) bdf2 with the derivative at tnp1 given by implicit
    # midpoint rule. Only works at half time steps.
    if pt.y_est == "bdf2 imr":
        y_func, _ = generate_predictor_scheme(bdf2_imr_ptinfos(pt), "ibdf2")

    # Using (implicit) bdf3 with the derivative at tnp1 given by implicit
    # midpoint rule. Only works at half time steps.
    elif pt.y_est == "bdf3 imr":
        y_func, _ = generate_predictor_scheme(bdf3_imr_ptinfos(pt), "ibdf3")

    # Just use corrector value at this point, counts as exact for lte.
    elif pt.y_est == "corr_val":
        y_func = lambda ts, ys: ys[-(1+pt.time)]  # ??ds

    # Use given exact y function
    elif pt.y_est == "exact":
        assert y_of_t_func is not None

        def y_func(ts, ys):
            return y_of_t_func(t_at_time_point(ts, pt.time))

    # None: don't use this value
    elif pt.y_est is None:
        y_func = None

    else:
        raise ValueError("Unrecognised y_est name " + str(pt.y_est))

    return y_func


def ptinfo2dyerr(pt):
    """Construct the sympy expression giving the error of the dy approximation
    requested.
    """

    # Use the dy estimate from implicit midpoint rule (obviously only works
    # at the midpoint).
    if pt.dy_est == "imr":
        if not is_half_integer(pt.time):
            raise ValueError("imr dy approximation Only works for half integer "
                             + "points but given the point " + str(pt.time))

        # This works even for time points other than nph because in the
        # Taylor expansion {Fddy}_nmh = {Fddy}_nph + higher order terms,
        # luckily for us these end up in O(dtn**4) in the end.
        dy_error = imr_f_approximation_error(Sdts[0], SFddynph)

    # Use the provided f with known values to calculate dydt.
    elif pt.dy_est == "exact":
        dy_error = 0

    elif pt.dy_est == "fd4":
        dy_error = 0  # assuming we use 4th order fd with all points at
                     # integers

    # ??ds
    elif pt.dy_est == "exp test":
        dy_error = 0

    # Don't use this value
    elif pt.dy_est is None:
        dy_error = None

    else:
        raise ValueError(
            "Unrecognised dy_est name in error construction " + str(pt.dy_est))

    return dy_error


def ptinfo2dyfunc(pt, dydt_func):
    """Construct a python function to calculate the approximation to dy
    requested. If not using "exact" for dy_est the dydt_func can/should
    be None.
    """

    # Use the dy estimate from implicit midpoint rule (obviously only works
    # at the midpoint).
    if pt.dy_est == "imr":
        if not is_half_integer(pt.time):
            raise ValueError("imr dy approximation Only works for half integer "
                             + "points but given the point " + str(pt.time))

        # Need to drop some of the later time values if time point is
        # not the midpoint of 0 and 1 points (i.e. 1/2). Specifically, we
        # need:
        # 1/2 -> ts[:None]
        # 3/2 -> ts[:-1]
        # 5/2 -> ts[:-2]
        if pt.time == sRat(1, 2):
            # dy_func = ode.imr_dydt
            def dy_func(ts, ys):
                val = ode.imr_dydt(ts, ys)
                print "imr f(t) =", val
                return val
        else:
            x = -math.floor(pt.time)

            def dy_func(ts, ys):
                val = ode.imr_dydt(ts[:-int(math.floor(pt.time))],
                                   ys[:-int(math.floor(pt.time))])
                print "imr f(t) =", val
                return val

    elif pt.dy_est == "fd4":
        def dy_func(ts, ys):
            coeffs = [-1/12, -2/3, 0, 2/3, -1/12]
            ys_used = ys[-6:-1]

            assert len(ys) >= 6

            # ??ds only for constant step! with variable steps we can put
            # the 0 at a midpoint to reduce n-prev-steps needed.

            return sp.dot(coeffs, ys_used)

    # ??ds Use real dydt for exp
    elif pt.dy_est == "exp test":
        def dy_func(ts, ys):
            return ys[-(pt.time+1)]

    # Use the provided f with known values to calculate dydt. Only for
    # integer time points (i.e. where we already know y etc.) for now.
    elif pt.dy_est == "exact":
        assert dydt_func is not None

        if is_integer(pt.time):
            dy_func = lambda ts, ys: dydt_func(ts[pt.time])
        else:
            # Can't get y without additional approximations...
            def dy_func(ts, ys):
                val = dydt_func(t_at_time_point(ts, pt.time))
                print "f(t) =", val
                return val

    # None = "this value at this point should not be used"
    elif pt.dy_est is None:
        dy_func = None

    else:
        raise ValueError(
            "Unrecognised dy_est name in function construction " + str(pt.dy_est))

    return dy_func


def generate_predictor_scheme(pt_infos, predictor_name, symbolic=None):

    # Extract symbolic expressions from what we're given.
    if symbolic is not None:
        try:
            symb_exact, symb_F = symbolic

        except TypeError:  # not iterable
            symb_exact = symbolic

            Sy = sympy.symbols('y', real=True)
            symb_dy = sympy.diff(symb_exact, St, 1).subs(symb_exact, Sy)
            symb_F = sympy.diff(symb_dy, Sy).subs(Sy, symb_exact)

        dydt_func = sympy.lambdify(St, sympy.diff(symb_exact, St, 1))
        f_y = sympy.lambdify(St, symb_exact)

    else:
        dydt_func = None
        f_y = None

    # To cancel errors we need the final step to be at t_np1
    # assert pt_infos[0].time == 0
    n_hist = len(pt_infos) + 5

    # Create symbolic and python functions of (corrector) dts and ts
    # respectively that give the appropriate dts for this predictor.
    p_dtns = []
    p_dtn_funcs = []
    for pt1, pt2 in zip(pt_infos[:-1], pt_infos[1:]):
        p_dtns.append(sum_dts(pt1.time, pt2.time))
        p_dtn_funcs.append(generate_p_dt_func(p_dtns[-1]))

    # For each time point the predictor uses construct symbolic error
    # estimates for the requested estimate at that point and python
    # functions to calculate the value of the estimate.
    y_errors = map(ptinfo2yerr, pt_infos)
    y_funcs = map(par(ptinfo2yfunc, y_of_t_func=f_y), pt_infos)
    dy_errors = map(ptinfo2dyerr, pt_infos)
    dy_funcs = map(par(ptinfo2dyfunc, dydt_func=dydt_func), pt_infos)

    # Construct errors and function for the predictor
    # ============================================================
    if predictor_name == "ebdf2" or predictor_name == "wrong step ebdf2":

        if predictor_name == "wrong step ebdf2":
            temp_p_dtnm1 = p_dtns[1] + Sdts[0]
        else:
            temp_p_dtnm1 = p_dtns[1]

        y_np1_p_expr = y_np1_exact - (
            # Natural ebdf2 lte:
            ebdf2_lte(p_dtns[0], temp_p_dtnm1, Sdddynph)
            # error due to approximation to derivative at tn:
            + ode.ebdf2_step(p_dtns[0], 0, dy_errors[1], p_dtns[1], 0)
            # error due to approximation to yn
            + ode.ebdf2_step(p_dtns[0], y_errors[1], 0, p_dtns[1], 0)
            # error due to approximation to ynm1 (typically zero)
            + ode.ebdf2_step(p_dtns[0], 0, 0, p_dtns[1], y_errors[2]))

        def predictor_func(ts, ys):

            dtn = p_dtn_funcs[0](ts)
            yn = y_funcs[1](ts, ys)
            dyn = dy_funcs[1](ts, ys)

            dtnm1 = p_dtn_funcs[1](ts)
            ynm1 = y_funcs[2](ts, ys)
            return ode.ebdf2_step(dtn, yn, dyn, dtnm1, ynm1)

    elif predictor_name == "ab2":

        y_np1_p_expr = y_np1_exact - (
            # Natural ab2 lte:
            ab2_lte(p_dtns[0], p_dtns[1], Sdddynph)
            # error due to approximation to derivative at tn
            + ode.ab2_step(p_dtns[0], 0, dy_errors[1], p_dtns[1], 0)
            # error due to approximation to derivative at tnm1
            + ode.ab2_step(p_dtns[0], 0, 0, p_dtns[1], dy_errors[2])
            # error due to approximation to yn
            + ode.ab2_step(p_dtns[0], y_errors[1], 0, p_dtns[1], 0))

        def predictor_func(ts, ys):

            dtn = p_dtn_funcs[0](ts)
            yn = y_funcs[1](ts, ys)
            dyn = dy_funcs[1](ts, ys)

            dtnm1 = p_dtn_funcs[1](ts)
            dynm1 = dy_funcs[2](ts, ys)

            return ode.ab2_step(dtn, yn, dyn, dtnm1, dynm1)

    elif predictor_name == "ibdf2":

        y_np1_p_expr = y_np1_exact - (
            # Natural bdf2 lte:
            bdf2_lte(p_dtns[0], p_dtns[1], Sdddynph)
            # error due to approximation to derivative at tnp1
            + ode.ibdf2_step(p_dtns[0], 0, dy_errors[0], p_dtns[1], 0)
            # errors due to approximations to y at tn and tnm1
            + ode.ibdf2_step(p_dtns[0], y_errors[1], 0, p_dtns[1], 0)
            + ode.ibdf2_step(p_dtns[0], 0, 0, p_dtns[1], y_errors[2]))

        def predictor_func(ts, ys):

            pdts = [f(ts) for f in p_dtn_funcs]

            dynp1 = dy_funcs[0](ts, ys)
            yn = y_funcs[1](ts, ys)
            ynm1 = y_funcs[2](ts, ys)

            tsin = [t_at_time_point(ts, pt.time)
                    for pt in reversed(pt_infos[1:])]
            ysin = [ys[-(pt.time+1)] for pt in reversed(pt_infos[1:])]
            dt = pdts[0]

            return ode.ibdf2_step(pdts[0], yn, dynp1, pdts[1], ynm1)

    elif predictor_name == "ibdf3":

        y_np1_p_expr = y_np1_exact - (
            # Natural bdf2 lte:
            bdf3_lte(p_dtns[0], p_dtns[1], Sdddynph)
            # error due to approximation to derivative at tnp1
            + ode.ibdf3_step(
                dy_errors[0],
                p_dtns[0],
                0,
                p_dtns[1],
                0,
                p_dtns[2],
                0)
            # errors due to approximations to y at tn, tnm1, tnm2
            + ode.ibdf3_step(
                0,
                p_dtns[0],
                y_errors[1],
                p_dtns[1],
                0,
                p_dtns[2],
                0)
            + ode.ibdf3_step(
                0,
                p_dtns[0],
                0,
                p_dtns[1],
                y_errors[2],
                p_dtns[2],
                0)
            + ode.ibdf3_step(0, p_dtns[0], 0, p_dtns[1], 0, p_dtns[2], y_errors[3]))

        def predictor_func(ts, ys):

            pdts = [f(ts) for f in p_dtn_funcs]

            dynp1 = dy_funcs[0](ts, ys)
            yn = y_funcs[1](ts, ys)
            ynm1 = y_funcs[2](ts, ys)
            ynm2 = y_funcs[3](ts, ys)

            return ode.ibdf3_step(dynp1, pdts[0], yn, pdts[1], ynm1,
                                  pdts[2], ynm2)

    elif predictor_name == "ebdf3":
        y_np1_p_expr = y_np1_exact - (
            # Natural ebdf3 lte error:
            ebdf3_lte(p_dtns[0], p_dtns[1], p_dtns[2], Sdddynph)
            # error due to approximation to derivative at tn
            + ode.ebdf3_step(
                p_dtns[0],
                0,
                dy_errors[1],
                p_dtns[1],
                0,
                p_dtns[2],
                0)
            # errors due to approximation to y at tn, tnm1, tnm2
            + ode.ebdf3_step(p_dtns[0], y_errors[1], 0,
                             p_dtns[1], 0, p_dtns[2], 0)
            + ode.ebdf3_step(p_dtns[0], 0, 0,
                             p_dtns[1], y_errors[2], p_dtns[2], 0)
            + ode.ebdf3_step(p_dtns[0], 0, 0,
                             p_dtns[1], 0, p_dtns[2], y_errors[3]))

        def predictor_func(ts, ys):

            pdts = [f(ts) for f in p_dtn_funcs]

            dyn = dy_funcs[1](ts, ys)
            yn = y_funcs[1](ts, ys)
            ynm1 = y_funcs[2](ts, ys)
            ynm2 = y_funcs[3](ts, ys)

            return ode.ebdf3_step(pdts[0], yn, dyn,
                                  pdts[1], ynm1,
                                  pdts[2], ynm2)

    elif predictor_name == "use exact dddy":

        symb_dddy = sympy.diff(symb_exact, St, 3)
        f_dddy = sympy.lambdify(St, symb_dddy)

        print symb_exact
        print "dddy =", symb_dddy

        # "Predictor" is just exactly dddy, error forumlated so that matrix
        # turns out right. Calculate at one before last point (same as lte
        # "location").
        def predictor_func(ts, ys):
            tn = t_at_time_point(ts, pt_infos[1].time)
            return f_dddy(tn)

        y_np1_p_expr = Sdddynph

    elif predictor_name == "use exact Fddy":

        symb_ddy = sympy.diff(symb_exact, St, 2)
        f_Fddy = sympy.lambdify(St, symb_F * symb_ddy)

        print "Fddy =", symb_F * symb_ddy

        # "Predictor" is just exactly dddy, error forumlated so that matrix
        # turns out right. Calculate at one before last point (same as lte
        # "location").
        def predictor_func(ts, ys):
            tn = t_at_time_point(ts, pt_infos[1].time)
            return f_Fddy(tn)

        y_np1_p_expr = SFddynph

    else:
        raise ValueError("Unrecognised predictor name " + predictor_name)

    return predictor_func, y_np1_p_expr


def generate_predictor_pair_scheme(p1_points, p1_predictor,
                                   p2_points, p2_predictor,
                                   **kwargs):
    """Generate two-predictor lte system of equations and predictor step
    functions.
    """

    # Generate the two schemes
    p1_func, y_np1_p1_expr = generate_predictor_scheme(p1_points, p1_predictor,
                                                       **kwargs)
    p2_func, y_np1_p2_expr = generate_predictor_scheme(p2_points, p2_predictor,
                                                       **kwargs)

    # LTE for IMR: just natural lte:
    y_np1_imr_expr = y_np1_exact - imr_lte(Sdts[0], Sdddynph, SFddynph)

    # Return error expressions and stepper functions
    return (y_np1_p1_expr, y_np1_p2_expr, y_np1_imr_expr), (p1_func, p2_func)


def generate_predictor_pair_lte_est(lte_equations, predictor_funcs):

    assert len(lte_equations) == 3, "only for imr: need 3 ltes to solve"

    (p1_func, p2_func) = predictor_funcs

    A = system2matrix(lte_equations, [Sdddynph, SFddynph, y_np1_exact])

    # Look at some things for the constant step case:
    cse_print(constify_step(A))
    print sympy.pretty(constify_step(A).det())

    x = A.inv()

    cse_print(constify_step(x))

    # We can get nice expressions by factorising things (row 2 dotted with
    # [predictor values] gives us y_np1_exact):
    exact_ynp1_symb = sum([y_est * xi.factor() for xi, y_est in
                           zip(x.row(2), [y_np1_p1, y_np1_p2, y_np1_imr])])

    exact_ynp1_func = sympy.lambdify((Sdts[0], Sdts[1], Sdts[2], Sdts[3],
                                      y_np1_p1, y_np1_p2, y_np1_imr),
                                     exact_ynp1_symb)

    # Debugging:
    dddy_symb = sum([y_est * xi.factor() for xi, y_est in
                    zip(x.row(0), [y_np1_p1, y_np1_p2, y_np1_imr])])
    Fddy_symb = sum([y_est * xi.factor() for xi, y_est in
                    zip(x.row(1), [y_np1_p1, y_np1_p2, y_np1_imr])])
    dddy_func = sympy.lambdify((Sdts[0], Sdts[1], Sdts[2], Sdts[3],
                                y_np1_p1, y_np1_p2, y_np1_imr),
                               dddy_symb)
    Fddy_func = sympy.lambdify((Sdts[0], Sdts[1], Sdts[2], Sdts[3],
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

        dddy_est = dddy_func(
            dtn,
            dtnm1,
            dtnm2,
            dtnm3,
            y_np1_p1,
            y_np1_p2,
            y_np1_imr)
        Fddy_est = Fddy_func(
            dtn,
            dtnm1,
            dtnm2,
            dtnm3,
            y_np1_p1,
            y_np1_p2,
            y_np1_imr)
        # print "%0.16f"%y_np1_imr, "%0.16f"%y_np1_p1, "%0.16f"%y_np1_p2
        print
        print "abs(y_np1_imr - y_np1_p1) =", abs(y_np1_imr - y_np1_p1)
        print "abs(y_np1_p2 - y_np1_imr) =", abs(y_np1_p2 - y_np1_imr)
        print "dddy_est =", dddy_est
        print "Fddy_est =", Fddy_est
        print "y_np1_exact - y_np1_imr =", y_np1_exact - y_np1_imr
        print "dtn**3 * (dddy_est/24 - Fddy_est/8) =", dtn**3 * (dddy_est/24 - Fddy_est/8)

        # Compare with IMR value to get truncation error
        return y_np1_exact - y_np1_imr

    return lte_est


# import simpleode.core.example_residuals as er
# import scipy as sp
# from matplotlib.pyplot import show as pltshow
# from matplotlib.pyplot import subplots

# def main():

# Function to integrate

# residual = er.exp_residual
# exact = er.exp_exact

#     residual = par(er.damped_oscillation_residual, 1, 0)
#     exact = par(er.damped_oscillation_exact, 1, 0)

# method
#     lte_est = generate_predictor_pair((0, sRat(1,2), 2, "ebdf2"),
#                                       (0, sRat(1,2), sRat(3,2), "ab2"),
#                                       "bdf3",
#                                       "midpoint")
#     my_adaptor = par(ode.general_time_adaptor, lte_calculator=lte_est,
#                       method_order=2)
#     init_actions = par(ode.higher_order_start, 6)

# Do it
#     ts, ys = ode._odeint(residual, [sp.array(exact(0.0), ndmin=1)], [0.0],
#                          1e-4, 5.0, ode.midpoint_residual,
#                          1e-6, my_adaptor, init_actions)

# Plot

# Get errors + exact solution
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


# Tests
# ============================================================
import simpleode.core.example_residuals as er
import scipy as sp
import operator as op

# def check_dddy_estimates(exact_symb):
#     dt = 5e-2

# Derive the required functions/derivatives:
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


# Solve with imr
#     ts, est_ys = ode.odeint(residual, exact(0.0), dt=dt,
#                             tmax=3.0, method='imr',
#                             newton_tol=1e-10, jacobian_fd_eps=1e-12)

# Construct predictors
#     p1_steps = (0, sRat(1,2), 3)
#     p2_steps = (0, sRat(1,2), 4)
#     lte_equations, (p1_func, p2_func) = general_two_predictor(p1_steps, p2_steps)

# Compare estimates of values with actual values
#     n = 5
#     for par_ts, par_ys in zip(utils.partial_lists(ts, n), utils.partial_lists(est_ys, n)):
#         y_np1_p1 = p1_func(par_ts, par_ys)
#         y_np1_p2 = p2_func(par_ts, par_ys)

#         dtn = par_ts[-1] - par_ts[-2]
#         dtnm1 = par_ts[-2] - par_ts[-3]
#         y_np1_imr = par_ys[-1]

# dddy_est = t17_dddy_est(dtn, dtnm1, y_np1_p1, y_np1_p2, y_np1_imr)

#         print dtnm1, dtn
#         print "%0.16f" % y_np1_imr, "%0.16f" % y_np1_p1, "%0.16f" % y_np1_p2
#         print
#         print abs(y_np1_imr - y_np1_p2)
#         assert abs(y_np1_imr - y_np1_p2) > 1e-8
#         assert abs(y_np1_imr - y_np1_p1) > 1e-8

# Fddy_est = t17_Fddy_est(dtn, dtnm1, y_np1_p1, y_np1_p2, y_np1_imr)

#         dddy = exact_dddy(par_ts[-1])
#         Fddy = Fdoty(par_ts[-1], par_ys[-1])

# utils.assert_almost_equal(dddy_est, , min(1e-6, 30* dt**4))
# utils.assert_almost_equal(Fddy_est, Fddy, min(1e-6, 30* dt**4))


# check we actually did something!
#     assert utils.partial_lists(ts, n) != []


# def test_dddy_estimates():

#
#     t = sympy.symbols('t')

#     equations = [
#         3*t**3,
#         sympy.tanh(t),
#         sympy.exp(-t)
#         ]


#     for exact_symb in equations:
#          yield check_dddy_estimates, exact_symb


def test_sum_dts():

    # Check a simple fractional case
    utils.assert_sym_eq(sum_dts(sRat(1, 2), 1), Sdts[0]/2)

    # Check two numbers the same gives zero always
    utils.assert_sym_eq(sum_dts(1, 1), 0)
    utils.assert_sym_eq(sum_dts(0, 0), 0)
    utils.assert_sym_eq(sum_dts(sRat(1, 2), sRat(1, 2)), 0)

    def check_sum_dts(a, b):

        # Check we can swap the sign
        utils.assert_sym_eq(sum_dts(a, b), -sum_dts(b, a))

        # Starting half a step earlier
        utils.assert_sym_eq(sum_dts(a, b + sRat(1, 2)),
                            sRat(1, 2)*Sdts[int(b)] + sum_dts(a, b))

        # Check that we can split it up
        utils.assert_sym_eq(sum_dts(a, b),
                            sum_dts(a, b-1) + sum_dts(b-1, b))

    # Make sure b>=1 or the last check will fail due to negative b.
    cases = [(0, 1),
             (5, 8),
             (sRat(1, 2), 3),
             (sRat(2, 2), 3),
             (sRat(3, 2), 3),
             (0, sRat(9, 7)),
             (sRat(3/4), sRat(9, 7)),
             ]

    for a, b in cases:
        yield check_sum_dts, a, b


def test_ltes():

    import numpy
    numpy.seterr(
        all='raise',
        divide='raise',
        over=None,
        under=None,
        invalid=None)

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


# ??ds test generated predictor calculation functions as well!

def test_imr_predictor_equivalence():
    # Check the we generate imr's lte if we plug in the right times and
    # approximations to ebdf2 (~explicit midpoint rule).
    ynp1_imr = y_np1_exact - imr_lte(Sdts[0], Sdddynph, SFddynph)

    _, ynp1_p1 = generate_predictor_scheme([PTInfo(0, None, None),
                                            PTInfo(
                                                sRat(1, 2), "bdf2 imr", "imr"),
                                            PTInfo(1, "corr_val", None)],
                                           "ebdf2")
    utils.assert_sym_eq(ynp1_imr, ynp1_p1)

    # Should be exactly the same with ebdf to calculate the y-est at the
    # midpoint (because the midpoint value is ignored).
    _, ynp1_p2 = generate_predictor_scheme([PTInfo(0, None, None),
                                            PTInfo(
                                                sRat(1, 2), "bdf3 imr", "imr"),
                                            PTInfo(1, "corr_val", None)],
                                           "ebdf2")
    utils.assert_sym_eq(ynp1_imr, ynp1_p2)


def test_tr_ab2_scheme_generation():
    """Make sure tr-ab can be derived using same methodology as I'm using
    (also checks lte expressions).
    """

    dddy = sympy.symbols("y'''")
    y_np1_tr = sympy.symbols("y_{n+1}_tr")

    ab_pred, ynp1_ab2_expr = generate_predictor_scheme([PTInfo(0, None, None),
                                                        PTInfo(
                                                            1,
                                                            "corr_val",
                                                            "exp test"),
                                                        PTInfo(
                                                            2, "corr_val", "exp test")],
                                                       "ab2",
                                                       symbolic=sympy.exp(St))

    # ??ds hacky, have to change the symbol to represent where y''' is being
    # evaluated by hand!
    ynp1_ab2_expr = ynp1_ab2_expr.subs(Sdddynph, dddy)

    # Check that it gives the same result as we know from the lte
    utils.assert_sym_eq(y_np1_exact - ab2_lte(Sdts[0], Sdts[1], dddy),
                        ynp1_ab2_expr)

    # Now do the solve etc.
    y_np1_tr_expr = y_np1_exact - tr_lte(Sdts[0], dddy)
    A = system2matrix([ynp1_ab2_expr, y_np1_tr_expr], [dddy, y_np1_exact])
    x = A.inv()

    exact_ynp1_symb = sum([y_est * xi.factor() for xi, y_est in
                           zip(x.row(1), [y_np1_p1, y_np1_tr])])

    exact_ynp1_f = sympy.lambdify(
        (y_np1_p1,
         y_np1_tr,
         Sdts[0],
         Sdts[1]),
        exact_ynp1_symb)

    utils.assert_sym_eq(exact_ynp1_symb - y_np1_tr,
                        (y_np1_p1 - y_np1_tr)/(3*(1 + Sdts[1]/Sdts[0])))

    # Construct an lte estimator from this estimate
    def lte_est(ts, ys):
        ynp1_p = ab_pred(ts, ys)

        dtn = ts[-1] - ts[-2]
        dtnm1 = ts[-2] - ts[-3]

        ynp1_exact = exact_ynp1_f(ynp1_p, ys[-1], dtn, dtnm1)
        return ynp1_exact - ys[-1]

    # Solve exp using tr
    t0 = 0.0
    dt = 1e-2
    ts, ys = ode.odeint(er.exp_residual, er.exp_exact(t0),
                        tmax=2.0, dt=dt, method='tr')

    # Get error estimates using standard tr ab and the one we just
    # constructed here, then compare.
    this_ltes = ode.get_ltes_from_data(ts, ys, lte_est)

    tr_ab_lte = par(ode.tr_ab_lte_estimate, dydt_func=lambda t, y: y)
    standard_ltes = ode.get_ltes_from_data(ts, ys, tr_ab_lte)

    # Should be the same (actually the sign is different, but this doesn't
    # matter in lte).
    utils.assert_list_almost_equal(
        this_ltes, map(lambda a: a*-1, standard_ltes), 1e-8)


def test_bdf2_ebdf2_scheme_generation():
    """Make sure adaptive bdf2 can be derived using same methodology as I'm
    using (also checks lte expressions).
    """

    dddy = sympy.symbols("y'''")
    y_np1_bdf2 = sympy.symbols("y_{n+1}_bdf2")

    y_np1_ebdf2_expr = y_np1_exact - ebdf2_lte(Sdts[0], Sdts[1], dddy)
    y_np1_bdf2_expr = y_np1_exact - bdf2_lte(Sdts[0], Sdts[1], dddy)

    A = system2matrix([y_np1_ebdf2_expr, y_np1_bdf2_expr], [dddy, y_np1_exact])
    x = A.inv()

    exact_ynp1_symb = sum([y_est * xi.factor() for xi, y_est in
                           zip(x.row(1), [y_np1_p1, y_np1_bdf2])])

    answer = -(Sdts[1] + Sdts[0])*(
        y_np1_bdf2 - y_np1_p1)/(3*Sdts[0] + 2*Sdts[1])

    utils.assert_sym_eq(exact_ynp1_symb - y_np1_bdf2,
                        answer)


def test_generate_predictor_dt_func():

    symbs = [Sdts[0], Sdts[1], Sdts[2], Sdts[0] + Sdts[1],
             Sdts[0]/2 + Sdts[1]/2]

    t = sympy.symbols('t')

    fake_ts = utils.dts2ts(t, Sdts[::-1])

    for symb in symbs:
        print fake_ts
        yield utils.assert_sym_eq, symb, generate_p_dt_func(symb)(fake_ts)


def test_is_integer():
    tests = [(0, True),
             (0.0, True),
             (0.1, False),
             (1, True),
             (1.0, True),
             (1.1, False),
             (-1.0, True),
             (-1, True),
             (-1.1, False),
             (sp.nan, False),
             # (sp.inf, False), # Returns True, not really what I want but
             # not easily fixable for all possible infs
             # (afaik)
             (1 + 1e-15, False),
             (1 - 1e-15, False),
             ]

    for t, result in tests:
        assert is_integer(t) == result


def test_is_half_integer():

    tests = [(0, False),
             (1239012424481273, False),
             (sRat(1, 2), True),
             (1.5, True),
             (sRat(5, 2), True),
             (2.0 + sRat(1, 2), True),
             (1.5 + 1e-15, False),  # 1e-16 fails (gives true)
             (1.51, False),
             ]

    for t, result in tests:
        assert is_half_integer(t) == result


def test_t_at_time_point():
    tests = [0, 1, sRat(1, 2), sRat(3, 2), 1 + sRat(1, 2), sRat(4, 5), ]
    ts = range(11)
    for pt in tests:
        utils.assert_almost_equal(t_at_time_point(ts, pt), 10 - pt)

# def test_symbolic_predictor_func_comparison():


#     base = sRat(1,2)
#     p2, _ = generate_predictor_scheme([PTInfo(base, None, "imr"),
#         PTInfo(base + sRat(1,2) + 1, "corr_val", None),
#         PTInfo(base + sRat(1,2) + 2, "corr_val", None)],
#         "ibdf2")
#     p3, _ = generate_predictor_scheme([PTInfo(base, None, "imr"),
#         PTInfo(base + sRat(1,2) + 1, "corr_val", None),
#         PTInfo(base + sRat(1,2) + 2, "corr_val", None),
#         PTInfo(base + sRat(1,2) + 3, "corr_val", None)],
#         "ibdf3")
#     t = sympy.symbols('t')
#     fake_ts = utils.dts2ts(t, Sdts[::-1])
#     fake_ys = sym_ys[::-1]
#     sym_p2 = p2(fake_ts, fake_ys)
#     sym_p3 = p3(fake_ts, fake_ys)
#     print sympy.pretty(sym_p2.simplify())
#     print sympy.pretty(constify_step(sym_p2).simplify())
#     print sympy.pretty(sym_p3.simplify())
#     print sympy.pretty(constify_step(sym_p3).simplify())
# assert False
#
# def test_predictor_error_numerical():
# def check_predictor_error_numerical(exact_symb, pts_info,
# predictor_name):
# Generate forumlae
#         f_exact, residual, f_dys, f_jacobian = utils.symb2functions(exact_symb)
#         f_dddy = f_dys[3]
#         f_ddy = f_dys[2]
# Get derivative in terms of t only (needed to get exact dydt at
# non-integer points since we don't know y there).
#         dydt = sympy.diff(exact_symb, St, 1)
# Generate some midpoint method steps
#         dt = 0.01
#         ts, ys = ode.odeint(residual, f_exact(0.0), dt=dt, tmax=0.5,
#                             method="imr", newton_tol=1e-10,
#                             jacobian_fd_eps=1e-12)
# Generate the predictor function + error estimate
#         pfunc, ynp1_p_expr = generate_predictor_scheme(pts_info, predictor_name,
#                                                         dydt)
#         perr = -1*ynp1_p_expr.subs(y_np1_exact, 0)
# Get values at predictor points
#         py_np1 = pfunc(ts, ys)
#         pts = [t_at_time_point(ts, pt.time) for pt in reversed(pts_info)]
# pys = map(f_exact, pts[:-1]) +[py_np1]
#         exact_np1_p = f_exact(pts[-1])
# Compare predicted value with exact value
#         error_n = f_exact(ts[-2]) - ys[-2]
#         error_np1_p = exact_np1_p - py_np1
#         error_change = error_np1_p - error_n
# print "e_n =", error_n, "e_np1 =", error_np1_p, "e diff =", error_change
#         exact_ys = map(lambda t: sp.array([f_exact(t)], ndmin=1), ts)
#         pfromexact = pfunc(ts, exact_ys)
# print "pfromexact =", pfromexact, "error_np1 =", exact_np1_p -
# pfromexact
# Get last few dts in reverse order (like Sdts order)
#         dts = utils.ts2dts(ts[-10:])[::-1]
# Compute lte
#         tnph = ts[-2] + (ts[-1] + ts[-2])/2
#         ynph = f_exact(tnph)
#         f_lte = sympy.lambdify([Sdddynph, SFddynph]+list(Sdts), perr)
#         dddynph = f_dddy(tnph, ynph)
#         Fddynph = f_jacobian(tnph, ynph) * f_ddy(tnph, ynph)
#         lte_estimate = f_lte(dddynph, Fddynph, *dts)
#         print "dddynph =", dddynph, "Fddynph =", Fddynph
#         print error_change, "~", lte_estimate, "??"
#         assert False
# The test function itself:
#     St = sympy.symbols('t')
#     exacts = [
#         sympy.exp(2*St),
#         sympy.sin(St),
#         ]
#     predictors = [
# Simple:
# ([PTInfo(sRat(1,2), None, "imr"),
# PTInfo(2, "corr_val", None),
# PTInfo(3, "corr_val", None)],
# "ibdf2"),
#         ([PTInfo(0, None, None),
#         PTInfo(sRat(1,2), "bdf3 imr", "imr"),
#         PTInfo(2, "corr_val", None)],
#         "wrong step ebdf2"),
# ([PTInfo(sRat(1,2), None, "imr"),
# PTInfo(1, "corr_val", None),
# PTInfo(2, "corr_val", None)],
# "ibdf2"),
# ([PTInfo(sRat(1,2), None, "exact"),
# PTInfo(2, "corr_val", None),
# PTInfo(3, "corr_val", None),
# PTInfo(4, "corr_val", None)],
# "ibdf3"),
# ([PTInfo(sRat(1,2), None, "imr"),
# PTInfo(1, "corr_val", None),
# PTInfo(2, "corr_val", None),
# PTInfo(3, "corr_val", None)],
# "ibdf3"),
# Two level (bdfx to get ynph):
# bdf3:
# ([PTInfo(0, None, None),
# PTInfo(sRat(1,2), "bdf3 imr", "imr"),
# PTInfo(sRat(3,2), None, "imr")],
# "ab2"),
#         ([PTInfo(0, None, None),
#         PTInfo(sRat(1,2), "bdf3 imr", "imr"),
#         PTInfo(2, "corr_val", None)],
#         "ebdf2"),
# bdf2:
# ([PTInfo(0, None, None),
# PTInfo(sRat(1,2), "bdf2 imr", "imr"),
# PTInfo(sRat(3,2), None, "imr")],
# "ab2"),
# ([PTInfo(0, None, None),
# PTInfo(sRat(1,2), "bdf2 imr", "imr"),
# PTInfo(2, "corr_val", None)],
# "ebdf2"),
#         ]
#     for exact in exacts:
#         for pts, pname in predictors:
#             yield check_predictor_error_numerical, exact, pts, pname
def test_exact_predictors():

    def check_exact_predictors(exact, predictor):
        p1_func, y_np1_p1_expr = \
            generate_predictor_scheme(*predictor, symbolic=exact)

        # LTE for IMR: just natural lte:
        y_np1_imr_expr = y_np1_exact - imr_lte(Sdts[0], Sdddynph, SFddynph)

        # Generate another predictor
        p2_func, y_np1_p2_expr = \
            generate_predictor_scheme([PTInfo(sRat(1, 2), None, "imr"),
                                       PTInfo(2, "corr_val", None),
                                       PTInfo(3, "corr_val", None)], "ibdf2")

        A = system2matrix([y_np1_p1_expr, y_np1_p2_expr, y_np1_imr_expr],
                          [Sdddynph, SFddynph, y_np1_exact])

        # Solve for dddy and Fddy:
        x = A.inv()
        dddy_symb = sum([y_est * xi.factor() for xi, y_est in
                         zip(x.row(0), [y_np1_p1, y_np1_p2, y_np1_imr])])
        Fddy_symb = sum([y_est * xi.factor() for xi, y_est in
                         zip(x.row(1), [y_np1_p1, y_np1_p2, y_np1_imr])])

        # Check we got the right matrix and the right formulae
        if predictor[1] == "use exact dddy":
            assert A.row(0) == sympy.Matrix([[1, 0, 0]]).row(0)
            utils.assert_sym_eq(dddy_symb.simplify(), y_np1_p1)

        elif predictor[1] == "use exact Fddy":
            assert A.row(0) == sympy.Matrix([[0, 1, 0]]).row(0)
            utils.assert_sym_eq(Fddy_symb.simplify(), y_np1_p1)
        else:
            assert False

    exacts = [sympy.exp(St),
              sympy.sin(St),
              ]

    exact_predictors = [
        ([PTInfo(0, None, None),
          PTInfo(sRat(1, 2), None, None)],
         "use exact dddy"),

        ([PTInfo(0, None, None),
          PTInfo(sRat(1, 2), None, None)],
         "use exact Fddy"),
    ]

    for exact in exacts:
        for p in exact_predictors:
            yield check_exact_predictors, exact, p


if __name__ == '__main__':
    sys.exit(test_predictor_error_numerical())
