
from __future__ import division
from __future__ import absolute_import

import sympy
import scipy.misc
import sys
import itertools as it

from sympy import Rational as sRat
from operator import mul
from functools import partial as par

import simpleode.core.utils as utils
from functools import reduce


# A set of functions for symbolically calculating bdf forumlas.


# Define some (global) symbols to use
dts = list(sympy.var('Delta:9', real=True))
dys = list(sympy.var('Dy:9', real=True))
ys = list(sympy.var('y:9', real=True))

# subs i => step n+1-i (with Delta_{n+1} = t_{n+1} - t_n)


def divided_diff(order, ys, dts):
    """Caluclate divided differences of the list of values given
    in ys at points separated by the values in dts.

    Should work with symbols or numbers, only tested with symbols.
    """
    assert(len(ys) == order+1)
    assert(len(dts) == order)

    if order > 1:
        return ((divided_diff(order-1, ys[:-1], dts[:-1])
                 - divided_diff(order-1, ys[1:], dts[1:]))
                /
                sum(dts))
    else:
        return (ys[0] - ys[-1])/(dts[0])


def old_bdf_prefactor(order, implicit):
    """Calculate the non-divided difference part of the bdf approximation.
    For implicit this is the product term in (5.12) on page 400 of
    Hairer1991. For explicit it's more tricky and not given in any book
    that I've found, I derived it from expressions on pgs 400, 366 of
    Hairer1991.
    """

    def accumulate(iterable):
        """Return running totals (from python 3.2), i.e.

        accumulate([1,2,3,4,5]) --> 1 3 6 10 15
        """
        it = iter(iterable)
        total = next(it)
        yield total
        for element in it:
            total = total + element
            yield total

    assert(order >= 0)

    if order == 0:
        return 0
    elif order == 1:
        return 1
    else:
        if implicit:
            # dt0 * (dt0 + dt1) * (dt0 + dt1 + dt2) * ... (dt0 + dt1 + ... +
            # dt_{order-1})
            return _product(accumulate(dts[0:order-1]))
        else:
            # the maths is messy here... Basically the same as above but
            # with some different terms
            terms = it.chain([dts[0]], accumulate(dts[1:order-1]))
            return -1 * _product(terms)


def _steps_diff_to_list_of_dts(a, b, missing=None):
    """Get t_a - t_b in terms of dts.

    e.g.
    a = 0, b = 2:
    t_0 - t_2 = t_{n+1} - t_{n-1} = dt_{n+1} + dt_n = dts[0] + dts[1]

    a = 2, b = 0:
    t_2 - t_0 = - dts[0] - dts[1]
    """
    # if a and b are in the "wrong" order then it's just the negative of
    # the sum with them in the "right" order.
    if a > b:
        return map(lambda x: -1*x, _steps_diff_to_list_of_dts(b, a, missing))

    return dts[a:b]


def _product(l):
    """Return the product (i.e. all entrys multiplied together) of a list or iterator.
    """
    return reduce(mul, l, 1)


def bdf_prefactor(order, derivative_point):
    """Calculate the non-divided difference part of the bdf approximation
    with the derivative known at any integer point. For implicit BDF the
    known derivative is at n+1 (so derivative point = 0), others it is
    further back in time (>0).
    """
    assert(order >= 0)
    assert(derivative_point >= 0)

    if order == 0:
        return 0

    terms = 0

    # For each i in the summation (Note that for most i, for low derivative
    # point the contribution is zero. It's possible to do fancy algebra to
    # speed this up, but it's just not worth it!)
    for i in range(0, order):
        # Get a list of l values for which to calculate the product terms.
        l_list = [l for l in range(0, order) if l != i]

        # Calculate a list of product terms (in terms of dts).
        c = map(lambda b: sum(_steps_diff_to_list_of_dts(derivative_point, b)),
                l_list)

        # Multiply together and add to total
        terms = terms + _product(c)

    return terms


def bdf_method(order, derivative_point=0):
    """Calculate the bdf approximation for dydt. If implicit approximation
    is at t_{n+1}, otherwise it is at t_n.
    """

    # Each term is just the appropriate order prefactor multiplied by a
    # divided difference. Basically just a python implementation of
    # equation (5.12) on page 400 of Hairer1991.
    def single_term(n):
        return (
            bdf_prefactor(
                n,
                derivative_point) * divided_diff(n,
                                                 ys[:n+1],
                                                 dts[:n])
        )

    return sum(map(single_term, range(1, order+1)))


def main():

    print sympy.pretty(sympy.collect(bdf_method(2, 0).expand(), ys).simplify())

    print "code for ibdf2 step:"
    print my_bdf_code_gen(2, 0, True)

    # print "\n\n code for eBDF3 step:"
    # print my_bdf_code_gen(3, 1, False)

    # print "\n\n code for iBDF3 dydt approximation:"
    # print my_bdf_code_gen(3, 0, True)

    print "\n\n code for iBDF3 step:"
    print my_bdf_code_gen(3, 0, True)

    # print "\n\n code for iBDF4 dydt approximation:"
    # print my_bdf_code_gen(4, 0, True)

    print "\n\n code for eBDF3 step w/ derivative at n-1:"
    print my_bdf_code_gen(3, 2, True)

    # print sympy.pretty(sympy.Eq(dys[2], bdf_method(1, 2)))
    # print sympy.pretty(sympy.Eq(dys[2], bdf_method(2, 2)))
    # print sympy.pretty(sympy.Eq(dys[2], bdf_method(3, 2)))


def my_bdf_code_gen(order, derivative_point, solve_for_ynp1):

    dydt_expr = bdf_method(order, derivative_point)

    if solve_for_ynp1:
        # Set equal to dydt at derivative-point-th step, then solve for y_{n+1}
        bdf_method_solutions = sympy.solve(sympy.Eq(dydt_expr,
                                                    dys[derivative_point]), y0)

        # Check there's one solution only
        assert(len(bdf_method_solutions) == 1)

        # Convert it to a string
        bdf_method_code = str(
            bdf_method_solutions[0].expand().collect(ys+dys).simplify())

    else:
        bdf_method_code = str(dydt_expr.expand().collect(ys+dys).simplify())

    # Replace the sympy variables with variable names consistent with my
    # code in ode.py
    sympy_to_odepy_code_string_replacements = \
        {'Delta0': 'dtn', 'Delta1': 'dtnm1', 'Delta2': 'dtnm2', 'Delta3': 'dtnm3',
         'Dy0': 'dynp1', 'Dy1': 'dyn', 'Dy2': 'dynm1',
         'y0': 'ynp1', 'y1': 'yn', 'y2': 'ynm1', 'y3': 'ynm2', 'y4': 'ynm3'}

    # This is a rubbish way to do mass replace (many passes through the
    # text, any overlapping replaces will cause crazy behaviour) but it's
    # good enough for our purposes.
    for key, val in sympy_to_odepy_code_string_replacements.iteritems():
        bdf_method_code = bdf_method_code.replace(key, val)

    # Check that none of the replacements contain things that will be
    # replaced by other replacement operations. Maybe slow but good to test
    # just in case...
    for _, replacement in sympy_to_odepy_code_string_replacements.iteritems():
        for key, _ in sympy_to_odepy_code_string_replacements.iteritems():
            assert(replacement not in key)

    return bdf_method_code


if __name__ == '__main__':
    sys.exit(main())


# Tests
# ============================================================

def assert_sym_eq(a, b):
    """Compare symbolic expressions. Note that the simplification algorithm
    is not completely robust: might give false negatives (but never false
    positives).

    Try adding extra simplifications if needed, e.g. add .trigsimplify() to
    the end of my_simp.
    """

    def my_simp(expr):
        # Can't .expand() ints, so catch the zero case separately.
        try:
            return expr.expand().simplify()
        except AttributeError:
            return expr

    print
    print sympy.pretty(my_simp(a))
    print "equals"
    print sympy.pretty(my_simp(b))
    print

    # Try to simplify the difference to zero
    assert (my_simp(a - b) == 0)


def check_const_step(order, exact, derivative_point):

    # Derive bdf method
    b = bdf_method(order, derivative_point)

    # Set all step sizes to be Delta0
    b_const_step = b.subs({k: Delta0 for k in dts})

    # Compare with exact
    assert_sym_eq(exact, b_const_step)


def test_const_step_implicit():
    """Check that the implicit methods are correct for fixed step size by
    comparison with Hairer et. al. 1991 pg 366.
    """

    exacts = [(y0 - y1)/Delta0,
              (sRat(3, 2)*y0 - 2*y1 + sRat(1, 2)*y2)/Delta0,
              (sRat(11, 6)*y0 - 3*y1 + sRat(3, 2)*y2 - sRat(1, 3)*y3)/Delta0,
              (sRat(25, 12)*y0 - 4*y1 + 3*y2 - sRat(4, 3)*y3 + sRat(1, 4)*y4)/Delta0]

    orders = [1, 2, 3, 4]

    for order, exact in zip(orders, exacts):
        yield check_const_step, order, exact, 0


def test_const_step_explicit():

    # Get explicit BDF2 (implicit midpoint)'s dydt approximation G&S pg 715
    a = sympy.solve(-y0 + y1 + (1 + Delta0/Delta1)*Delta0*Dy1
                    - (Delta0/Delta1)**2*(y1 - y2), Dy1)
    assert(len(a) == 1)
    IMR_bdf_form = a[0].subs({k: Delta0 for k in dts})

    orders = [1, 2, 3]
    exacts = [(y0 - y1)/Delta0,
              IMR_bdf_form,
              #Hairer pg 364
              (sRat(1, 3)*y0 + sRat(1, 2)*y1 - y2 + sRat(1, 6)*y3)/Delta0
              ]

    for order, exact in zip(orders, exacts):
        yield check_const_step, order, exact, 1


def test_variable_step_implicit_bdf2():

    # From Gresho and Sani pg 715
    exact = sympy.solve(-(y0 - y1)/Delta0 +
                        (Delta0 / (2*Delta0 + Delta1)) * (y1 - y2)/Delta1 +
                        ((Delta0 + Delta1)/(2*Delta0 + Delta1)) * Dy0, Dy0)

    # Should only be one solution, get it
    assert(len(exact) == 1)
    exact = exact[0]

    # Get the method using my code
    mine = bdf_method(2, 0)

    assert_sym_eq(exact, mine)


def test_variable_step_explicit_bdf2():

    # Also from Gresho and Sani pg 715
    exact = sympy.solve(-y0 + y1 + (1 + Delta0/Delta1)*Delta0*Dy1
                        - (Delta0/Delta1)**2*(y1 - y2), Dy1)

    # Should only be one solution, get it
    assert(len(exact) == 1)
    exact = exact[0]

    # Get the method using my code
    mine = bdf_method(2, 1)

    assert_sym_eq(exact, mine)


def test_list_dts():

    # Check we have a list (not a tuple like before...)
    assert list(
        _steps_diff_to_list_of_dts(2,
                                   0)) == _steps_diff_to_list_of_dts(2,
                                                                     0)
    assert list(
        _steps_diff_to_list_of_dts(0,
                                   2)) == _steps_diff_to_list_of_dts(0,
                                                                     2)

    map(assert_sym_eq, _steps_diff_to_list_of_dts(0, 2), [dts[0], dts[1]])
    map(assert_sym_eq, _steps_diff_to_list_of_dts(2, 0), [-dts[0], -dts[1]])

    assert _steps_diff_to_list_of_dts(2, 2) == []


def test_product():
    assert _product([]) == 1
    assert _product(xrange(1, 11)) == scipy.misc.factorial(10, True)
    assert _product(xrange(0, 101)) == 0


def test_generalised_bdf_prefactor():
    def check(order, implicit):
        old = old_bdf_prefactor(order, implicit)
        if implicit:
            new = bdf_prefactor(order, 0)
        else:
            new = bdf_prefactor(order, 1)
        assert_sym_eq(old, new)

    orders = [0, 1, 2, 3, 4]
    for order in orders:
        for implicit in [True, False]:
            yield check, order, implicit

    def check_new_ones(order, real_value):
        calculated = bdf_prefactor(order, 2)
        assert_sym_eq(real_value, calculated)

    real_values = [(0, 0),
                   (1, 1),
                   (2, (-dts[0] - dts[1]) - dts[1]),
                   (3, (-dts[0] - dts[1])*-dts[1]),
                   (4, (-dts[0] - dts[1])*-dts[1]*dts[2]),
                   ]
    for order, real_value in real_values:
        yield check_new_ones, order, real_value
