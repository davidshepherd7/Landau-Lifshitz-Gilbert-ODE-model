
import sympy
import sys
from sympy import Rational as sRat
from operator import mul

# A set of functions for symbolically calculating bdf forumlas.


# Define some (global) symbols to use
dts = sympy.var('Delta:9', real=True)
dys = sympy.var('Dy:9', real=True)
ys = sympy.var('y:9', real=True)

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


def bdf2_prefactor(order, implicit):
    """Calculate the non-divided difference part of the bdf approximation.
    For implicit this is the product term in (5.12) on page 400 of
    Hairer1991. For explicit it's more tricky and not given in any book
    that I've found, I derived it from expressions on pgs 400, 366 of
    Hairer1991.
    """
    assert(order >= 0)

    if order == 0:
        return 0
    elif order == 1:
        return 1
    else:
        if implicit:
            return reduce(mul, accumulate(dts[0:order-1]), 1)
        else:
            # the maths is messy here...
            terms = [dts[0]] + list(accumulate(dts[1:order-1]))
            return -1 * reduce(mul, terms, 1)


def bdf_method(order, implicit=True):
    """Calculate the bdf approximation for dydt. If implicit approximation
    is at t_{n+1}, otherwise it is at t_n.
    """

    # Basically just a python implementation of equation (5.12) on page 400
    # of Hairer1991. Calculate two lists of terms then multiply them
    # together and sum up.
    a = range(1, order+1)
    prefactors = map(lambda n: bdf2_prefactor(n, implicit), a)
    divided_diffs = map(lambda n: divided_diff(n, ys[:n+1], dts[:n]), a)

    return sum([p*d for p, d in zip(prefactors, divided_diffs)])


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


def main():

    print "\n\n code for eBDF3 step:"
    print my_bdf_code_gen(3, False, True)

    print "\n\n code for iBDF3 dydt approximation:"
    print my_bdf_code_gen(3, True, False)

    print "\n\n code for iBDF4 dydt approximation:"
    print my_bdf_code_gen(4, True, False)

def my_bdf_code_gen(order, implicit, solve_for_ynp1):

    dydt_expr = bdf_method(order, implicit)

    if solve_for_ynp1:
        # Set equal to dydt at either n+1 or n th step, then solve for y_{n+1}
        if implicit:
            bdf_method_solutions = sympy.solve(sympy.Eq(dydt_expr, Dy0), y0)
        else:
            bdf_method_solutions = sympy.solve(sympy.Eq(dydt_expr, Dy1), y0)

        # Check there's one solution only
        assert(len(bdf_method_solutions) == 1)

        # Convert it to a string
        bdf_method_code = str(bdf_method_solutions[0].collect(ys+dys))

    else:
        bdf_method_code = str(dydt_expr.collect(ys+dys))

    # Replace the sympy variables with variable names consistent with my
    # code in ode.py
    sympy_to_odepy_code_string_replacements = \
      {'Delta0':'dtn', 'Delta1':'dtnm1', 'Delta2':'dtnm2', 'Delta3':'dtnm3',
       'Dy0':'dynp1', 'Dy1':'dyn',
       'y0':'ynp1', 'y1':'yn', 'y2':'ynm1', 'y3':'ynm2', 'y4':'ynm3'}

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
        return expr.expand().simplify()

    print sympy.pretty(my_simp(a))
    print sympy.pretty(my_simp(b))

    # Try to simplify the difference to zero
    assert(my_simp(a - b) == 0)


def check_const_step(order, exact, implicit):

    # Derive implicit bdf method
    b = bdf_method(order, implicit)

    # Set all step sizes to be Delta0
    b_const_step = b.subs({k:Delta0 for k in dts})

    # Compare with exact
    assert_sym_eq(b_const_step, exact)


def test_const_step_implicit():
    """Check that the implicit methods are correct for fixed step size by
    comparison with Hairer et. al. 1991 pg 366.
    """

    exacts = [(y0 - y1)/Delta0,
              (sRat(3,2)*y0 - 2*y1 + sRat(1,2)*y2)/Delta0,
              (sRat(11,6)*y0 - 3*y1 + sRat(3,2)*y2 - sRat(1,3)*y3)/Delta0,
              (sRat(25,12)*y0 - 4*y1 + 3*y2 - sRat(4,3)*y3 + sRat(1,4)*y4)/Delta0]

    orders = [1, 2, 3, 4]

    for order, exact in zip(orders, exacts):
        yield check_const_step, order, exact, True


def test_const_step_explicit():

    # Get explicit BDF2 (implicit midpoint)'s dydt approximation G&S pg 715
    a = sympy.solve(-y0 + y1 + (1 + Delta0/Delta1)*Delta0*Dy1
                    - (Delta0/Delta1)**2*(y1 - y2), Dy1)
    assert(len(a) == 1)
    IMR_bdf_form = a[0].subs({k:Delta0 for k in dts})


    orders = [2, 3]
    exacts = [IMR_bdf_form,
              (sRat(1,3)*y0 + sRat(1,2)*y1 - y2 + sRat(1,6)*y3)/Delta0 #Hairer pg 364
              ]

    for order, exact in zip(orders, exacts):
        yield check_const_step, order, exact, False


def test_variable_step_implicit_bdf2():

    # From Gresho and Sani pg 715
    exact = sympy.solve(-(y0 - y1)/Delta0 +
                                 (Delta0/ (2*Delta0 + Delta1))* (y1 - y2)/Delta1 +
                                 ((Delta0 + Delta1)/(2*Delta0 + Delta1)) * Dy0, Dy0)

    # Should only be one solution, get it
    assert(len(exact) == 1)
    exact = exact[0]

    # Get the method using my code
    mine = bdf_method(2, True)

    assert_sym_eq(exact, mine)


def test_variable_step_explicit_bdf2():

    # Also from Gresho and Sani pg 715
    exact = sympy.solve(-y0 + y1 + (1 + Delta0/Delta1)*Delta0*Dy1
                                 - (Delta0/Delta1)**2*(y1 - y2), Dy1)

    # Should only be one solution, get it
    assert(len(exact) == 1)
    exact = exact[0]

    # Get the method using my code
    mine = bdf_method(2, False)

    assert_sym_eq(exact, mine)
