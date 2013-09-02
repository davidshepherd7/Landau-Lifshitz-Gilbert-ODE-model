
import sympy
from sympy import Rational as sRat
from operator import mul
import sys



# A set of functions for symbolically calculating bdf forumlas.


# Define some (global) symbols to use
dts = sympy.var('Delta:9', real=True)
dys = sympy.var('Dy:2', real=True)
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

    for order in range(1,4):
        print "implicit BDF", order, "is given by:"
        print sympy.pretty(sympy.Eq(Dy0, bdf_method(order, True)))



    for order in range(1,4):
        print "explicit BDF", order, "is given by:"
        print sympy.pretty(sympy.Eq(Dy1, bdf_method(order, False)))


    print "\n\n"
    print r"notation:"
    print r"subscript i => step n+1-i"
    print r"Delta_i : t_{n+1-1} - t_{n-i}"
    print r"Dy_i: \frac{\partial y(t_{n+1-i}, y_{n+1-i})}{\partial t}"

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
