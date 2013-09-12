# Some pairs of residuals and exact solutions for testing with:

from scipy import exp, tanh, sin, cos
import scipy as sp
from functools import partial as par
import sympy


import utils

# define some symbolic variables and functions
sym_t = sympy.Symbol('t')
sym_y = sympy.Function('y')(sym_t)
sym_dy = sympy.diff(sym_y, sym_t, 1)

# symbolic residual and ODE defining functions
sym_f = sympy.Symbol('f')(sym_t, sym_y)
sym_r = sympy.Symbol('r')(sym_t, sym_y, sym_dy)


# Maths libs to use. Use numpy if possible for speed.
_mlibs = ("numpy", "math", "mpmath", "sympy")


def triple_from_symfunc(sympy_string):
    """Convert a string of a symbolic equation into the triple of functions
    needed for an ode solve: residual, dydt and exact solution.

    Symbolic equation must be a STRING of the form:
    Eq(y, exp(t))
    Eq(f, exp(t))
    Eq(r, Dy - exp(t))

    where:
    t = independent variable
    y = y(t) = ode solution
    Dy = first derivative of y
    f = derivative function defining the ode (i.e. f(t, y) = dy/dt
    r = residual function

    Given an expression for one of y, f or r the others are derived
    automatically where possible.
    """

    expr = sympy.sympify(sympy_string)

    print sympy.srepr(expr)

    # Try to solve for each possible given value in turn:
    y_func = sympy.solve(expr, sym_y)
    f_func = sympy.solve(expr, sym_f)
    r_func = sympy.solve(expr, sym_r)

    # Check that only one of them has a solution and that it has only one
    # solution.
    funcs = [y_func, f_func, r_func]
    solved_funcs = filter(lambda l: len(l) > 0, funcs)
    assert len(solved_funcs) > 0, "Couldn't solve for y, f or r. Maybe you forgot to set equality?"
    assert len(solved_funcs) == 1
    assert len(solved_funcs[0]) == 1


    # Various different input equalities are possible, deal with them:
    if y_func is not []:
        # We have the exact solution in the string, extract it and
        # calculate derivative and y0.

        symbolic_exact = y_func[0]
        exact = sympy.lambdify((sym_t), symbolic_exact, _mlibs)

        symbolic_dydt = sympy.diff(symbolic_exact, sym_t, 1).subs(symbolic_exact, sym_y)
        dydt = sympy.lambdify((sym_t, sym_y), symbolic_dydt)

        residual = lambda t, y, dydt_var: dydt_var - dydt(t, y)

    elif f_func is not []:
        # We have the derivative
        exact = None
        symbolic_dydt = f_func[0]
        dydt = sympy.lambdify((sym_t, sym_y), symbolic_dydt, _mlibs)
        residual = lambda t, y, dydt_var: dydt_var - dydt(t, y)

    elif r_func is not []:
        # We have only the residual
        exact = None
        dydt = None
        symbolic_residual = r_func[0]
        residual = sympy.lambdify((sym_t, sym_y, sym_dy), symbolic_residual, _mlibs)

    else:
        raise ValueError("function type must be either exact, dydt or residual")

    return residual, dydt, exact

# Some simple exactly solvable examples
exp = "Eq("+str(sym_y)+", exp(t))"
square = "Eq("+str(sym_y)+", t**2)"
constant = "Eq("+str(sym_y)+", 1)",
exp3 = "Eq("+str(sym_y)+", exp(3*t))"
poly = "Eq("+str(sym_y)+", t**4 + t**2)"
exp_of_poly = "Eq("+str(sym_y)+", exp(t - t**3))"
stiff_trig = "Eq("+str(sym_y)+", (sin(t) - 0.01*cos(t) + 0.01*exp(-100*t))/1.0001)"

# van_der_pol_problem = None
# stiff_damped_example_problem = None
# stiff_example_problem = None

# And some problems with coefficients:
def tanh_problem(alpha=1.0, step_time=1.0):
    return "Eq(y, (tanh("+str(alpha)+"*(t - "+str(step_time)+")) + 1)/2)"

def midpoint_method_killer_problem(y0, g_string, l):
    g = sympy.sympify(g_string, strict=True)
    dgdt = sympy.diff(g, sym_t, 1)
    g0 = g.subs(sym_t, 0).evalf()
    exact = (y0 - g0)*sympy.exp(-l*sym_t) + g
    exact = 1
    return "Eq("+str(sym_y)+", "+ str(exact) + ")"

trig_midpoint_killer_problem = par(midpoint_method_killer_problem, 5,
                                   "sin(t) + cos(t)")

poly_midpoint_killer_problem = par(midpoint_method_killer_problem, 5, "t**2")

def damped_oscillation_problem(alpha, beta):
    return "Eq("+str(sym_y)+", exp(-"+str(beta) +"*t) * sin("+str(omega)+"*t))"

def van_der_pol_problem(mu):
    ys = sympy.symarray('y', 2)
    # exact =
    # print exact
    return "Eq(f, sp.array([ys[1], "+str(mu)+" * (1 - ys[0]**2)*ys[1] - ys[0]]))"



# The classical example of a stiff ODE
def stiff_damped_example_residual(t, y, dydt):
    return dydt - stiff_damped_example_dydt(t, y)
def stiff_damped_example_dydt(t, y):
    return sp.dot(sp.array([[-1, 1], [1, -1000]]), y)

# From G&S pg 258
def stiff_example_residual(t, y, dydt):
    return dydt - stiff_damped_example_dydt(t, y)
def stiff_example_dydt(t, y):
    return sp.dot(sp.array([[-1, 1], [1, -1000]]), y)
def stiff_example_exact(t):
    l1 = 1000.0010001
    l2 = 0.998999
    return sp.array([(-l2/(l1 - l2))*sp.exp(-l1*t) + (l1/(l1 - l2))*sp.exp(-l2*t),
                     (l2*(l1 - 1))*sp.exp(-l1*t)/(l1 - l2) + l1*(1 - l2)*sp.exp(-l2*t)/(l1 - l2)
                     ])


# Testing
# ============================================================

def exp_residual(t, y, dydt): return y - dydt
def exp_dydt(t, y): return y
def exp_exact(t): return sp.exp(t)


def test_sympy_exact_func_conversion():
    test_points = sp.arange(-3, 3, 0.1)

    new_residual, new_dydt, new_exact = triple_from_symfunc(exp)

    old_ys = map(exp_exact, test_points)
    new_ys = map(new_exact, test_points)
    utils.assert_list_almost_equal(old_ys, new_ys)

    old_dys = map(exp_dydt, test_points, old_ys)
    new_dys = map(new_dydt, test_points, new_ys)
    utils.assert_list_almost_equal(old_dys, new_dys)

    new_residual = map(new_residual, test_points, old_ys, old_dys)
    utils.assert_list_almost_zero(new_residual)


def test_sympy_dydt_substitution():
    _, dydt, _ = triple_from_symfunc(exp)

    # Check that dydt = y in two ways: by running it with t=None and by
    # comparing over a large range of long decimal numbers
    ys = sp.arange(-100, 100, sp.sqrt(2)/2)
    utils.assert_list_almost_equal(ys, map(dydt, [], ys))


# def test_vector_symbolics():
#     print van_der_pol_problem(2)
#     print triple_from_symfunc(van_der_pol_problem(2))
#     assert False

    #
