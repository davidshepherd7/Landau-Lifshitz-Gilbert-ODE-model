# Some pairs of residuals and exact solutions for testing with:

from scipy import exp, tanh, sin, cos
import scipy as sp

def exp_residual(t, y, dydt): return y - dydt
def exp_dydt(t, y): return y
def exp_exact(t): return exp(t)
def exp_dfdy(t, y): return 1

def exp3_residual(t, y, dydt): return 3*y - dydt
def exp3_dydt(t, y): return 3 * y
def exp3_exact(t): return exp(3*t)

# Not sure this works..
def exp_of_minus_t_residual(t, y, dydt): return exp_of_minus_t_dydt(t,y) - dydt
def exp_of_minus_t_exact(t): return exp(-t)
def exp_of_minus_t_dydt(t, y): return -y


def poly_residual(t, y, dydt): return 4*t**3 + 2*t - dydt
def poly_exact(t): return t**4 + t**2
def poly_dydt(t, y): return poly_residual(t, y, 0)
def poly_dfdy(t, y): return 0

# Useful because 2nd order integrators should be exact (and adaptive ones
# should recognise this and rapidly increase dt).
def square_residual(t, y, dydt): return 2*t - dydt
def square_dydt(t, y): return square_residual(t, y, 0)
def square_exact(t): return t**2
def square_dfdy(t, y): return 0


def exp_of_poly_residual(t, y, dydt): return exp_of_poly_dydt(t, y) - dydt
def exp_of_poly_dydt(t, y): return y*(1 - 3*t**2)
def exp_of_poly_exact(t): return exp(t - t**3)


def tanh_residual(t, y, dydt, alpha=1.0, step_time=1.0):
    return (alpha * (1 - (tanh(alpha*(t - step_time)))**2))/2 - dydt
def tanh_exact(t, alpha=1.0, step_time=1.0):
    return (tanh(alpha*(t - step_time)) + 1)/2

def tanh_simple_residual(t, y, dydt, alpha=1.0, step_time=1.0):
    return (alpha * (1 - (tanh(alpha*(t - step_time)))**2)) - dydt
def tanh_simple_exact(t, alpha=1.0, step_time=1.0):
    return tanh(alpha*(t - step_time))

# Stiff example
def van_der_pol_residual(t, y, dydt, mu=10):
    return dydt - van_der_pol_dydt(t, y, mu)
def van_der_pol_dydt(t, y, mu=10):
    return sp.array([y[1], mu * (1 - y[0]**2)*y[1] - y[0]])
# No exact solution for van der pol afaik

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


def stiff_trig_residual(t, y, dydt):
    return dydt - stiff_trig_dydt(t, y)
def stiff_trig_dydt(t, y):
    return 100 * (sin(t) - y)
def stiff_trig_exact(t):
    return (sin(t) - 0.01*cos(t) + 0.01*exp(-100*t))/1.0001

# ODE example for paper?
def damped_oscillation_residual(omega, beta, t, y, dydt):
    return dydt - damped_oscillation_dydt(omega, beta, t, y)
def damped_oscillation_dydt(omega, beta, t, y):
    return beta*sp.exp(-beta*t)*sp.sin(omega*t) - omega*sp.exp(-beta*t)*sp.cos(omega*t)
def damped_oscillation_exact(omega, beta, t):
    return sp.exp(-beta*t) * sp.sin(omega*t)

def damped_oscillation_dddydt(omega, beta, t):
    """See notes 16/8/2013."""
    a = sp.exp(- beta*t) * sp.sin(omega * t) # y in notes
    b = sp.exp(-beta * t) * sp.cos(omega * t) # k in notes
    return - beta**3 * a - omega**3 * b + omega*beta**2 * b + 3*beta*omega**2 * a

def damped_oscillation_ddydt(omega, beta, t):
    """See notes 19/8/2013."""
    a = sp.exp(- beta*t) * sp.sin(omega * t)
    b = sp.exp(-beta * t) * sp.cos(omega * t)
    return (beta**2 - omega**2)*a - 2*omega*beta*b


def constant_dydt(t, y):
    return 0
def constant_residual(t, y, dydt):
    return dydt - constant_dydt(t, y)
def constant_exact(t):
    return 1
def constant_dfdy(t, y):
    return 0
