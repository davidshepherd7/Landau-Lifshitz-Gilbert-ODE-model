# Some pairs of residuals and exact solutions for testing with:

from scipy import exp, tanh
import scipy as sp

def exp_residual(ts, y, dydt): return y - dydt
def exp_dydt(t, y): return exp(t)
def exp_exact(t, y): return exp(t)


def exp3_residual(ts, y, dydt): return 3*y - dydt
def exp3_dydt(t, y): return 3 * exp(3*t)
def exp3_exact(t): return exp(3*t)

# Not sure this works..
def exp_of_minus_t_residual(ts, y, dydt): return y + dydt
def exp_of_minus_t_exact(t): return exp(-t)


def poly_residual(t, y, dydt): return 4*t**3 + 2*t - dydt
def poly_exact(t): return t**4 + t**2

# Useful because 2nd order integrators should be exact (and adaptive ones
# should recognise this and rapidly increase dt).
def square_residual(t, y, dydt): return 2*t - dydt
def square_dydt(t, y): return square_residual(t, y, 0)
def square_exact(t): return t**2


def exp_of_poly_residual(t, y, dydt): return y*(1 - 3*t**2) - dydt
def exp_of_poly_dydt(t, y): return exp_of_poly_residual(t, y, 0)
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
def van_der_pol_residual(t, y, dydt, mu=1000):
    return dydt - van_der_pol_dydt(t, y, mu)
def van_der_pol_dydt(t, y, mu=1000):
    return sp.array([y[1], mu * (1 - y[0]**2)*y[1] - y[0]])
# No exact solution for van der pol afaik

# ODE example for paper?
def damped_oscillation_residual(omega, beta, t, y, dydt):
    return dydt + beta*sp.exp(-beta*t)*sp.sin(omega*t) - omega*sp.exp(-beta*t)*sp.cos(omega*t)

def damped_oscillation_dydt(omega, beta, t, y):
    return - damped_oscillation_residual(omega, beta, t, y, 0)

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
