# Some pairs of residuals and exact solutions for testing with:

from scipy import exp, tanh
import scipy as sp

def exp_residual(ts, y, dydt): return y - dydt


def exp3_residual(ts, y, dydt): return 3*y - dydt
def exp3_exact(t): return exp(3*t)

# Not sure this works..
def exp_of_minus_t_residual(ts, y, dydt): return y + dydt
def exp_of_minus_t_exact(t): return exp(-t)


def poly_residual(t, y, dydt): return 4*t**3 + 2*t - dydt
def poly_exact(t): return t**4 + t**2


def exp_of_poly_residual(t, y, dydt): return y*(1 - 3*t**2) - dydt
def exp_of_poly_exact(t): return exp(t - t**3)


def tanh_residual(t, y, dydt, alpha=1.0, step_time=1.0):
    return (alpha * (1 - (tanh(alpha*(t - step_time)))**2))/2 - dydt
def tanh_exact(t, alpha=1.0, step_time=1.0):
    return (tanh(alpha*(t - step_time)) + 1)/2

def tanh_simple_residual(t, y, dydt, alpha=1.0, step_time=1.0):
    return (alpha * (1 - (tanh(alpha*(t - step_time)))**2)) - dydt
def tanh_simple_exact(t, alpha=1.0, step_time=1.0):
    return tanh(alpha*(t - step_time))

# Stiff examples
def van_der_pol_residual(t, y, dydt, mu=1000):
    return sp.array([
        dydt[0] - y[1],
        dydt[1] - mu * (1 - y[0]**2)*y[1] + y[0]
        ])
# No exact solution for van der pol...

# ODE example for paper?
def damped_oscillation_residual(omega, beta, t, y, dydt):
    return dydt + beta*sp.exp(-beta*t)*sp.sin(omega*t) - omega*sp.exp(-beta*t)*sp.cos(omega*t)

def damped_oscillation_exact(omega, beta, t):
    return sp.exp(-beta*t) * sp.sin(omega*t)
