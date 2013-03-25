# Some pairs of residuals and exact solutions for testing with:

from scipy import exp, tanh


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
