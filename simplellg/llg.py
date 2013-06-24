"""Various residuals for the LLG equation
"""
from __future__ import division
from math import sin, cos, tan, log, atan2, acos, pi, sqrt
import scipy as sp
import simplellg.utils as utils
import simplellg.ode as ode


# def absmod2pi(a):
#     return abs(a % (2 * pi))


# def llg_spherical_residual(magnetic_parameters, t, m_sph, dmdt_sph):
#     """ Calculate the residual for a guess at dmdt.
#     """
#     # Extract the parameters
#     alpha = magnetic_parameters.alpha
#     gamma = magnetic_parameters.gamma
#     Ms = magnetic_parameters.Ms
#     H, hazi, hpol = utils.cart2sph(magnetic_parameters.Hvec)

#     # Ensure that the angles are in the correct range
#     # ??ds improve
#     _, mazi, mpol = utils.cart2sph(utils.sph2cart([Ms, m_sph[0], m_sph[1]]))

#     dmazidt = dmdt_sph[0]
#     dmpoldt = dmdt_sph[1]

#     # Calculate fields:
#     # no exchange, no Hms for now
#     # anisotropy:
#     # dEdmazi = 0
#     # dEdmpol = -k1 * 2 * sin(pol) * cos(pol)

#     if hazi < 0. or hazi > 2*pi or mazi < 0. or mazi > 2*pi:
#         raise ValueError

#     # Zeeman: ??ds minus sign?
#     if (mazi - hazi) == 0:
#         dEdmazi = 0.
#     else:
# dEdmazi = - Ms * H * sin(absmod2pi(mazi - hazi)) * sp.sign(mazi - hazi)

#     if (mpol - hpol) == 0:
#         dEdmpol = 0.
#     else:
#         dEdmpol = - Ms * H * sin(abs(mpol - hpol)) * sp.sign(mpol - hpol)

#     print abs(mazi - hazi), abs(mpol - hpol)

#     # From Nonlinear Magnetization Dynamics in Nanosystems By Isaak
#     # D. Mayergoyz, Giorgio Bertotti, Claudio Serpico pg. 39 with theta =
#     # polar angle, phi = azimuthal angle.
#     residual = sp.empty((2))
#     residual[0] = dmpoldt + (alpha * sin(mpol) * dmazidt) \
#         + (gamma/Ms) * (1.0/sin(mpol)) * dEdmazi
#     residual[1] = (sin(mpol) * dmazidt) \
#         - (gamma/Ms) * dEdmpol - (alpha * dmpoldt)

#     return residual


def llg_cartesian_residual(magnetic_parameters, t, m_cart, dmdt_cart):

    # Extract the parameters
    alpha = magnetic_parameters.alpha
    gamma = magnetic_parameters.gamma
    Ms = magnetic_parameters.Ms
    Hk_vec = magnetic_parameters.Hk_vec(m_cart)

    # Nasty hack to allow Hvec functions or vectors
    try:
        Hvec = magnetic_parameters.Hvec(t)
    except TypeError:
        Hvec = magnetic_parameters.Hvec

    h_eff = Hvec # + Hk

    residual = ((alpha/Ms) * sp.cross(m_cart, dmdt_cart)
                - gamma * sp.cross(m_cart, h_eff)
                - dmdt_cart)
    return residual


def llg_cartesian_dfdm(magnetic_parameters, t, m_cart, dmdt_cart):

    # Extract the parameters
    alpha = magnetic_parameters.alpha
    gamma = magnetic_parameters.gamma
    Ms = magnetic_parameters.Ms

    # Nasty hack to allow Hvec functions or vectors
    try:
        Hvec = magnetic_parameters.Hvec(t)
    except TypeError:
        Hvec = magnetic_parameters.Hvec

    h_eff = Hvec

    dfdm = - gamma * utils.skew(h_eff) + (alpha/Ms) * utils.skew(dmdt_cart)

    return dfdm


# def llg_constrained_cartesian_residual(magnetic_parameters, t, m_cart,
#                                        dmdt_cart):

#     base_residual = llg_cartesian_residual(magnetic_parameters, t, m_cart[:-1],
#                                            dmdt_cart[:-1])

#     pass


# Testing
# ============================================================

import simplellg.mallinson as mlsn
import functools as ft
import matplotlib.pyplot as plt

# ??ds replace with substituting the exact solution into the residual and
# checking it is zero?


def test_llg_residuals():
    m0_sph = [0.0, pi/18]
    m0_cart = utils.sph2cart(tuple([1.0] + m0_sph))
    # m0_constrained = list(m0_cart) + [None]  # ??ds

    residuals = [(llg_cartesian_residual, m0_cart),
                 #(llg_spherical_residual, m0_sph),
                 # (llg_constrained_cartesian_residual, m0_constrained),
                 ]

    for r, i in residuals:
        yield check_residual, r, i


def check_residual(residual, initial_m):
    mag_params = utils.MagParameters()
    tmax = 3.0
    f_residual = ft.partial(residual, mag_params)

    # Timestep to a solution + convert to spherical
    m_list, result_times = ode.odeint(f_residual, sp.array(initial_m),
                                      tmax, dt=0.01)
    m_sph = [utils.array2sph(m) for m in m_list]
    result_pols = [m.pol for m in m_sph]
    result_azis = [m.azi for m in m_sph]

    # Calculate exact solutions
    exact_times, exact_azis = \
        mlsn.calculate_equivalent_dynamics(mag_params, result_pols)

    # Check
    utils.assert_list_almost_equal(exact_azis, result_azis, 1e-3)
    utils.assert_list_almost_equal(exact_times, result_times, 1e-3)


def test_dfdm():

    ms = [utils.sph2cart([1.0, 0.0, pi/18]),
          utils.sph2cart([1.0, 0.0, 0.0001*pi]),
          utils.sph2cart([1.0, 0.0, 0.999*pi]),
          utils.sph2cart([1.0, 0.3*2*pi, 0.5*pi]),
          utils.sph2cart([1.0, 2*pi, pi/18]),
          ]

    for m in ms:
        yield check_dfdm, m


def check_dfdm(m_cart):
    """Compare dfdm function with finite differenced dfdm."""

    # Some parameters
    magnetic_parameters = utils.MagParameters()
    t = 0.3

    # Use LL to get dmdt:
    alpha = magnetic_parameters.alpha
    gamma = magnetic_parameters.gamma
    Hvec = magnetic_parameters.Hvec
    Ms = magnetic_parameters.Ms

    h_eff = Hvec
    dmdt_cart = (gamma/(1+alpha**2)) * sp.cross(m_cart, h_eff) \
        - (alpha*gamma/((1+alpha**2)*Ms)) * sp.cross(m_cart, sp.cross(
                                                     m_cart, h_eff))

    # Calculate with function
    dfdm_func = llg_cartesian_dfdm(magnetic_parameters, t, m_cart, dmdt_cart)

    def f(t, m_cart, dmdt_cart):
        # f is the residual + dm/dt (see notes 27/2/13)
        return llg_cartesian_residual(magnetic_parameters,
                                      t, m_cart, dmdt_cart) + dmdt_cart
    # FD it
    dfdm_fd = sp.zeros((3, 3))
    r = f(t, m_cart, dmdt_cart)
    delta = 1e-8
    for i, m in enumerate(m_cart):
        m_temp = sp.array(m_cart).copy()  # Must force a copy here
        m_temp[i] += delta
        r_temp = f(t, m_temp, dmdt_cart)
        r_diff = (r_temp - r)/delta

        for j, r_diff_j in enumerate(r_diff):
            dfdm_fd[i][j] = r_diff_j

    print dfdm_fd

    # Check the max of the difference
    utils.assert_almost_zero(sp.amax(dfdm_func - dfdm_fd), 1e-6)
