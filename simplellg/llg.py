"""Various residuals for the LLG equation
"""
from __future__ import division
from math import sin, cos, tan, log, atan2, acos, pi, sqrt
import scipy as sp
import simplellg.utils as utils
import simplellg.ode as ode


def absmod2pi(a): return abs(a % (2 * pi))


def llg_spherical_residual(magnetic_parameters, t, m_sph, dmdt_sph):
    """ Calculate the residual for a guess at dmdt.
    """
    # Extract the parameters
    alpha = magnetic_parameters.alpha
    gamma = magnetic_parameters.gamma
    Ms = magnetic_parameters.Ms
    H, hazi, hpol = utils.cart2sph(magnetic_parameters.Hvec)

    # Ensure that the angles are in the correct range
    # ??ds improve
    _, mazi, mpol = utils.cart2sph(utils.sph2cart([Ms, m_sph[0], m_sph[1]]))

    dmazidt = dmdt_sph[0]
    dmpoldt = dmdt_sph[1]

    # Calculate fields:
    # no exchange, no Hms for now
    # anisotropy:
    # dEdmazi = 0
    # dEdmpol = -k1 * 2 * sin(pol) * cos(pol)

    # Zeeman:
    dEdmazi = - mu0 * Ms * hazi
    dEdmpol = - mu0 * Ms * hpol


    # From Nonlinear Magnetization Dynamics in Nanosystems By Isaak
    # D. Mayergoyz, Giorgio Bertotti, Claudio Serpico pg. 39 with theta =
    # polar angle, phi = azimuthal angle.
    residual = sp.empty((2))
    residual[0] = dpoldt + (alpha * sin(pol) * dazidt) + (gamma/Ms) *(1.0/sin(pol)) * dEdmazi
    residual[1] = (sin(pol) * dazidt) - (gamma/Ms) * dEdmpol - (alpha * dpoldt)

    # From some thesis,a ssuming theta = polar, phi = azi
    residual2 = sp.empty((2))
    residual2[0] = dpoldt + alpha * dazidt * sin(pol) + (gamma / (Ms * sin(pol))) * dEdmazi
    residual2[1] = dazidt * sin(pol) -  (gamma/Ms) * dEdmpol - alpha * dpoldt

    assert all(map(utils.almostEqual, residual, residual2))

    print m_sph


    #print m_sph, dmdt_sph, residual
    return residual


def llg_cartesian_residual(magnetic_parameters, t, m_cart, dmdt_cart):

    # Extract the parameters
    alpha = magnetic_parameters.alpha
    gamma = magnetic_parameters.gamma
    Hvec = magnetic_parameters.Hvec
    Ms = magnetic_parameters.Ms

    h_eff = Hvec

    residual = (alpha/Ms) * sp.cross(m_cart, dmdt_cart) \
      - gamma * sp.cross(m_cart, h_eff) \
      - dmdt_cart
    return residual


def llg_constrained_cartesian_residual(magnetic_parameters, t, m_cart,
                                       dmdt_cart):

    base_residual = llg_cartesian_residual(magnetic_parameters, t, m_cart[:-1],
                                           dmdt_cart[:-1])

    pass

    #

# Testing
# ============================================================

import simplellg.mallinson as mlsn
import functools as ft
import matplotlib.pyplot as plt


def test_llg_residuals():
    m0_sph = [0.0, pi/18]
    m0_cart = utils.sph2cart(tuple([1.0] + m0_sph))
    m0_constrained = list(m0_cart) + [None] #??ds

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
                                      tmax, dt = 0.01)
    m_sph = [utils.array2sph(m) for m in m_list]
    result_pols = [m.pol for m in m_sph]
    result_azis = [m.azi for m in m_sph]

    # Calculate exact solutions
    exact_times, exact_azis = \
      mlsn.calculate_equivalent_dynamics(mag_params, result_pols)

    # Check
    utils.assertListAlmostEqual(exact_azis, result_azis, 1e-3)
    utils.assertListAlmostEqual(exact_times, result_times, 1e-3)
