
from math import sin, cos, tan, log, atan2, acos, pi, sqrt
import scipy as sp

import simplellg.utils as utils
import simplellg.ode as ode

def llg_spherical_residual(magnetic_parameters, t, m_sph, dmdt_sph):
    """ Calculate the residual for the error in dmdt.
    """
    # Extract the parameters
    alpha = magnetic_parameters.alpha
    gamma = magnetic_parameters.gamma
    k1 = magnetic_parameters.K1()
    mu0 = magnetic_parameters.mu0
    Ms = magnetic_parameters.Ms
    _, hazi, hpol = utils.cart2sph(magnetic_parameters.Hvec)

    # Extract components from arguments
    azi = m_sph[0]
    pol = m_sph[1]

    dazidt = dmdt_sph[0]
    dpoldt = dmdt_sph[1]

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
    k1 = magnetic_parameters.K1()
    Hvec = magnetic_parameters.Hvec
    mu0 = magnetic_parameters.mu0
    Ms = magnetic_parameters.Ms

    # Extract components from arguments
    mx = m_cart[0]
    my = m_cart[1]
    mz = m_cart[2]

    h_eff = Hvec

    residual = (alpha/Ms) * sp.cross(m_cart, dmdt_cart) \
      - gamma * sp.cross(m_cart, h_eff) \
      - dmdt_cart
    return residual

    #

# Testing
# ============================================================


import simplellg.mallinson as mlsn
import functools as ft
import matplotlib.pyplot as plt

# def test_spherical_llg():
#     mp = utils.MagParameters()
#     initial_m = [0.0, pi/18]
#     tmax = 1.8

#     # # ?/ds
#     #mp.alpha = 0.0
#     mp.gamma = 0.5

#     # Calculate llg timestepped solutions
#     f_residual = ft.partial(llg_spherical_residual, mp)
#     result_times, m_array = ode.odeint(f_residual, [initial_m], tmax, dt = 0.01)

#     # Extract the solution in spherical polar coordinates and compute exact
#     # solutions.
#     m_as_sph_points = map(utils.array2sph, m_array)
#     result_pols = [m.pol for m in m_as_sph_points]
#     result_azis = [m.azi for m in m_as_sph_points]
#     #exact_times, exact_azis = mlsn.calculate_equivalent_dynamics(mp, result_pols)

#     # # Plot
#     # mlsn.plot_vs_exact(mp, result_times, m_array)

#     # fig = utils.plot_sph_points(m_as_sph_points)
#     # ax = fig.axes[0]
#     # ax.plot([0,0],[0,0],[-1,1],'--')
#     # ax.set_zlim(-1,1)
#     # fig.axes[0].set_xlim(-1,1)
#     # fig.axes[0].set_ylim(-1,1)


#     plt.figure()
#     m_as_cart_points = map(utils.sph2cart, m_as_sph_points)

#     plt.plot(result_times, [m.x for m in m_as_cart_points])
#     plt.plot(result_times, [m.y for m in m_as_cart_points])
#     plt.plot(result_times, [m.z for m in m_as_cart_points])
#     plt.draw()

#     # # Check
#     # map(utils.assertAlmostEqual, exact_azis, result_azis)
#     # map(utils.assertAlmostEqual, exact_times, result_times)


def test_cartesian_llg():
    mp = utils.MagParameters()
    initial_m = utils.sph2cart([1.0, 0.0, pi/18])
    tmax = 1.0

    # Generate llg timestepped solutions
    f_residual = ft.partial(llg_cartesian_residual, mp)
    m_cart, result_times, _ = ode.odeint(f_residual, [initial_m], tmax, dt = 0.01)

    # Extract the solution in spherical polar coordinates and compute exact
    # solutions.
    m_as_sph_points = map(utils.array2sph, m_cart)
    result_pols = [m.pol for m in m_as_sph_points]
    result_azis = [m.azi for m in m_as_sph_points]
    exact_times, exact_azis = mlsn.calculate_equivalent_dynamics(mp, result_pols)

    # # # Plot
    # # mlsn.plot_vs_exact(mp, result_times, m_cart)
    # plt.figure()
    # m_as_cart_points = map(utils.sph2cart, m_as_sph_points)

    # plt.plot(result_times, [m.x for m in m_as_cart_points])
    # plt.plot(result_times, [m.y for m in m_as_cart_points])
    # plt.plot(result_times, [m.z for m in m_as_cart_points])
    # plt.show()

    # Check
    map(ft.partial(utils.assertAlmostEqual, tol = 1e-3), exact_azis, result_azis)
    map(ft.partial(utils.assertAlmostEqual, tol = 1e-3), exact_times, result_times)
