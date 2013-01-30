
from math import sin, cos, tan, log, atan2, acos, pi, sqrt
import scipy as sp
import utils
import ode

def llg_spherical_residual(t, m_sph, dmdt_sph):
    """ Given m, h_app and a set of magnetic parameters return a residual
    function for the error in dmdt. In spherical polar coordinates.
    """
    # ??ds fix this!
    alpha = 0.1
    gamma = 1.0
    k1 = 0.0
    Hvec = [0, 0, -2.0]
    mu0 = 1.0
    Ms = 1.0

    _, hazi, hpol = utils.cart2sph(Hvec)

    azi = m_sph[0]
    pol = m_sph[1]

    dazidt = dmdt_sph[0]
    dpoldt = dmdt_sph[1]

    # no exchange, no Hms for now
    # anisotropy:
    dEdmazi = 0
    dEdmpol = -k1 * 2 * sin(pol) * cos(pol)

    # Zeeman:
    dEdmazi += - mu0 * Ms * hazi
    dEdmpol += - mu0 * Ms * hpol

    # From Nonlinear Magnetization Dynamics in Nanosystems By Isaak
    # D. Mayergoyz, Giorgio Bertotti, Claudio Serpico pg. 39 with theta =
    # polar angle, phi = azimuthal angle.
    residual = sp.empty((2))
    residual[0] = dpoldt + (alpha * sin(pol) * dazidt) + (1/sin(pol)) * dEdmazi
    residual[1] = (-1 * alpha * dpoldt) + (sin(pol) * dazidt) - dEdmpol


    return residual

def llg_cartesian_residual(t, m_cart, dmdt_cart):

    # ??ds fix this!
    alpha = 1.0
    gamma = 1.0
    k1 = 0.0
    Hvec = [0, 0, -2.0]
    mu0 = 1.0
    Ms = 1.0

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


import mallinson
import functools as ft
import matplotlib.pyplot as plt

def test_spherical_llg():
    mp = utils.MagParameters()
    mp.alpha = 0.001
    initial_m = [0.0, pi/18]
    sph_initial_m = utils.SphPoint(1.0, *initial_m)
    tmax = 1.0

    # Generate partial functions for azimuthal angle and time as functions
    # of current polar angle.
    mpazi = ft.partial(mallinson.calculate_azimuthal, mp, sph_initial_m.pol)
    mptime = ft.partial(mallinson.calculate_switching_time, mp, sph_initial_m.pol)

    # Generate llg timestepped solutions
    result_times, ms = \
      ode.odeint(llg_spherical_residual, [initial_m], tmax, dt = 0.01)

    # Extract spherical components
    m_results = map(lambda m: utils.SphPoint(1.0, *m), ms)
    result_azis = map(lambda m: m.azi, m_results)
    result_pols = map(lambda m: m.pol, m_results)

    # Compare with exact azimuthal and switching time
    exact_azis = map(mpazi, result_pols)
    exact_times = map(mptime, result_pols)

    # # # Plot
    # plt.plot(result_times, result_pols, '--',
    #          exact_times, result_pols)
    # # plt.figure()
    # # plt.plot(result_pols, result_azis, '--',
    # #          result_pols, exact_azis)
    # plt.show()

    # Check
    #map(utils.assertAlmostEqual, exact_azis, result_azis)
    map(utils.assertAlmostEqual, exact_times, result_times)


def test_cartesian_llg():
    mp = utils.MagParameters()
    initial_m = utils.sph2cart([1.0, 0.0, pi/18])
    sph_initial_m = utils.SphPoint(1.0, 0.0, pi/18)
    tmax = 5.0

    # Generate llg timestepped solutions
    result_times, ms = \
      ode.odeint(llg_cartesian_residual, [initial_m], tmax, dt = 0.01, method = 'bdf2')

    print ms[-1]

    # Extract spherical components
    m_results = map(lambda m: utils.cart2sph(m), ms)
    result_azis = map(lambda m: m.azi, m_results)
    result_pols = map(lambda m: m.pol, m_results)
    # Generate partial functions for azimuthal angle and time as functions
    # of current polar angle.
    mpazi = ft.partial(mallinson.calculate_azimuthal, mp, sph_initial_m.pol)
    mptime = ft.partial(mallinson.calculate_switching_time, mp, sph_initial_m.pol)

    # Compare with exact azimuthal and switching time
    exact_azis = map(mpazi, result_pols)
    exact_times = map(mptime, result_pols)

    # # Plot
    # plt.plot(result_times, map(lambda m: m[2], ms))
    # plt.plot(result_times, map(lambda m: m[1], ms))
    # plt.plot(result_times, map(lambda m: m[0], ms))

    # plt.figure()
    # plt.plot(result_times, result_pols, '--',
    #          exact_times, result_pols)
    # plt.figure()
    # plt.plot(result_pols, result_azis, '--',
    #          result_pols, exact_azis)

    # plt.show()

    # Check
    map(ft.partial(utils.assertAlmostEqual, tol = 1e-3), exact_azis, result_azis)
    map(ft.partial(utils.assertAlmostEqual, tol = 1e-3), exact_times, result_times)
