"""Calculate energies for a single magnetisation vector in a field.
"""

from __future__ import division
import operator as op
from math import sin, cos, tan, log, atan2, acos, pi, sqrt
import scipy as sp
import itertools as it

import simplellg.utils as utils
from simplellg.llg import heff

def llg_state_energy(sph, mag_params, t=None):
    """Assuming unit volume, spatially constant, spherical particle.

    Energies taken from [Coey2010].

    Ignore stress and magnetostriction.

    t can be None if applied field is not time dependant.
    """
    return exchange_energy(sph, mag_params) \
        + magnetostatic_energy(sph, mag_params) \
        + magnetocrystalline_anisotropy_energy(sph, mag_params) \
        + zeeman_energy(sph, mag_params, t)


def exchange_energy(sph, mag_params):
    """Always zero because 0D/spatially constant."""
    return 0.0


# ??ds generalise!
def magnetostatic_energy(sph, mag_params):
    """ For a small, spherical particle:
    Ems = - 0.5 mu0 (M.Hms) = 0.5/3 * Ms**2
    """
    Ms = mag_params.Ms
    mu0 = mag_params.mu0
    return (0.5/3) * mu0 * Ms**2


def magnetocrystalline_anisotropy_energy(sph, mag_params):
    """ Eca = K1 (m.e)^2"""
    K1 = mag_params.K1
    m_cart = utils.sph2cart(sph)
    return K1 * (1 - sp.dot(m_cart, mag_params.easy_axis)**2)


def zeeman_energy(sph, mag_params, t=None):
    """ Ez = - mu0 * (M.Happ(t)), t can be None if the field is not time
    dependant.
    """
    Ms = mag_params.Ms
    Happ = mag_params.Hvec(t)
    mu0 = mag_params.mu0

    m = utils.sph2cart(sph)
    return -1 * mu0 * Ms * sp.dot(m, Happ)


def recompute_alpha(sph_start, sph_end, t_start, t_end, mag_params):
    """
    From a change in energy we can calculate what the effective damping
    was. For more details see [Albuquerque2001,
    http://dx.doi.org/10.1063/1.1355322].

    alpha' = - (1/(M**2))  *( dE/dt  /  spaceintegral( (dm/dt**2  )))

    No space integral is needed because we are working with 0D LLG.
    """

    Ms = mag_params.Ms
    dt = t_end - t_start

    # Estimate dEnergy / dTime
    dEdt = (llg_state_energy(sph_end, mag_params, t_end)
            - llg_state_energy(sph_start, mag_params, t_start)
            )/dt

    # Estimate dMagnetisation / dTime then take sum of squares
    dmdt = [(m2 - m1)/dt for m1, m2 in
            zip(utils.sph2cart(sph_start), utils.sph2cart(sph_end))]
    dmdt_sq = sp.dot(dmdt, dmdt)

    # dE should be negative so the result should be positive.
    return - (1/(Ms**2)) * (dEdt / dmdt_sq)


def low_accuracy_recompute_alpha_varying_fields(sph_start, sph_end, t_start, t_end, mag_params):
    """
    Compute effective damping from change in magnetisation and change in
    applied field.

    From Nonlinear magnetization dynamics in nanosystems eqn (2.15).

    See notes 30/7/13.

    Derivatives are estimated using BDF1 finite differences.
    """

    # Only for normalised problems!
    assert(mag_params.Ms == 1)

    # Get some values
    dt = t_end - t_start
    m_cart_end = utils.sph2cart(sph_end)
    h_eff_end = heff(mag_params, t_end, m_cart_end)
    mxh = sp.cross(m_cart_end, h_eff_end)

    # Finite difference derivatives
    dhadt = (mag_params.Hvec(t_start) - mag_params.Hvec(t_end))/dt

    assert(all(dhadt == 0)) # no field for now

    dedt = (llg_state_energy(sph_end, mag_params, t_end)
            - llg_state_energy(sph_start, mag_params, t_start)
            )/dt

    sigma = sp.dot(mxh, mxh) / (dedt + sp.dot(m_cart_end, dhadt))

    possible_alphas = sp.roots([1, sigma, 1])

    a = (-sigma + sqrt(sigma**2 - 4))/2
    b = (-sigma - sqrt(sigma**2 - 4))/2

    possible_alphas2 = [a,b]
    utils.assert_list_almost_equal(possible_alphas, possible_alphas2)

    print(sigma, possible_alphas)

    def real_and_positive(x): return sp.isreal(x) and x > 0

    alphas = filter(real_and_positive, possible_alphas)
    assert(len(alphas) == 1)
    return sp.real(alphas[0])


def recompute_alpha_varying_fields(sph_start, sph_end, t_start, t_end, mag_params):
    """
    Compute effective damping from change in magnetisation and change in
    applied field.

    See notes 30/7/13 pg 5.

    Derivatives are estimated using BDF1 finite differences.
    """

    # Only for normalised problems!
    assert(mag_params.Ms == 1)

    # Get some values
    dt = t_end - t_start
    m_cart_end = utils.sph2cart(sph_end)
    h_eff_end = heff(mag_params, t_end, m_cart_end)
    mxh = sp.cross(m_cart_end, h_eff_end)

    # Finite difference derivatives
    dhadt = (mag_params.Hvec(t_start) - mag_params.Hvec(t_end))/dt
    dedt = (llg_state_energy(sph_end, mag_params, t_end)
            - llg_state_energy(sph_start, mag_params, t_start)
            )/dt
    dmdt = (sp.array(utils.sph2cart(sph_start))
            - sp.array(m_cart_end))/dt

    utils.assert_almost_equal(dedt, sp.dot(m_cart_end, dhadt)
                              + sp.dot(dmdt, h_eff_end), 1e-2)

    # print(sp.dot(m_cart_end, dhadt), dedt)

    # Calculate alpha itself using the forumla derived in notes
    alpha = ( (dedt - sp.dot(m_cart_end, dhadt))
               /(sp.dot(h_eff_end, sp.cross(m_cart_end, dmdt))))

    return alpha

def recompute_alpha_varying_fields_at_midpoint(sph_start, sph_end,
                                               t_start, t_end, mag_params):
    """
    Compute effective damping from change in magnetisation and change in
    applied field.

    See notes 30/7/13 pg 5.

    Derivatives are estimated using midpoint method finite differences, all
    values are computed at the midpoint (m = (m_n + m_n-1)/2, similarly for
    t).
    """

    # Only for normalised problems!
    assert(mag_params.Ms == 1)

    # Get some values
    dt = t_end - t_start
    t = (t_end + t_start)/2
    m = (sp.array(utils.sph2cart(sph_end)) + sp.array(utils.sph2cart(sph_start)))/2

    h_eff = heff(mag_params, t, m)
    mxh = sp.cross(m, h_eff)

    # Finite difference derivatives
    dhadt = (mag_params.Hvec(t_end) - mag_params.Hvec(t_start))/dt
    dedt = (llg_state_energy(sph_end, mag_params, t_end)
            - llg_state_energy(sph_start, mag_params, t_start)
            )/dt
    dmdt = (sp.array(utils.sph2cart(sph_end))
            - sp.array(utils.sph2cart(sph_start)))/dt

    # utils.assert_almost_equal(dedt, sp.dot(m_cart_end, dhadt)
    #                           + sp.dot(dmdt, h_eff_end), 1e-2)

    # print(sp.dot(m_cart_end, dhadt), dedt)

    # Calculate alpha itself using the forumla derived in notes
    alpha = -( (dedt + sp.dot(m, dhadt))
               /(sp.dot(h_eff, sp.cross(m, dmdt))))

    return alpha


def recompute_alpha_list(m_sph_list, t_list, mag_params,
                         alpha_func=recompute_alpha):
    """Compute a list of effective dampings, one for each step in the input
    lists.
    """
    alpha_list = it.imap(alpha_func, m_sph_list, m_sph_list[1:],
                         t_list, t_list[1:], it.repeat(mag_params))
    return alpha_list


    #
# Testing
# ============================================================
def test_zeeman():
    """Test zeeman energy for some simple cases.
    """
    H_tests = [lambda t: sp.array([0, 0, 10]),
               lambda t: sp.array([-sqrt(2)/2, -sqrt(2)/2, 0.0]),
               lambda t: sp.array([0, 1, 0]),
               lambda t: sp.array([0.01, 0.0, 0.01]),
               ]

    m_tests = [(1.0, 0.0, 0.0),
               utils.cart2sph((sqrt(2)/2, sqrt(2)/2, 0.0)),
               (1, 0, 1),
               (0.0, 100.0, 0.0),
               ]

    answers = [lambda mP: -1 * mP.mu0 * mP.Ms * mP.H(None),
               lambda mP: mP.mu0 * mP.Ms * mP.H(None),
               lambda _:0.0,
               lambda _:0.0,
               ]

    for m, H, ans in zip(m_tests, H_tests, answers):
        yield check_zeeman, m, H, ans


def check_zeeman(m, H, ans):
    """Helper function for test_zeeman."""
    mag_params = utils.MagParameters()
    mag_params.Hvec = H
    utils.assert_almost_equal(zeeman_energy(m, mag_params),
                              ans(mag_params))


# See also mallinson.py: test_self_consistency for checks on alpha
# recomputation using Mallinson's exact solution.

# To test for applied fields we would need to do real time integration,
# which is a bit too large of a dependancy to have in a unit test, so do it
# somewhere else.
