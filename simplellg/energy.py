"""Calculate energies for a single magnetisation vector in a field.
"""

from __future__ import division
import operator as op
from math import sin, cos, tan, log, atan2, acos, pi, sqrt
import scipy as sp

import simplellg.utils as utils


def llg_state_energy(sph, mag_params):
    """Assuming unit volume, spatially constant, spherical particle.

    Energies taken from [Coey2010].

    Ignore stress and magnetostriction.
    """
    return exchange_energy(sph, mag_params) \
        + magnetostatic_energy(sph, mag_params) \
        + magnetocrystalline_anisotropy_energy(sph, mag_params) \
        + zeeman_energy(sph, mag_params)


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


def zeeman_energy(sph, mag_params):
    """ Ez = - mu0 * (M.Happ)
    """
    Ms = mag_params.Ms
    Happ = mag_params.Hvec
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
    dE = llg_state_energy(sph_end, mag_params) \
        - llg_state_energy(sph_start, mag_params)
    dEdt = dE / dt

    # Estimate dMagentisation / dTime then take sum of squares
    dm = [m2 - m1 for m1, m2 in
          zip(utils.sph2cart(sph_start), utils.sph2cart(sph_end))]
    dmdt_sq_sum = sum([(dm_i / dt)**2 for dm_i in dm])

    # dE should be negative so the result should be positive.
    return - (1/(Ms**2)) * (dEdt / dmdt_sq_sum)


def recompute_alpha_list(m_sph_list, t_list, mag_params):
    """Compute a list of effective dampings, one for each step in the input
    lists.
    """

    alpha_list = []
    for m_start, m_end, t_start, t_end in \
            zip(m_sph_list, m_sph_list[1:], t_list, t_list[1:]):
        a = recompute_alpha(m_start, m_end, t_start, t_end, mag_params)
        alpha_list.append(a)

    return alpha_list


    #
# Testing
# ============================================================
def test_zeeman():
    """Test zeeman energy for some simple cases.
    """
    H_tests = [(0, 0, 10),
               (-sqrt(2)/2, -sqrt(2)/2, 0.0),
               (0, 1, 0),
               (0.01, 0.0, 0.01),
               ]

    m_tests = [(1.0, 0.0, 0.0),
               utils.cart2sph((sqrt(2)/2, sqrt(2)/2, 0.0)),
               (1, 0, 1),
               (0.0, 100.0, 0.0),
               ]

    answers = [lambda mP: -1 * mP.mu0 * mP.Ms * mP.H(),
               lambda mP: mP.mu0 * mP.Ms * mP.H(),
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


# See also mallinson.py: test_self_consistency
