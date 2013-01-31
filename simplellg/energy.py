
import operator as op
from math import sin, cos, tan, log, atan2, acos, pi, sqrt
import scipy as sp

import simplellg.utils as utils


def llg_state_energy(sph, magParameters):
    """ Assuming unit volume, spatially constant, spherical.

    Energies taken from [Coey2010].

    Ignore stress and magnetostriction.
    """
    return exchange_energy(sph, magParameters) \
      + magnetostatic_energy(sph, magParameters) \
      + magnetocrystalline_anisotropy_energy(sph, magParameters) \
      + zeeman_energy(sph, magParameters)


def exchange_energy(sph, magParameters):
    """Always zero because 0D/spatially constant."""
    return 0.0


# ??ds generalise!
def magnetostatic_energy(sph, magParameters):
    """ For a small, spherical particle:
    Ems = - 0.5 mu0 (M.Hms) = 0.5/3 * Ms**2
    """
    Ms = magParameters.Ms
    mu0 = magParameters.mu0
    return (0.5/3) * mu0 * Ms**2


#??ds check the sign is correct
def magnetocrystalline_anisotropy_energy(sph, magParameters):
    """ Eca = K1 sin^2(polar)

    Assuming easy axis = z axis."""
    K1 = magParameters.K1()
    return K1 * (sin(sph.pol))**2


def zeeman_energy(sph, magParameters):
    """ Ez = - mu0 * (M.Happ)
    """
    Ms = magParameters.Ms
    Happ = magParameters.Hvec
    mu0 = magParameters.mu0

    m = utils.sph2cart(sph)
    return -1 * mu0 * Ms * sp.dot(m, Happ)


def recompute_alpha(sph_start, sph_end, t_start, t_end, magParameters):
    """
    From a change in energy we can calculate what the effective damping
    was. For more details see [Albuquerque2001,
    http://dx.doi.org/10.1063/1.1355322].

    alpha' = - (1/(M**2))  *( dE/dt  /  spaceintegral( (dm/dt**2  )))

    No space integral is needed because we are working with 0D LLG.
    """

    Ms = magParameters.Ms
    dt = t_end - t_start

    # Estimate dEnergy / dTime
    dE = llg_state_energy(sph_end, magParameters) \
        - llg_state_energy(sph_start, magParameters)
    dEdt = dE / dt

    # Estimate dMagentisation / dTime then take sum of squares
    dm = map(op.sub, utils.sph2cart(sph_start), utils.sph2cart(sph_end))
    dmdt = [dm_i / dt for dm_i in dm]
    def square(x): return pow(x,2)
    dmdt_sq_sum = sum(map(square, dmdt))

    # dE should be negative so the result should be positive.
    return - (1/(Ms**2)) * ( dEdt / dmdt_sq_sum)


def recompute_alpha_list(m_sph_list, t_list, magParameters):
    """Compute a list of effective dampings, one for each step in the input
    lists.
    """

    alpha_list = []
    for m_start, m_end, t_start, t_end in \
            zip(m_sph_list, m_sph_list[1:], t_list, t_list[1:]):
        a = recompute_alpha(m_start, m_end, t_start, t_end, magParameters)
        alpha_list.append(a)

    return alpha_list



    #

# Testing
# ============================================================

def test_zeeman():
    H_tests = [(0,0,10),
          (-sqrt(2)/2, -sqrt(2)/2, 0.0),
          (0,1,0),
          (0.01,0.0,0.01),
          ]

    m_tests = [(1.0, 0.0, 0.0),
          utils.cart2sph((sqrt(2)/2, sqrt(2)/2, 0.0)),
          (1,0,1),
          (0.0,100.0,0.0),
          ]

    answers = [lambda mP: -1 * mP.mu0 * mP.Ms * mP.H(),
               lambda mP: mP.mu0 * mP.Ms * mP.H(),
               lambda _:0.0,
               lambda _:0.0,
               ]

    for m, H, ans in zip(m_tests, H_tests, answers):
        yield check_zeeman, m, H, ans

def check_zeeman(m, H, ans):
    mP = utils.MagParameters()
    mP.Hvec = H
    ze = zeeman_energy(m,mP)
    utils.assertAlmostEqual(ze,ans(mP))


# See also mallinson.py: test_self_consistency
