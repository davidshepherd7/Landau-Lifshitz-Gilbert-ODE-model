
import utils
import operator as o


def llg_state_energy(sph, llg_parameters):
    """ Assuming unit volume, spatially constant, spherical.

    Energies taken from [Coey2010].

    Ignore stress and magnetostriction.
    """
    return exchange_energy(sph, llg_parameters) \
      + magnetostatic_energy(sph, llg_parameters) \
      + magnetocrystalline_anisotropy_energy(sph, llg_parameters) \
      + zeeman_energy(sph, llg_parameters)


def exchange_energy(sph, llg_parameters):
    """Always zero because 0D/spatially constant."""
    return 0.0


# ??ds generalise!
def magnetostatic_energy(sph, llg_parameters):
    """ For a small, spherical particle:
    Ems = - 0.5 mu0 (M.Hms) = 0.5/3 * Ms**2
    """
    Ms = llg_parameters['Ms']
    mu0 = llg_parameters['mu0']
    energy = (0.5/3) * mu0 * Ms**2
    return energy


#??ds check the sign is correct
def magnetocrystalline_anisotropy_energy(sph, llg_parameters):
    """ Eca = K1 sin^2(polar)

    Assuming easy axis = z axis."""
    K1 = llg_parameters['K1']
    return K1 * (sin(sph.pol))**2


#??ds check the sign is correct
def zeeman_energy(sph, llg_parameters):
    """ Ez = - mu0 * (M.Happ)
    """
    Ms = llg_parameters['Ms']
    Happ = llg_parameters['H']
    mu0 = llg_parameters['mu0']

    sph.r = Ms
    M = sph2cart(sph)
    return - mu0 * sum(op.mul, M, H)


def recomputed_alpha(sph_start, sph_end, t_start, t_end, llg_parameters):
    """
    From a change in energy we can calculate what the effective damping
    was. For more details see [Albuquerque2001,
    http://dx.doi.org/10.1063/1.1355322].

    alpha' = - (1/(M**2))  *( dE/dt  /  spaceintegral( (dm/dt**2  )))

    No space integral is needed because we are working with 0D LLG.
    """

    Ms = llg_parameters['Ms']

    dE = llg_state_energy(sph_end, llg_parameters)
    - llg_state_energy(sph_start, llg_parameters)

    dmdtsq = [ ((m_endi - m_starti) / (t_end - t_start))**2 \
      for m_endi, m_starti in zip (utils.sph2cart(sph_end),\
                                   utils.sph2cart(sph_start))]

    return - (1/(Ms**2)) * ( dE / sum(dmdtsq))

    #

# Testing
# ============================================================

# See mallinson.py test_mallinson_self_consistency and
# test_non_negative_energy.
