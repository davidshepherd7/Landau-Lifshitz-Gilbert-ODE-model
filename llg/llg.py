"""Various residuals for the LLG equation
"""

from __future__ import division
from __future__ import absolute_import

from math import sin, cos, tan, log, atan2, acos, pi, sqrt
import scipy as sp

import simpleode.core.utils as utils
import simpleode.core.ode as ode


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


def heff(magnetic_parameters, t, m_cart):
    Hk_vec = magnetic_parameters.Hk_vec(m_cart)
    h_eff = magnetic_parameters.Hvec(t) + Hk_vec # - ((1.0/3)*sp.array(m_cart))

    return h_eff


def llg_cartesian_residual(magnetic_parameters, t, m_cart, dmdt_cart):

    # Extract the parameters
    alpha = magnetic_parameters.alpha
    gamma = magnetic_parameters.gamma
    Ms = magnetic_parameters.Ms
    h_eff = heff(magnetic_parameters, t, m_cart)

    residual = ((alpha/Ms) * sp.cross(m_cart, dmdt_cart)
                - gamma * sp.cross(m_cart, h_eff)
                - dmdt_cart)
    return residual


def llg_cartesian_dfdm(magnetic_parameters, t, m_cart, dmdt_cart):

    # Extract the parameters
    alpha = magnetic_parameters.alpha
    gamma = magnetic_parameters.gamma
    Ms = magnetic_parameters.Ms

    h_eff = heff(magnetic_parameters, t, m_cart)

    dfdm = - gamma * utils.skew(h_eff) + (alpha/Ms) * utils.skew(dmdt_cart)

    return dfdm


def ll_dmdt(magnetic_parameters, t, m):
    alpha = magnetic_parameters.alpha
    h_eff = heff(magnetic_parameters, t, m)

    return -1/(1 + alpha**2) *(sp.cross(m, h_eff) + alpha*sp.cross(m, sp.cross(m, h_eff)))


def ll_residual(magnetic_parameters, t, m, dmdt):
    return dmdt - ll_dmdt(magnetic_parameters, t, m)


def simple_llg_residual(t, m, dmdt):
    mp = utils.MagParameters()
    return llg_cartesian_residual(mp, t, m, dmdt)

def simple_llg_initial(*_):
    return utils.sph2cart([1.0, 0.0, sp.pi/18])


def linear_H(t):
    return sp.array([0,0,-0.5*t])

def ramping_field_llg_residual(t, m, dmdt):
    mp = utils.MagParameters()
    mp.Hvec = linear_H
    mp.alpha = 0.1
    return llg_cartesian_residual(mp, t, m, dmdt)

def ramping_field_llg_initial(*_):
    return utils.sph2cart([1.0, 0.0, sp.pi/18])


# def llg_constrained_cartesian_residual(magnetic_parameters, t, m_cart,
#                                        dmdt_cart):

#     base_residual = llg_cartesian_residual(magnetic_parameters, t, m_cart[:-1],
#                                            dmdt_cart[:-1])

#     pass
