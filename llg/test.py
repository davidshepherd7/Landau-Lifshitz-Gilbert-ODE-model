from __future__ import division
from __future__ import absolute_import


import functools as ft
import matplotlib.pyplot as plt
import unittest
import operator as op
import scipy as sp
import itertools as it

from math import sin, cos, tan, log, atan2, acos, pi, sqrt

import simpleode.core.utils as utils
import simpleode.core.ode as ode
import simpleode.llg.mallinson as mlsn
import simpleode.llg.energy as energy
import simpleode.llg.llg as llg


# We have to split out these tests into their own file because otherwise we
# get circular dependencies.


# Mallinson
# ============================================================

class MallinsonSolverCheckerBase():
    """Base class to define the test functions but not actually run them.
    """

    def base_init(self, magParameters=None, steps=1000,
                  p_start=pi/18):

        if magParameters is None:
            self.mag_params = utils.MagParameters()
        else:
            self.mag_params = magParameters

        (self.sphs, self.times) = mlsn.generate_dynamics(
            self.mag_params, steps=steps)

        def f(sph): energy.llg_state_energy(sph, self.mag_params)
        self.energys = map(f, self.sphs)

    # Monotonically increasing time
    def test_increasing_time(self):
        print(self.mag_params.Hvec)
        for a, b in zip(self.times, self.times[1:]):
            assert(b > a)

    # Azimuthal is in correct range
    def test_azimuthal_in_range(self):
        for sph in self.sphs:
            utils.assert_azi_in_range(sph)

    # Monotonically decreasing azimuthal angle except for jumps at 2*pi.
    def test_increasing_azimuthal(self):
        for a, b in zip(self.sphs, self.sphs[1:]):
            assert(a.azi > b.azi or
                   (a.azi - 2*pi <= 0.0 and b.azi >= 0.0))

    def test_damping_self_consistency(self):
        a2s = energy.recompute_alpha_list(self.sphs, self.times,
                                          self.mag_params)

        # Check that we get the same values with the varying fields version
        a3s = energy.recompute_alpha_list(self.sphs, self.times,
                                          self.mag_params,
                                          energy.recompute_alpha_varying_fields)
        utils.assert_list_almost_equal( a2s, a3s, (1.1/len(self.times)))
        # one of the examples doesn't quite pass with tol=1.0/len, so use
        # 1.1

        # Use 1/length as error estimate because it's proportional to dt
        # and so proportional to the expected error
        def check_alpha_ok(a2):
            return abs(a2 - self.mag_params.alpha) < (1.0/len(self.times))
        assert(all(map(check_alpha_ok, a2s)))

    # This is an important test. If this works then it is very likely that
    # the Mallinson calculator, the energy calculations and most of the
    # utils (so far) are all working. So tag it as "core".
    test_damping_self_consistency.core = True



# Now run the tests with various intial settings (tests are inherited from
# the base class.
class TestMallinsonDefaults(MallinsonSolverCheckerBase, unittest.TestCase):
    def setUp(self):
        self.base_init() # steps=10000) ??ds


class TestMallinsonHk(MallinsonSolverCheckerBase, unittest.TestCase):
    def setUp(self):
        mag_params = utils.MagParameters()
        mag_params.K1 = 0.6
        self.base_init(mag_params)


class TestMallinsonLowDamping(MallinsonSolverCheckerBase, unittest.TestCase):
    def setUp(self):
        mag_params = utils.MagParameters()
        mag_params.alpha = 0.1
        self.base_init(mag_params) # , steps=10000) ??ds


class TestMallinsonStartAngle(MallinsonSolverCheckerBase,
                              unittest.TestCase):
    def setUp(self):
        self.base_init(p_start=pi/2)





# llg.py
# ============================================================

# ??ds replace with substituting the exact solution into the residual and
# checking it is zero?


def test_llg_residuals():
    m0_sph = [0.0, pi/18]
    m0_cart = utils.sph2cart(tuple([1.0] + m0_sph))
    # m0_constrained = list(m0_cart) + [None]  # ??ds

    residuals = [(llg.llg_cartesian_residual, m0_cart),
                 (llg.ll_residual, m0_cart),
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
    Hvec = magnetic_parameters.Hvec(None)
    Ms = magnetic_parameters.Ms

    h_eff = Hvec
    dmdt_cart = (gamma/(1+alpha**2)) * sp.cross(m_cart, h_eff) \
        - (alpha*gamma/((1+alpha**2)*Ms)) * sp.cross(m_cart, sp.cross(
                                                     m_cart, h_eff))

    # Calculate with function
    dfdm_func = llg.llg_cartesian_dfdm(magnetic_parameters, t, m_cart, dmdt_cart)

    def f(t, m_cart, dmdt_cart):
        # f is the residual + dm/dt (see notes 27/2/13)
        return llg.llg_cartesian_residual(magnetic_parameters,
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




# energy.py
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
    utils.assert_almost_equal(energy.zeeman_energy(m, mag_params),
                              ans(mag_params))


# See also mallinson.py: test_self_consistency for checks on alpha
# recomputation using Mallinson's exact solution.

# To test for applied fields we would need to do real time integration,
# which is a bit too large of a dependancy to have in a unit test, so do it
# somewhere else.
