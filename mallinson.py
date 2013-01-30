
from math import sin, cos, tan, log, atan2, acos, pi, sqrt
import scipy as sp
import matplotlib.pyplot as plt

import utils


def calculate_switching_time(magnetic_parameters, p_start, p_now):
    """Calculate the time taken to switch from polar angle p_start to p_now
    with the magnetic parameters given.
    """
    # Cache some things to simplify the expressions later
    H = magnetic_parameters.H()
    Hk = magnetic_parameters.Hk
    alpha = magnetic_parameters.alpha
    gamma = magnetic_parameters.gamma

    # Calculate the various parts of the expression
    prefactor = ((alpha**2 + 1)/(gamma * alpha)) \
                  * (1.0 / (H**2 - Hk**2))

    a = H * log( tan(p_now/2) / tan(p_start/2) )
    b = Hk * log( (H - Hk*cos(p_start)) /
                  (H - Hk*cos(p_now)) )
    c = Hk * log(sin(p_now) / sin(p_start))

    # Put everything together
    return prefactor * (a + b + c)

def calculate_azimuthal(magnetic_parameters, p_start, p_now):
    """Calculate the azimuthal angle corresponding to switching from
    p_start to p_now with the magnetic parameters given.
    """
    def azi_into_range(azi):
        a = azi % (2*pi)
        if a < 0: a+= 2*pi
        return a

    alpha = magnetic_parameters.alpha

    no_range_azi = (-1/alpha) * log(tan(p_now/2) / tan(p_start/2))
    return azi_into_range(no_range_azi)

def generate_dynamics(magnetic_parameters,
                      start_angle = pi/18,
                      end_angle = 17*pi/18,
                      steps = 1000):
    """Generate a list of polar angles then return a list of corresponding
    m directions (in spherical polar coordinates) and switching times.
    """
    mp = magnetic_parameters

    pols = sp.linspace(start_angle, end_angle, steps)
    azis = [calculate_azimuthal(mp, start_angle, p) for p in pols]
    rs = [1.0] * len(pols) # r is always 1.0

    sphs = map(utils.SphPoint, rs, azis, pols) # wrap into a named tuple
    times = [calculate_switching_time(mp, start_angle, p) for p in pols]

    return (sphs, times)

def plot_dynamics(magnetic_parameters,
                  start_angle = pi/18,
                  end_angle = 17*pi/18,
                  steps = 1000):

    sphs, times = generate_dynamics(magnetic_parameters, start_angle,
                                    end_angle, steps)

    sphstitle = "Path of m for " + str(magnetic_parameters) \
      + "\n (starting point is marked)."
    utils.plot_sph_points(sphs, title = sphstitle)

    timestitle = "Polar angle vs time for " + str(magnetic_parameters)
    utils.plot_polar_vs_time(sphs,times, title=timestitle)

    plt.show()



# Test this file's code
# ============================================================

import unittest
import energy

class MallinsonSolverCheckerBase():
    """Base class to define the test functions but not actually run them."""

    def base_init(self, magParameters = None, steps = 1000,
                  p_start = pi/18):

        # self.mSolver = MallinsonSolver(magParameters,
        #                                p_start=p_start)

        if magParameters is None:
            self.mP = utils.MagParameters()
        else:
            self.mP = magParameters

        (self.sphs, self.times) = generate_dynamics(self.mP, steps = steps)

        def partial_energy(sph):
            energy.llg_state_energy(sph, self.mP)
        self.energys = map(partial_energy, self.sphs)

    # Monotonically increasing time
    def test_increasing_time(self):
        for a, b in zip(self.times, self.times[1:]):
            assert(b > a)

    # Azimuthal is in correct range
    def test_azimuthal_in_range(self):
        for sph in self.sphs: utils.assertAziInRange(sph)

    # Monotonically decreasing azimuthal angle except for jumps at 2*pi.
    def test_increasing_azimuthal(self):
        for a, b in zip(self.sphs, self.sphs[1:]):
            assert(a.azi > b.azi or
                   (a.azi - 2*pi <= 0.0 and b.azi >= 0.0))

    def test_damping_self_consistency(self):
        a2s = energy.recompute_alpha_list(self.sphs, self.times,
                                          self.mP)

        # Use 1/length as error estimate because it's proportional to dt.
        def check_alpha_ok(a2):
            return abs(a2 - self.mP.alpha) < 1.0/len(self.times)
        assert(all(map(check_alpha_ok, a2s)))

    # This is an important test. If this works then it is very likely that
    # the Mallinson calculator, the energy calculations and most of the
    # utils (so far) are all working. So tag it as "core".
    test_damping_self_consistency.core = True

    # def test_non_negative_energy(self):
    #     assert(all(e > 0 for e in self.energys))


# Now run the tests with various intial settings (tests are inherited from
# the base class.
class TestMallinsonDefaults(MallinsonSolverCheckerBase, unittest.TestCase):
    def setUp(self):
        self.base_init(steps=10000)

class TestMallinsonHk(MallinsonSolverCheckerBase, unittest.TestCase):
    def setUp(self):
        mP = utils.MagParameters()
        mP.Hk = 1.2
        self.base_init(mP)

class TestMallinsonLowDamping(MallinsonSolverCheckerBase, unittest.TestCase):
    def setUp(self):
        mP = utils.MagParameters()
        mP.alpha = 0.1
        self.base_init(mP, steps=10000)

class TestMallinsonStartAngle(MallinsonSolverCheckerBase,
                                     unittest.TestCase):
    def setUp(self):
        self.base_init(p_start = pi/2)
