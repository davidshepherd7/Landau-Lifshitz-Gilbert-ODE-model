
from math import sin, cos, tan, log, atan2, acos, pi, sqrt
import scipy as sp
import matplotlib.pyplot as plt

import utils

class MallinsonSolver(object):
    """Calculate analytical solutions to the zero dimensional LLG when H is
    aligned with the negative z direction and z is the easy axis.

    See [Mallinson2000, doi: 10.1109/20.875251] for details.
    """

    def __init__(self, _alpha = 0.5, _gamma = 1.0, _H = 2.0,
                 _Hk = 0.0, _starting_polar_angle = pi/18):
        self.alpha = _alpha
        self.gamma = _gamma
        self.H = _H  # Field in -z direction
        self.Hk = _Hk
        self.Ms = 1.0
        self.starting_polar_angle = _starting_polar_angle

    def parameter_string(self):
        """Return a string representation of the parameters."""
        return "alpha = "+ str(self.alpha) \
          + ", gamma = "+ str(self.gamma) +",\n" \
          + "H = "+ str(self.H) \
          + ", Hk = "+ str(self.Hk) \
          + " and start angle = "+ str(self.starting_polar_angle)

    def llg_parameters(self):
        """Return a dictionary of parameters."""
        return { 'alpha' : self.alpha,
                 'gamma' : self.gamma,
                 'H' : self.H,
                 'Hk' : self.Hk,
                 'Ms' : self.Ms}

    def time(self, polar_angle):
        """Calculate the time taken to switch from starting_polar_angle to
        polar_angle with the stored parameter set."""
        # Cache some things to simplify the expressions later
        H = self.H
        Hk = self.Hk
        p_start = self.starting_polar_angle
        p_now = polar_angle

        # Calculate the various parts of the expression
        prefactor = ((self.alpha**2 + 1)/(self.gamma * self.alpha)) \
                      * (1.0 / (H**2 - Hk**2))

        a = H * log( tan(p_now/2) / tan(p_start/2) )
        b = Hk * log( (H - Hk*cos(p_start)) /
                      (H - Hk*cos(p_now)) )
        c = Hk * log(sin(p_now) / sin(p_start))

        # Put everything together
        return prefactor * (a + b + c)

    def azi(self, polar_angle):
        """Calculate the azimuthal angle corresponding to switching from
        starting_polar_angle to polar angle with the stored parameter set."""
        def azi_into_range(azi):
            a = azi % (2*pi)
            if a < 0: a+= 2*pi
            return a

        no_range_azi = (-1/self.alpha) * log(tan(polar_angle/2)/
                                             tan(self.starting_polar_angle/2))
        return azi_into_range(no_range_azi)

    def generate_dynamics(self, end_angle = 17*pi/18, steps = 1000):
        """Generate a list of polar angles then return a list of
        corresponding m directions spherical polar coordinates and
        switching times."""

        pols = sp.linspace(self.starting_polar_angle, end_angle, steps)
        azis = map(self.azi, pols)
        rs = [1] * len(pols)

        sphs = map(utils.SphPoint, rs, azis, pols)
        times = map(self.time, pols)

        return (sphs, times)

    def plot_dynamics(self, end_angle = 17*pi/18, steps = 1000):
        sphs, times = self.generate_dynamics(end_angle, steps)

        sphstitle = "Path of m for " + self.parameter_string() \
          + "\n (starting point is marked)."
        utils.plot_sph_points(sphs, title = sphstitle)

        timestitle = "Polar angle vs time for " + self.parameter_string()
        utils.plot_polar_vs_time(sphs,times, title=timestitle)

        plt.show()



# Test this file's code
# ============================================================

import unittest
import energy

class MallinsonSolverCheckerBase():
    """Base class to define the test functions but not actually run them."""

    def base_init(self, alpha=0.5, Hk=0.0, steps=1000,
                  starting_polar_angle = pi/18):

        self.mSolver = MallinsonSolver(_alpha=alpha, _Hk=Hk,
                                       _starting_polar_angle=starting_polar_angle)
        (self.sphs, self.times) = self.mSolver.generate_dynamics(steps = steps)

        def partial_energy(sph):
            energy.llg_state_energy(sph, self.mSolver.llg_parameters)
        self.energys = map(partial_energy, self.sphs)

    # Monotonically increasing time
    def test_increasing_time(self):
        for a, b in zip(self.times, self.times[1:]):
            assert(b > a)

    # Azimuthal is in correct range
    def test_azimuthal_in_range(self):
        for sph in self.sphs: utils.assertAziInRange(sph)

    # Monotonically decreasing azimuthal angle except for jumps at
    # 2*pi. Closeness of point to 2*pi depends on time discretisation so we
    # give it some space, seems to work.
    def test_increasing_azimuthal(self):
        for a, b in zip(self.sphs, self.sphs[1:]):
            assert(a.azi > b.azi or
                   (b.azi + a.azi - 2 * pi) < 3.0/(len(self.sphs)))


    def test_mallinson_self_consistency(self):
        a2 = energy.recomputed_alpha(self.sphs[0], self.sphs[-1],
                                     self.times[0], self.times[-1],
                                     self.mSolver.llg_parameters())
        assert(abs(a2 - self.mSolver.alpha) < 1e-6)

    def test_non_negative_energy(self):
        assert(all(e > 0 for e in self.energys))


# Now run the tests with various intial settings (tests are inherited from
# the base class.
class TestMallinsonDefaults(MallinsonSolverCheckerBase, unittest.TestCase):
    def setUp(self):
        self.base_init(steps=10000)

class TestMallinsonHk(MallinsonSolverCheckerBase, unittest.TestCase):
    def setUp(self):
        self.base_init(Hk = 0.9)

class TestMallinsonStartAngle(MallinsonSolverCheckerBase,
                                     unittest.TestCase):
    def setUp(self):
        self.base_init(starting_polar_angle = pi/2)
