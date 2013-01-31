
import collections
from math import sin, cos, tan, log, atan2, acos, pi, sqrt
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# General
# ============================================================

def unzip(iterable_of_iterables):
    """Inverse of zip. E.g. given a list of tuples returns a tuple of
    lists.

    To understand why: think about what * does to a list and what zip then
    does with this list.

    See http://www.shocksolution.com/2011/07/python-lists-to-tuples-and-tuples-to-lists/"""
    return zip(*iterable_of_iterables)


# Testing helpers
# ============================================================

# Some useful asserts. Use the assert command in each to make sure we get
# useful output from nose -d.
def almostEqual(a, b, tol = 1e-9): return abs(a - b) < tol
def assertAlmostEqual(a, b, tol = 1e-9): assert(almostEqual(a, b, tol))
def assertAlmostZero(a, tol = 1e-9): assert(abs(a) < tol)
def assertTupleAlmostEqual(tup1, tup2): map(assertAlmostEqual, tup1, tup2)

# Spherical polar coordinates asserts
def assertAziInRange(sph):
    assert(sph.azi > 0 or almostEqual(sph.azi, 0.0))
    assert(sph.azi < 2*pi or almostEqual(sph.azi, 2*pi))
def assertPolarInRange(sph):
    assert(sph.pol >= 0 and sph.pol <= pi)

# Coordinate systems
# ============================================================

# Some data structures
SphPoint = collections.namedtuple('SphPoint', ['r', 'azi', 'pol'])
CartPoint = collections.namedtuple('CartPoint', ['x', 'y', 'z'])

def cart2sph(cartesian_point):
    """
    Convert a 3D cartesian tuple into spherical polars.

    In the form (r,azi, pol) = (r, theta, phi) (following convention from
    mathworld).

    In Mallinson's notation this is (r, phi, theta).
    """
    x, y, z = cartesian_point

    r = sqrt(sum(map(lambda x: x**2, cartesian_point)))

    # Get azimuthal then shift from [-pi,pi] to [0,2pi]
    azi = atan2(y,x)
    if azi < 0: azi += 2*pi

    # Dodge the problem at central singular point...
    if r < 1e-9: polar = 0
    else: polar = acos(z/r)

    return SphPoint(r, azi, polar)

def sph2cart(spherical_point):
    """
    Convert a 3D spherical polar coordinate tuple into cartesian
    coordinates. See cart2sph(...) for spherical coordinate scheme."""

    r, azi, pol = spherical_point

    x = r * cos(azi) * sin(pol)
    y = r * sin(azi) * sin(pol)
    z = r * cos(pol)

    return CartPoint(x,y,z)

def array2sph(point_as_array):
    """ Convert from an array representation to a SphPoint.
    """

    assert point_as_array.ndim == 1

    # Hopefully 2 dims => spherical coords
    if point_as_array.shape[0] == 2:
        azi = point_as_array[0]
        pol = point_as_array[1]
        return SphPoint(1.0, azi, pol)

    # Presumably in cartesian...
    elif point_as_array.shape[0] == 3:
        return cart2sph(SphPoint(point_as_array[0],
                                 point_as_array[1],
                                 point_as_array[2]))

    else:
        raise IndexError


def plot_sph_points(sphs, title = 'Path of m'):
    carts = map(sph2cart, sphs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs,ys,zs = unzip(carts)
    ax.plot(xs,ys,zs)

    start_point = carts[1]
    ax.scatter(start_point.x, start_point.y, start_point.z)

    plt.title(title)

    return fig

def plot_polar_vs_time(sphs, times, title = 'Polar angle vs time'):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rs,azis,pols = unzip(sphs)
    ax.plot(times,pols)

    plt.xlabel('time/ arb. units')
    plt.ylabel('polar angle/ radians')
    plt.title(title)

    return fig


class MagParameters():

    def __init__(self):
        self.alpha = 1.0
        self.gamma = 1.0
        self.Hvec = (0.0, 0.0, -2.0)
        self.Hk = 0.0
        self.Ms = 1.0
        self.mu0 = 1.0

    def H(self):
        return sp.linalg.norm(self.Hvec, ord=2)

    # Hk = (2* K1) / (mu0* Ms) ??ds maybe....
    #??ds probably wrong!
    def K1(self):
        return self.Hk * self.mu0 * self.Ms / 2

    def __repr__(self):
        """Return a string representation of the parameters.
        """
        return "alpha = "+ str(self.alpha) \
          + ", gamma = "+ str(self.gamma) +",\n" \
          + "H = "+ str(self.Hvec) \
          + ", Hk = "+ str(self.Hk) \
          + ", Ms = "+ str(self.Ms)

def relative_error(exact, estimate):
    return abs(exact - estimate) / exact

# Test this file's code
# ============================================================

import unittest
from random import random
from scipy import linspace

class TestCoordinateConversion(unittest.TestCase):

    # Pick some coordinate lists to try out
    def setUp(self):
        def carttuple(x): return (x*random(), x*random(), x*random())
        self.carts = map(carttuple, linspace(0, 2, 20))
        self.sphs = map(cart2sph, self.carts)

    # Check that applying both operations gives back the same thing
    def check_cart_sph_composition(self, cart, sph):
         assertTupleAlmostEqual(cart, sph2cart(sph))
    def test_composition_is_identity(self):
        for (cart, sph) in zip(self.carts, self.sphs):
            self.check_cart_sph_composition(cart, sph)

    # Check that the azimuthal angle is in the correct range
    def test_azi_range(self):
        for sph in self.sphs: assertAziInRange(sph)
    def test_azimuthal_edge_cases(self):
        assertAlmostEqual(cart2sph((-1, -1, 0)).azi, 5*pi/4)

    # Check that the polar angle is in the correct range
    def test_polar_range(self):
        for sph in self.sphs: assertPolarInRange(sph)
