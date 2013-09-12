
from __future__ import division
import collections
from math import sin, cos, tan, log, atan2, acos, pi, sqrt
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools as it
import operator as op
import functools as ft

from functools import partial as par
from os.path import join as pjoin

# General
# ============================================================


def unzip(iterable_of_iterables):
    """Inverse of zip. E.g. given a list of tuples returns a tuple of
    lists.

    To understand why: think about what * does to a list and what zip then
    does with this list.

    See http://www.shocksolution.com/2011/07/python-lists-to-tuples-and-tuples-to-lists/"""
    return zip(*iterable_of_iterables)


def _apply_to_list_and_print_args(function, list_of_args):
    """Does what it says. Should really be a lambda function but
    multiprocessing requires named functions
    """
    print list_of_args
    return function(*list_of_args)


def parallel_parameter_sweep(function, parameter_lists, serial_mode=False):
    """Run function with all combinations of parameters in parallel using
    all available cores.

    parameter_lists should be a list of lists of parameters,
    """

    import multiprocessing

    # Generate a complete set of combinations of parameters
    parameter_sets = it.product(*parameter_lists)

    # multiprocessing doesn't include a "starmap", requires all functions
    # to take a single argument. Use a function wrapper to fix this. Also
    # print the list of args while we're in there.
    wrapped_function = par(_apply_to_list_and_print_args, function)

    # For debugging we often need to run in serial (to get useful stack
    # traces).
    if serial_mode:
        results_iterator = it.imap(wrapped_function, parameter_sets)
         # Force evaluation (to be exactly the same as in parallel)
        results_iterator = list(results_iterator)

    else:
        # Run in all parameter sets in parallel
        pool = multiprocessing.Pool()
        results_iterator = pool.imap_unordered(wrapped_function, parameter_sets)
        pool.close()

        # wait for everything to finish
        pool.join()

    return results_iterator


def partial_lists(l, min_list_length=1):
    """Given a list l return a list of "partial lists" (probably not the
    right term...).

    Optionally specify a minimum list length.

    ie.

    l = [0, 1, 2, 3]

    partial_lists(l) = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
    """
    all_lists = [l[:i] for i in range(0, len(l)+1)]
    return filter(lambda x: len(x) >= min_list_length, all_lists)


def myfigsave(figure, name, texpath="/home/david/Dropbox/phd/reports/ongoing-writeup/images"):
    """Fix up layout and save a pdf of an image into my latex folder.
    """

    # Fix layout
    figure.tight_layout(pad=0.3)

    # Save a pdf into my tex image dir
    figpath = pjoin(texpath, name)
    figure.savefig(figpath, dpi=300, orientation='portrait',
                   transparent=False)

    print "Saved to", figpath
    return

def memoize(f):
    """ Memoization decorator for a function taking multiple arguments.

    From http://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/
    (in the comments)
    """
    class memodict(dict):
        def __init__(self, f):
            self.f = f
        def __call__(self, *args):
            return self[args]
        def __missing__(self, key):
            ret = self[key] = self.f(*key)
            return ret
    return memodict(f)


def latex_escape(s):
    """Escape all characters that latex will cry about.
    """
    s = s.replace(r'{', r'\{')
    s = s.replace(r'}', r'\}')
    s = s.replace(r'&', r'\&')
    s = s.replace(r'%', r'\%')
    s = s.replace(r'$', r'\$')
    s = s.replace(r'#', r'\#')
    s = s.replace(r'_', r'\_')
    s = s.replace(r'^', r'\^{}')

    # Can't handle backslashes... ?

    return s


# Testing helpers
# ============================================================


def almost_equal(a, b, tol=1e-9):
    return abs(a - b) < tol


def abs_list_diff(list_a, list_b):
    return [abs(a - b) for a, b in zip(list_a, list_b)]


def list_almost_zero(list_x, tol=1e-9):
    return max(list_x) < tol


def list_almost_equal(list_a, list_b, tol=1e-9):
    return list_almost_zero(abs_list_diff(list_a, list_b), tol)


# Some useful asserts. We explicitly use the assert command in each
# (instead of defining the almost equal commands in terms of each
# other) to make sure we get useful output from nose -d.
# ??ds fix the names to use _ and lower case?
def assert_almost_equal(a, b, tol=1e-9):
    assert(abs(a - b) < tol)


def assert_almost_zero(a, tol=1e-9):
    assert(abs(a) < tol)


def assert_list_almost_equal(list_a, list_b, tol=1e-9):
    assert(len(list(list_a)) == len(list(list_b)))
    for a, b in zip(list_a, list_b):
        assert(abs(a - b) < tol)


def assert_list_almost_zero(values, tol=1e-9):
    for x in values:
        assert abs(x) < tol


def assert_sym_eq(a, b):
    """Compare symbolic expressions. Note that the simplification algorithm
    is not completely robust: might give false negatives (but never false
    positives).

    Try adding extra simplifications if needed, e.g. add .trigsimplify() to
    the end of my_simp.
    """

    def my_simp(expr):
        # Can't .expand() ints, so catch the zero case separately.
        try:
            return expr.expand().simplify()
        except AttributeError:
            return expr

    print
    print sympy.pretty(my_simp(a))
    print "equals"
    print sympy.pretty(my_simp(b))
    print

    # Try to simplify the difference to zero
    assert (my_simp(a - b) == 0)


# Spherical polar coordinates asserts
def assert_azi_in_range(sph):
    assert(sph.azi > 0 or almost_equal(sph.azi, 0.0))
    assert(sph.azi < 2*pi or almost_equal(sph.azi, 2*pi))


def assert_polar_in_range(sph):
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

    r = sp.linalg.norm(cartesian_point, 2)

    # Get azimuthal then shift from [-pi,pi] to [0,2pi]
    azi = atan2(y, x)
    if azi < 0:
        azi += 2*pi

    # Dodge the problem at central singular point...
    if r < 1e-9:
        polar = 0
    else:
        polar = acos(z/r)

    return SphPoint(r, azi, polar)


def sph2cart(spherical_point):
    """
    Convert a 3D spherical polar coordinate tuple into cartesian
    coordinates. See cart2sph(...) for spherical coordinate scheme."""

    r, azi, pol = spherical_point

    x = r * cos(azi) * sin(pol)
    y = r * sin(azi) * sin(pol)
    z = r * cos(pol)

    return CartPoint(x, y, z)


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


def plot_sph_points(sphs, title='Path of m'):
    carts = map(sph2cart, sphs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the path
    xs, ys, zs = unzip(carts)
    ax.plot(xs, ys, zs)

    # Draw on the starting point
    start_point = carts[0]
    ax.scatter(start_point.x, start_point.y, start_point.z)

    # Draw on z-axis
    ax.plot([0, 0], [0, 0], [-1, 1], '--')

    plt.title(title)

    # Axes
    ax.set_zlim(-1, 1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    return fig


def plot_polar_vs_time(sphs, times, title='Polar angle vs time'):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rs, azis, pols = unzip(sphs)
    ax.plot(times, pols)

    plt.xlabel('time/ arb. units')
    plt.ylabel('polar angle/ radians')
    plt.title(title)

    return fig


class MagParameters():

    def Hvec(self, t):
        return sp.array([0,0,-2])

    gamma = 1.0
    K1 = 0.0
    Ms = 1.0
    mu0 = 1.0
    easy_axis = sp.array([0, 0, 1])


    def __init__(self, alpha=1.0):
        self.alpha = alpha


    def dimensional_H(self, t):
        return sp.linalg.norm(self.Hvec(t), ord=2)


    def H(self, t):
        return sp.linalg.norm(self.Hvec(t)/self.Ms, ord=2)


    def dimensional_Hk(self):
        """Ansiotropy field strength."""
        # ??ds if m is always unit vector then this is right, if not we
        # need extra factor of Ms on bottom...
        return (2 * self.K1) / (self.mu0 * self.Ms)


    def Hk(self):
        """Ansiotropy field strength."""
        # ??ds if m is always unit vector then this is right, if not we
        # need extra factor of Ms on bottom...
        return self.dimensional_Hk() / self.Ms


    def dimensional_Hk_vec(self, m_cart):
        """Uniaxial anisotropy field. Magnetisation should be in normalised
        cartesian form."""
        return self.dimensional_Hk() * sp.dot(m_cart, self.easy_axis) * self.easy_axis


    def Hk_vec(self, m_cart):
        """Normalised uniaxial anisotropy field. Magnetisation should be in
        normalised cartesian form."""
        return self.dimensional_Hk_vec(m_cart) / self.Ms


    def __repr__(self):
        """Return a string representation of the parameters.
        """
        return "alpha = " + str(self.alpha) \
            + ", gamma = " + str(self.gamma) + ",\n" \
            + "H(0) = " + str(self.Hvec(0)) \
            + ", K1 = " + str(self.K1) \
            + ", Ms = " + str(self.Ms)


# Smaller helper functions
# ============================================================


def relative_error(exact, estimate):
    return abs(exact - estimate) / exact


def dts_from_ts(ts):
    return list(map(op.sub, ts[1:], ts[:-1]))

ts2dts = dts_from_ts


def ts2dtn(ts):
    return ts[-1] - ts[-2]


def ts2dtnm1(ts):
    return ts[-2] - ts[-3]


# Matrices
# ============================================================

def skew(vector_with_length_3):
    v = vector_with_length_3

    if len(v) != 3:
        raise TypeError("skew is only defined for vectors of length 3")

    return sp.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

# Test this file's code
# ============================================================

import unittest
from random import random
import nose.tools as nt


class TestCoordinateConversion(unittest.TestCase):

    # Pick some coordinate lists to try out
    def setUp(self):
        def carttuple(x):
            return (x*random(), x*random(), x*random())
        self.carts = map(carttuple, sp.linspace(0, 2, 20))
        self.sphs = map(cart2sph, self.carts)

    # Check that applying both operations gives back the same thing
    def check_cart_sph_composition(self, cart, sph):
        assert_list_almost_equal(cart, sph2cart(sph))

    def test_composition_is_identity(self):
        for (cart, sph) in zip(self.carts, self.sphs):
            self.check_cart_sph_composition(cart, sph)

    # Check that the azimuthal angle is in the correct range
    def test_azi_range(self):
        for sph in self.sphs:
            assert_azi_in_range(sph)

    def test_azimuthal_edge_cases(self):
        assert_almost_equal(cart2sph((-1, -1, 0)).azi, 5*pi/4)

    # Check that the polar angle is in the correct range
    def test_polar_range(self):
        for sph in self.sphs:
            assert_polar_in_range(sph)


def example_f(x, y):
    return cos(x) + sin(y)


def test_parallel_sweep():
    """Check that a parallel run gives the same results as a non-parallel
    run for a simple function.
    """
    xs = sp.linspace(-pi, +pi, 30)
    ys = sp.linspace(-pi, +pi, 30)

    parallel_result = list(parallel_parameter_sweep(example_f, [xs, ys]))
    serial_result = list(parallel_parameter_sweep(example_f, [xs, ys], True))
    exact_result = list(it.starmap(example_f, it.product(xs, ys)))

    # Use sets for the comparison because the parallel computation destroys
    # any ordering we had before (and sets order their elements).
    assert_list_almost_equal(set(parallel_result), set(exact_result))
    assert_list_almost_equal(set(serial_result), set(exact_result))


def test_skew_size_check():
    xs = [sp.linspace(0.0, 1.0, 4), 1.0, sp.identity(3)]
    for x in xs:
        nt.assert_raises(TypeError, skew, [x])


def test_skew():
    xs = [sp.linspace(0.0, 1.0, 3), sp.zeros((3, 1)), [1, 2, 3], ]
    a = sp.rand(3)

    for x in xs:
        # Anything crossed with itself is zero:
        skew_mat = skew(x)
        assert_list_almost_zero(sp.dot(skew_mat, sp.array(x)))

        # a x b = - b x a
        assert_list_almost_zero(sp.dot(skew_mat, a) + sp.dot(a, skew_mat))
