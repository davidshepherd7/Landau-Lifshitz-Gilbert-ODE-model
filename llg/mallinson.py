"""Calculate exact solutions for the zero dimensional LLG as given by
[Mallinson2000]
"""

from __future__ import division
from __future__ import absolute_import

from math import sin, cos, tan, log, atan2, acos, pi, sqrt
import scipy as sp
import matplotlib.pyplot as plt
import functools as ft

import simpleode.core.utils as utils


def calculate_switching_time(magnetic_parameters, p_start, p_now):
    """Calculate the time taken to switch from polar angle p_start to p_now
    with the magnetic parameters given.
    """

    # Should never quite get to pi/2
    # if p_now >= pi/2:
    #     return sp.inf

    # Cache some things to simplify the expressions later
    H = magnetic_parameters.H(None)
    Hk = magnetic_parameters.Hk()
    alpha = magnetic_parameters.alpha
    gamma = magnetic_parameters.gamma

    # Calculate the various parts of the expression
    prefactor = ((alpha**2 + 1)/(gamma * alpha)) \
        * (1.0 / (H**2 - Hk**2))

    a = H * log(tan(p_now/2) / tan(p_start/2))
    b = Hk * log((H - Hk*cos(p_start)) /
                (H - Hk*cos(p_now)))
    c = Hk * log(sin(p_now) / sin(p_start))

    # Put everything together
    return prefactor * (a + b + c)


def calculate_azimuthal(magnetic_parameters, p_start, p_now):
    """Calculate the azimuthal angle corresponding to switching from
    p_start to p_now with the magnetic parameters given.
    """
    def azi_into_range(azi):
        a = azi % (2*pi)
        if a < 0:
            a += 2*pi
        return a

    alpha = magnetic_parameters.alpha

    no_range_azi = (-1/alpha) * log(tan(p_now/2) / tan(p_start/2))
    return azi_into_range(no_range_azi)


def generate_dynamics(magnetic_parameters,
                      start_angle=pi/18,
                      end_angle=17*pi/18,
                      steps=1000):
    """Generate a list of polar angles then return a list of corresponding
    m directions (in spherical polar coordinates) and switching times.
    """
    mag_params = magnetic_parameters

    # Construct a set of solution positions
    pols = sp.linspace(start_angle, end_angle, steps)
    azis = [calculate_azimuthal(mag_params, start_angle, p) for p in pols]
    sphs = [utils.SphPoint(1.0, azi, pol) for azi, pol in zip(azis, pols)]

    # Calculate switching times for these positions
    times = [calculate_switching_time(mag_params, start_angle, p)
             for p in pols]

    return (sphs, times)


def plot_dynamics(magnetic_parameters,
                  start_angle=pi/18,
                  end_angle=17*pi/18,
                  steps=1000):
    """Plot exact positions given start/finish angles and magnetic
    parameters.
    """

    sphs, times = generate_dynamics(magnetic_parameters, start_angle,
                                    end_angle, steps)

    sphstitle = "Path of m for " + str(magnetic_parameters) \
        + "\n (starting point is marked)."
    utils.plot_sph_points(sphs, title=sphstitle)

    timestitle = "Polar angle vs time for " + str(magnetic_parameters)
    utils.plot_polar_vs_time(sphs, times, title=timestitle)

    plt.show()


def calculate_equivalent_dynamics(magnetic_parameters, polars):
    """Given a list of polar angles (and some magnetic parameters)
    calculate what the corresponding azimuthal angles and switching times
    (from the first angle) should be.
    """
    start_angle = polars[0]

    f_times = ft.partial(calculate_switching_time, magnetic_parameters,
                         start_angle)
    exact_times = [f_times(p) for p in polars]

    f_azi = ft.partial(calculate_azimuthal, magnetic_parameters, start_angle)
    exact_azis = [f_azi(p) for p in polars]

    return exact_times, exact_azis


def plot_vs_exact(magnetic_parameters, ts, ms):

    # Extract lists of the polar coordinates
    m_as_sph_points = map(utils.array2sph, ms)
    pols = [m.pol for m in m_as_sph_points]
    azis = [m.azi for m in m_as_sph_points]

    # Calculate the corresponding exact dynamics
    exact_times, exact_azis = \
        calculate_equivalent_dynamics(magnetic_parameters, pols)

    # Plot
    plt.figure()
    plt.plot(ts, pols, '--',
             exact_times, pols)

    plt.figure()
    plt.plot(pols, azis, '--',
             pols, exact_azis)

    plt.show()
