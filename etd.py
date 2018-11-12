from __future__ import print_function, division

import numpy as np


def z(time, midpoint, duration, impact_param, planet_radius):
    """
    Calculate the normalized center-to-center distance

    Parameters
    ----------

    time : float
        Time at which to calculate separation.
    midpoint : float
        Midpoint of transit, in same units as ``time``.
    duration : float
        Duration of transit, in same units as ``time``.
    impact_param : float
        Normalized impact parameter for transit.

    Returns
    -------

    float
    """
    p_term = (1 + planet_radius)**2
    z = np.sqrt(4 * (time - midpoint)**2 / duration**2 * (p_term - impact_param**2)
                + impact_param**2)
    #orb_freq = 2*np.pi/orb_period
    #incl_rad = inclination/180*np.pi
    # z = semi_major * np.sqrt(np.sin(orb_freq*(time-midpoint))**2 +
    #                          (np.cos(incl_rad)*np.cos(orb_freq*(time-midpoint)))**2)
    return z


def scaled_intensity(z, radius_p, linear_darkening):
    """
    Calculated normalized intensity assuming linear limb darkening
    """
    p = radius_p
    c1 = linear_darkening
    zp_minus = np.sqrt(1 - (z - p)**2)
    zp_plus = np.sqrt(1 - (z + p)**2)

    I_star_z_big = (1 - c1 + 2*c1/3 * zp_minus)
    I_star_z_small = 1 - c1 - c1/(6*z*p)*(zp_plus**3 - zp_minus**3)
    I_star = I_star_z_big
    z_small = z < (1 - p)
    z_too_big = z > 1
    I_star[z_small] = I_star_z_small[z_small]
    #I_star[z_too_big] = 0
    return I_star


def scaled_flux(z, p, c1):
    I_star = scaled_intensity(z, p, c1)
    omega = (3-c1)/12

    flux = 1 - I_star/4/np.pi/omega*(p**2 * np.arccos((z-1)/p) -
                               (z-1) * np.sqrt(p**2 - (z-1)**2))
    all_in = z < (1 - p)
    all_out = z > (1 + p)
    flux[all_out] = 1
    flux[all_in] = 1 - I_star[all_in]/4/omega*p**2
    return flux


def mag_fit(params, t, z_0, c1=0.5, t_mean=0):
    midpoint = params[1]
    duration = params[2]
    scaled_radius = params[3]
    z_t = z(t, midpoint, duration, z_0, scaled_radius)
    zero_point = params[0]
    linear_term = params[4] * (t - t_mean)
    quadratic_term = params[5] * (t - t_mean)**2
    flux_term = scaled_flux(z_t, scaled_radius, c1)
    return (zero_point - 2.5 * np.log10(flux_term)
            + linear_term + quadratic_term)


def err_fit(params, t, mag, z_0, c1=0.5, t_mean=0):
    return mag_fit(params, t, z_0, c1=0.5, t_mean=0) - mag
