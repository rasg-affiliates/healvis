from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import c

# -----------------------
#  Functions and constants related
#  to conversions between instrument and
#  cosmological units.
# -----------------------

f21 = 1.420405751e9
c_ms = c.to('m/s').value


def jy2Tsr(f, bm=1.0, mK=False):
    '''Return [K sr] / [Jy] vs. frequency (in Hz)
        Arguments:
            f = frequencies (Hz)
            bm = Reference solid angle in steradians (Defaults to 1)
            mK = Return in mK sr instead of K sr
    '''
    c_cmps = c_ms * 100.   # cm/s
    k_boltz = 1.380658e-16   # erg/K
    lam = c_cmps / f  # cm
    fac = 1.0
    if mK:
        fac = 1e3
    return 1e-23 * lam**2 / (2 * k_boltz * bm) * fac


def comoving_distance(z):
    return cosmo.comoving_distance(z).to('Mpc').value


def dL_df(z):
    '''
    Comoving differential distance at redshift per frequency.

    [cMpc]/Hz, from Furlanetto et al. (2006)
    '''
    c_kms = c_ms/1e3  # km/s
    return c_kms / (cosmo.H0.value * cosmo.efunc(z)) * (z + 1)**2 / f21


def dL_dth(z):
    '''
    Comoving transverse distance per radian in Mpc
    [cMpc]/radian
    '''
    return cosmo.comoving_transverse_distance(z).value


def dk_deta(z):
    '''2pi * [Mpc^-1] / [Hz^-1]'''
    return 2 * np.pi / dL_df(z)


def dk_du(z):
    return 2 * np.pi / dL_dth(z)


def X2Y(z):
    '''[Mpc^3] / [sr * Hz] scalar conversion between observing and cosmological coordinates'''
    return dL_dth(z)**2 * dL_df(z)


def comoving_voxel_volume(z, dnu, omega):
    """
        Get comoving voxel volume in Mpc^3

        dnu = Channel width in Hz
        Omega = pixel area in steradian
    """
    if isinstance(z, np.ndarray):
        if isinstance(omega, np.ndarray):
            z, omega = np.meshgrid(z, omega)
        elif isinstance(dnu, np.ndarray):
            z, dnu = np.meshgrid(z, dnu)
    elif isinstance(dnu, np.ndarray) and isinstance(omega, np.ndarray):
        dnu, omega = np.meshgrid(dnu, omega)
    nu0 = f21 / (z + 1) - dnu / 2.
    nu1 = nu0 + dnu
    dz = f21 * (1 / nu0 - 1 / nu1)
    vol = cosmo.differential_comoving_volume(z).value * dz * omega
    return vol
