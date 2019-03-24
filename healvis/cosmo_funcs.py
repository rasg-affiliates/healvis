from math import pi
from astropy.cosmology import Planck15 as cosmo
import os
import sys
import numpy as np


def jy2Tstr(f, mK=False):  # , bm):
    '''Return [mK sr] / [Jy] vs. frequency (in Hz)
        f = frequencies (Hz!)
    '''
    c_cmps = 2.99792458e10   # cm/s
    k_boltz = 1.380658e-16   # erg/K
    lam = c_cmps / f  # cm
    bm = 1.0  # steradian
    fac = 1.0
    if mK:
        fac = 1e3
    return 1e-23 * lam**2 / (2 * k_boltz * bm) * fac


def dL_df(z, omega_m=0.266):
    '''[Mpc]/Hz, from Furlanetto et al. (2006)'''
    c_kms = 299792.458  # km/s
    nu21 = 1420e6
    return c_kms / (cosmo.H0.value * cosmo.efunc(z)) * (z + 1)**2 / nu21


def dL_dth(z):
    '''[Mpc]/radian, from Furlanetto et al. (2006)'''
    return cosmo.comoving_distance(z).value


def dk_deta(z):
    '''2pi * [Mpc^-1] / [Hz^-1]'''
    return 2 * pi / dL_df(z)


def dk_du(z):
    return 2 * pi / dL_dth(z)


def X2Y(z):
    '''[Mpc^3] / [str * Hz] scalar conversion between observing and cosmological coordinates'''
    return dL_dth(z)**2 * dL_df(z)
