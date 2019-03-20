from __future__ import absolute_import, division, print_function

import numpy as np
import multiprocessing as mp
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import c


c_ms = c.to('m/s').value


class mparray(np.ndarray):
    """
    A multiprocessing RawArray accessible with numpy array slicing.
    """
    # TODO --- replace this. numpy no longer supports assignment to the data attribute:
    # https://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing

    def __init__(self, *args, **kwargs):
        super(mparray, self).__init__(*args, **kwargs)
        size = np.prod(self.shape)
        ctype = np.sctype2char(self.dtype)
        arr = mp.RawArray(ctype, size)
        self.data = arr
        self.reshape(self.shape)


def jy2Tstr(f, bm=1.0):
    '''Return [K sr] / [Jy] vs. frequency (in Hz)
        Arguments:
            f = frequencies (Hz)
            bm = Reference area (defaults to 1 steradian)
    '''
    c_cmps = c_ms * 100.   # cm/s
    k_boltz = 1.380658e-16   # erg/K
    lam = c_cmps / f  # cm
    return 1e-23 * lam**2 / (2 * k_boltz * bm)


def comoving_voxel_volume(z, dnu, omega):
    """
        From z, dnu, and pixel size (omega), get voxel volume.
        dnu = MHz
        Omega = sr
    """
    if isinstance(z, np.ndarray):
        if isinstance(omega, np.ndarray):
            z, omega = np.meshgrid(z, omega)
        elif isinstance(dnu, np.ndarray):
            z, dnu = np.meshgrid(z, dnu)
    elif isinstance(dnu, np.ndarray) and isinstance(omega, np.ndarray):
        dnu, omega = np.meshgrid(dnu, omega)
    nu0 = 1420. / (z + 1) - dnu / 2.
    nu1 = nu0 + dnu
    dz = 1420. * (1 / nu0 - 1 / nu1)
    vol = cosmo.differential_comoving_volume(z).value * dz * omega
    return vol


def comoving_distance(z):
    return cosmo.comoving_distance(z).to('Mpc').value
