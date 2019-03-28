# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

from __future__ import absolute_import, division, print_function

import numpy as np
import multiprocessing as mp
from astropy.constants import c


def freq_array_to_params(freq_array):
    """

    Give the channel width, bandwidth, start, and end frequencies corresponding
    to a given frequency array.

    Args:
        freq_array : (ndarray, shape = (Nfreqs,)) of frequencies.
    Returns:
        Dictionary of frequency parameters.

    """
    if not isinstance(freq_array, np.ndarray):
        freq_array = np.array(freq_array)

    fdict = {}
    if freq_array.size < 2:
        raise ValueError("Frequency array must be longer than 1 to give meaningful results.")

    fdict['channel_width'] = np.diff(freq_array)[0]
    f0, f1 = freq_array[[0, -1]]
    fdict['bandwidth'] = (f1 - f0) + fdict['channel_width']
    fdict['start_freq'] = f0
    fdict['end_freq'] = f1

    return fdict


def time_array_to_params(time_array):
    """

    Give the time cadence, duration, and start and end times corresponding to a given frequency array.

    Args:
        time_array : (ndarray) of julian dates
    Returns:
        Dictionary of frequency parameters.

    """
    if not isinstance(time_array, np.ndarray):
        time_array = np.array(time_array)

    tdict = {}
    if time_array.size < 2:
        raise ValueError("Time array must be longer than 1 to give meaningful results.")

    tdict['time_cadence'] = np.diff(time_array)[0] * 24 * 3600
    t0, t1 = time_array[[0, -1]]
    tdict['duration_days'] = (t1 - t0)
    tdict['start_time'] = t0
    tdict['end_time'] = t1

    return tdict


def jy2Tsr(f, bm=1.0, mK=False):
    '''Return [K sr] / [Jy] vs. frequency (in Hz)
        Arguments:
            f = frequencies (Hz)
            bm = Reference solid angle in steradians (Defaults to 1)
            mK = Return in mK sr instead of K sr
    '''
    c_cmps = c.to('cm/s').value  # cm/s
    k_boltz = 1.380658e-16   # erg/K
    lam = c_cmps / f  # cm
    fac = 1.0
    if mK:
        fac = 1e3
    return 1e-23 * lam**2 / (2 * k_boltz * bm) * fac


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
