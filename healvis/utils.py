# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

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
    freq_array = np.asarray(freq_array).ravel()

    fdict = {}
    if freq_array.size < 2:
        raise ValueError(
            "Frequency array must be longer than 1 to give meaningful results."
        )

    fdict["channel_width"] = np.diff(freq_array)[0]
    fdict["Nfreqs"] = freq_array.size
    fdict["bandwidth"] = fdict["channel_width"] * fdict["Nfreqs"]
    fdict["start_freq"] = freq_array[0]
    fdict["end_freq"] = freq_array[-1]

    return fdict


def time_array_to_params(time_array):
    """
    Give the time cadence, duration, and start and end times corresponding to a given time array.

    Args:
        time_array : (ndarray) of julian dates

    Returns:
        Dictionary of time parameters.

    """
    time_array = np.asarray(time_array)

    tdict = {}
    if time_array.size < 2:
        raise ValueError("Time array must be longer than 1 to give meaningful results.")

    tdict["time_cadence"] = np.diff(time_array)[0] * (24.0 * 3600.0)
    tdict["Ntimes"] = time_array.size
    tdict["duration"] = tdict["time_cadence"] * tdict["Ntimes"] / (24.0 * 3600.0)
    tdict["start_time"] = time_array[0]
    tdict["end_time"] = time_array[-1]

    return tdict


def jy2Tsr(f, bm=1.0, mK=False):
    """Return [K sr] / [Jy] vs. frequency (in Hz)
    Arguments:
        f = frequencies (Hz)
        bm = Reference solid angle in steradians (Defaults to 1)
        mK = Return in mK sr instead of K sr
    """
    c_cmps = c.to("cm/s").value  # cm/s
    k_boltz = 1.380658e-16  # erg/K
    lam = c_cmps / f  # cm
    fac = 1.0
    if mK:
        fac = 1e3
    return 1e-23 * lam ** 2 / (2 * k_boltz * bm) * fac


class mparray(np.ndarray):
    """
    A multiprocessing RawArray accessible with numpy array slicing.
    """

    # TODO --- replace this. numpy no longer supports assignment to the data attribute:
    # https://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing

    def __init__(self, *args, **kwargs):
        ctype = np.sctype2char(self.dtype)
        arr = mp.RawArray(ctype, self.size)
        self.data = arr
        self.reshape(self.shape)


def enu_array_to_layout(enu_arr, fname):
    """
    Write out an array of antenna positions in ENU to a text file.
    """
    with open(fname, "w") as lfile:
        for i in range(enu_arr.shape[0]):
            e, n, u = enu_arr[i]
            name = "ant{}".format(i)
            num = i
            beam_id = 0
            line = ("{:6} {:8d} {:8d} {:10.4f} {:10.4f} {:10.4f}\n").format(
                name, num, beam_id, e, n, u
            )
            lfile.write(line)


def npix2nside(npix):
    test = np.log2(npix / 12)
    if not test == np.floor(test):
        raise ValueError(f"Invalid number of pixels {npix}")
    return int(test)
