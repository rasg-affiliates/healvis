from __future__ import absolute_import, division, print_function

import numpy as np
import multiprocessing as mp
from astropy.constants import c


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
