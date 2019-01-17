import numpy as np
import multiprocessing as mp
from astropy.cosmology import WMAP9 as cosmo

## TODO --- replace this. numpy no longer supports assignment to the data attribute:
##  https://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing

class mparray(np.ndarray):

    """
    A multiprocessing RawArray accessible with numpy array slicing.
    """

    def __init__(self, *args, **kwargs):
        super(mparray, self).__init__(*args, **kwargs)
        size = np.prod(self.shape)
        ctype = np.sctype2char(self.dtype)
        arr = mp.RawArray(ctype, size)
        self.data = arr
        self.reshape(self.shape)


def comoving_voxel_volume(z,dnu,omega):
    """
        From z, dnu, and pixel size (omega), get voxel volume.
        dnu = MHz
        Omega = sr
    """
    if isinstance(z,np.ndarray):
        if isinstance(omega,np.ndarray):
            z, omega = np.meshgrid(z,omega)
        elif isinstance(dnu,np.ndarray):
            z, dnu = np.meshgrid(z,dnu)
    elif isinstance(dnu, np.ndarray) and isinstance(omega, np.ndarray):
        dnu, omega = np.meshgrid(dnu, omega)
    nu0 = 1420./(z + 1) - dnu/2.
    nu1 = nu0 + dnu
    dz = 1420.*(1/nu0 - 1/nu1)
    vol = cosmo.differential_comoving_volume(z).value*dz*omega
    return vol

