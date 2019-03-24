
# Code for making, reading, and writing sky models
# skymodel = Npix, Nside, Nfreqs, and data arrays
#            data = either a shared array or numpy array shape (Npix, Nfreqs)
#            (optionally) pixels = (Npix,) array of pixel numbers (for incomplete maps)
#

from __future__ import absolute_import, division, print_function

import numpy as np
import os
import h5py
from astropy.cosmology import Planck15 as cosmo
import healpy as hp

from .utils import mparray, comoving_voxel_volume, comoving_distance

try:
    import pygsm
    pygsm_import = True
except ImportError:
    pygsm_import = False


f21 = 1.420405751e9


class SkyModel(object):
    """
    SkyModel class
    """
    Npix = None
    Nside = None
    Nskies = 1
    Nfreqs = None
    indices = None
    ref_chan = None
    pspec_amp = None
    freqs = None
    pix_area_sr = None
    Z_array = None
    data = None
    _updated = []

    def __init__(self, **kwargs):
        """
        Instantiate a SkyModel object.
        """
        self.valid_params = ['Npix', 'Nside', 'Nskies', 'Nfreqs', 'indices', 'Z_array', 'ref_chan', 'pspec_amp', 'freqs', 'data']
        self._updated = []
        for k, v in kwargs.items():
            if k in self.valid_params:
                setattr(self, k, v)
        self._update()

    def __setattr__(self, name, value):
        if self._updated is None:
            self._updated = []
        self._updated.append(name)
        super(SkyModel, self).__setattr__(name, value)

    def __eq__(self, other):
        for k in self.valid_params:
            if not np.all(getattr(self, k) == getattr(other, k)):
                print('Mismatch: ', k)
                return False
        return True

    def set_data(self, data):
        """
        Assign data to self. Must be an ndarray or mparray of shape (Nskies, Npix, Nfreqs).

        SkyModel currently only supports Stokes I sky models in Kelvin.
        """
        self.data = data
        self._update()

    def _update(self):
        """
            Assume that whatever parameter was just changed has priority over others.
            If two parameters from the same group change at the same time, confirm that they're consistent.
        """
        hpx_params = ['Nside', 'Npix', 'data', 'indices']
        z_params = ['Z_array', 'freqs', 'Nfreqs', 'r_mpc']
        ud = list(np.unique(self._updated))
        if 'freqs' in ud:
            self.Z_array = f21 / self.freqs - 1.
            self.r_mpc = comoving_distance(self.Z_array)
            self.Nfreqs = self.freqs.size
        if 'Nside' in ud:
            if self.Npix is None:
                self.Npix = 12 * self.Nside**2
            self.pix_area_sr = 4 * np.pi / (12. * self.Nside**2)
            if 'indices' not in ud:
                if self.indices is None:
                    self.indices = np.arange(self.Npix)
        if 'indices' in ud:
            self.Npix = self.indices.size
        if 'data' in ud:
            # Make sure the data array has a Nskies axis
            s = self.data.shape
            if len(s) == 2:
                # print(self.Npix, self.Nfreqs)
                if not ((s[0] == self.Npix) and (s[1] == self.Nfreqs)):
                    raise ValueError("Invalid data array shape: " + str(s))
                else:
                    self.data = self.data.reshape((1,) + s)
        self._updated = []

    def make_flat_spectrum_shell(self, sigma, shared_mem=False):
        """
        sigma = Spectrum amplitude
        shared_mem = put data in a multiprocessing shared memory block
        """

        self._update()
        required = ['freqs', 'ref_chan', 'Nside', 'Npix', 'Nfreqs']
        missing = []
        for p in required:
            if getattr(self, p) is None:
                missing.append(p)
        if len(missing) > 0:
            raise ValueError("Missing required parameters: " + ', '.join(missing))

        self.data = flat_spectrum_noise_shell(sigma, self.freqs, self.Nside, self.Nskies,
                                              ref_chan=self.ref_chan, shared_mem=shared_mem)
        self.pspec_amp = sigma
        self._update()

    def read_hdf5(self, filename, chan_range=None, load_data=True, shared_mem=False):
        if not os.path.exists(filename):
            raise ValueError("File {} not found.".format(filename))
        print('Reading: ', filename)
        with h5py.File(filename) as infile:
            for k in infile.keys():
                if chan_range is not None:
                    c0, c1 = chan_range
                if k == 'freqs':
                    if chan_range is not None:
                        self.freqs = infile[k][c0:c1]
                    else:
                        self.freqs = infile[k][()]
                elif k == 'data':
                    if not load_data:
                        self.data = infile[k]   # Keep on disk until needed.
                        continue
                    dat = infile[k]
                    if shared_mem:
                        self.data = mparray(dat.shape, dtype=float)
                        if chan_range:
                            self.data = dat[:, :, c0:c1]
                        else:
                            self.data[()] = dat[()]
                    else:
                        if chan_range:
                            self.data = dat[:, :, c0:c1]
                        else:
                            self.data = dat[()]
                elif k in self.valid_params:
                    dat = infile[k][()]
                    setattr(self, k, dat)
        self._update()

    def write_hdf5(self, filename):
        with h5py.File(filename, 'w') as fileobj:
            for k in self.valid_params:
                d = getattr(self, k)
                if k.startswith('N'):
                    fileobj.attrs
                if d is not None:
                    if np.isscalar(d):
                        dset = fileobj.create_dataset(k, data=d)
                    else:
                        dset = fileobj.create_dataset(k, data=d, compression='gzip', compression_opts=9)


def flat_spectrum_noise_shell(sigma, freqs, Nside, Nskies, ref_chan=0, shared_mem=False):
    """
    Make a flat-spectrum noise-like shell.

    Args:
        sigma : float
            Power spectrum amplitude
        freqs : ndarray
            Frequencies [Hz]
        Nside : int
            HEALpix Nside resolution
        Nskies : int
            Number of indepenent skies to simulate
        ref_chan : int
            freqs reference channel index for comoving volume factor
        shared_mem : bool
            If True use mparray to generate data

    Returns:
        data : ndarray, shape (Nskies, Npix, Nfreqs)
            EoR shell as HEALpix maps
    """
    # generate empty array
    Nfreqs = len(freqs)
    Npix = hp.nside2npix(Nside)
    if shared_mem:
        data = mparray((Nskies, Npix, Nfreqs), dtype=float)
    else:
        data = np.zeros((Nskies, Npix, Nfreqs), dtype=float)

    # setup parameters
    dnu = np.diff(freqs)[0]
    om = 4 * np.pi / float(12 * Nside**2)
    Zs = f21 / freqs - 1.0
    dV0 = comoving_voxel_volume(Zs[ref_chan], dnu, om)

    # iterate over frequencies
    for i in range(Nfreqs):
        dV = comoving_voxel_volume(Zs[i], dnu, om)
        amp = sigma * np.sqrt(dV0 / dV)
        data[:, :, i] = np.random.normal(0.0, amp, (Nskies, Npix))

    return data


def gsm_shell(Nside, freqs):
    """
    Generate a Global Sky Model shell

    Args:
        Nside : int
            Nside resolution of HEALpix maps
        freqs : ndarray
            Array of frequencies [Hz]

    Returns:
        data : ndarray, shape (Npix, Nfreqs)
            GSM shell as HEALpix maps
    """
    assert pygsm_import, "pygsm package not found. This is required to use GSM functionality."

    maps = pygsm.GlobalSkyModel(freq_unit='Hz', basemap='haslam').generate(freqs)  # Units K

    rot = hp.Rotator(coord=['G', 'C'])
    Npix = Nside**2 * 12
    for fi, f in enumerate(freqs):
        maps[fi] = rot.rotate_map_pixel(maps[fi])  # Convert to equatorial coordinates (ICRS)
        maps[fi, :Npix] = hp.ud_grade(maps[fi], Nside)
    maps = maps[:, :Npix]

    return maps.T


def construct_skymodel(sky_type, freqs=None, Nside=None, ref_chan=0, sigma=None, **kwargs):
    """
    Construct a SkyModel object or read from disk

    Args:
        sky_type : str, options=["flat_spec", "gsm", "<filepath>"]
            Specify the kind of SkyModel to create. Intepreted as
            either a flat-spectrum noise sky, a GSM sky otherwise
            will attempt to read sky_type as an HDF5 filepath.
        freqs : 1D ndarray
            Frequency array [Hz]
        Nside : int
            HEALpix Nside resolution to use if creating a SkyModel
        ref_chan : int
            Frequency reference channel for cosmological conversions
        sigma : float
            If sky_type == 'flat_spec', this is the power spectrum amplitude
    
    Returns:
        SkyModel object
    """
    sky = SkyModel()
    sky.Nside = Nside
    sky.freqs = freqs
    sky.ref_chan = ref_chan

    # make a flat-spectrum noise shell
    if sky_type == 'flat_spec':
        sky.make_flat_spectrum_shell(sigma, shared_mem=True)

    # make a GSM shell
    elif sky_type == 'gsm':
        sky.data = gsm_shell(Nside, freqs)
        sky._update()

    # load healpix map from disk
    else:
        sky.read_hdf5(sky__type, shared_mem=True)
        sky._update()

    return sky
