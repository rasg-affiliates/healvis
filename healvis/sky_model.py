
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
from .version import history_string

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
    valid_params = ['Npix', 'Nside', 'Nskies', 'Nfreqs', 'indices', 'Z_array', 'ref_chan',
                    'pspec_amp', 'freqs', 'data', 'history']
    _updated = []
    # keys give HDF5 datasets, value give their dtype
    dsets = {'data': np.float64, 'indices': np.int32, 'freqs': np.float64,
             'history': h5py.special_dtype(vlen=unicode)}

    def _defaults(self):
        """
        Set valid parameters to default value if they do not exist
        """
        for k in self.valid_params:
            if not hasattr(self, k):
                if k == 'history':
                    setattr(self, k, '')
                else:
                    setattr(self, k, None)
        self._updated = []

    def __init__(self, **kwargs):
        """
        Instantiate a SkyModel object.
        See SkyModel.valid_params for viable kwargs.
        """
        # defaults
        self._defaults()

        # populate kwargs
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

    def read_hdf5(self, filename, chan_range=None, shared_mem=False):
        """
        Read HDF5 HEALpix map(s)

        Args:
            filename : str
                Path to HDF5 file with HEALpix maps in SkyModel format
            chan_range : len-2 tuple
                Frequency channel index range to read
            shared_mem : bool
                If True, share memory across processes
        """
        if not os.path.exists(filename):
            raise ValueError("File {} not found.".format(filename))

        if chan_range is None:
            freq_slice = slice(None)
        else:
            freq_slice = slice(chan_range[0], chan_range[1])

        print('...reading {}'.format(filename))
        with h5py.File(filename) as infile:
            # load lightweight attributes
            for k in infile.attrs:
                setattr(self, k, infile.attrs[k])

            # load heavier datasets
            for k in self.dsets:
                if k in infile:
                    if k == 'data':
                        if shared_mem:
                            setattr(self, k, mparray(infile[k].shape, dtype=np.float))
                        # load the data via slice
                        setattr(self, k, infile[k][:, :, freq_slice])
                    elif k == 'freqs':
                        setattr(self, k, infile[k][freq_slice])
                    elif k == 'history':
                        setattr(self, k, infile[k].value)
                    else:
                        setattr(self, k, infile[k][:])

            # make sure Nfreq agrees
            self.Nfreqs = len(self.freqs)

        self._update()

    def write_hdf5(self, filename, clobber=False):
        """
        Write a SkyModel HEALpix map in celestial coordinates to HDF5.

        Args:
            filename : str
                Path to output HDF5 file
            clobber : bool
                If True, overwrite output file if it exists
        """
        if os.path.exists(filename) and clobber is False:
            print("...{} exists and clobber == False, skipping".format(filename))
            return
        print("...writing {}".format(filename))
        with h5py.File(filename, 'w') as fileobj:
            for k in self.valid_params:
                d = getattr(self, k, None)
                if d is None:
                    continue
                if k == 'history':
                    d += history_string()
                if k in self.dsets:
                    if np.isscalar(d):
                        dset = fileobj.create_dataset(k, data=d, dtype=self.dsets[k])
                    else:
                        dset = fileobj.create_dataset(k, data=d, dtype=self.dsets[k], compression='gzip', compression_opts=9)
                else:
                    fileobj.attrs[k] = d


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


def construct_skymodel(sky_type, freqs=None, Nside=None, ref_chan=0, Nskies=1, sigma=None):
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
    sky.Nskies = Nskies
    sky.ref_chan = ref_chan

    # make a flat-spectrum noise shell
    if sky_type.lower() == 'flat_spec':
        sky.make_flat_spectrum_shell(sigma, shared_mem=True)

    # make a GSM shell
    elif sky_type.lower() == 'gsm':
        sky.data = gsm_shell(Nside, freqs)
        sky._update()

    # load healpix map from disk
    else:
        sky.read_hdf5(sky_type, shared_mem=True)
        sky._update()

    return sky
