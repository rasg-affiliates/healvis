
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

from .utils import mparray, comoving_voxel_volume, comoving_distance

# Moving a lot from eorsky to here.

f21 = 1.420405751e9


class SkyModel(object):

    Npix = None
    Nside = None
    Nskies = 1
    Nfreqs = None
    indices = None
    ref_chan = None
    pspec_amp = None
    freq_array = None
    pix_area_sr = None
    Z_array = None
    data = None
    _updated = []

    def __init__(self, **kwargs):
        self.valid_params = ['Npix', 'Nside', 'Nskies', 'Nfreqs', 'indices', 'Z_array', 'ref_chan', 'pspec_amp', 'freq_array', 'data']
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
        self.data = data
        self._update()

    def _update(self):
        """
            Assume that whatever parameter was just changed has priority over others.
            If two parameters from the same group change at the same time, confirm that they're consistent.
        """
        hpx_params = ['Nside', 'Npix', 'data', 'indices']
        z_params = ['Z_array', 'freq_array', 'Nfreqs', 'r_mpc']
        ud = np.unique(self._updated)
        if 'freq_array' in ud:
            self.Z_array = f21 / self.freq_array - 1.
            self.r_mpc = comoving_distance(self.Z_array)
            self.Nfreqs = self.freq_array.size
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
                print(self.Npix, self.Nfreqs)
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
        required = ['freq_array', 'ref_chan', 'Nside', 'Npix', 'Nfreqs']
        missing = []
        for p in required:
            if getattr(self, p) is None:
                missing.append(p)
        if len(missing) > 0:
            raise ValueError("Missing required parameters: " + ', '.join(missing))

        sig = sigma
        if shared_mem:
            self.data = mparray((self.Nskies, self.Npix, self.Nfreqs), dtype=float)
        else:
            self.data = np.zeros((self.Nskies, self.Npix, self.Nfreqs), dtype=float)

        dnu = np.diff(self.freq_array)[0] / 1e6
        om = 4 * np.pi / float(12 * self.Nside**2)
        Zs = f21 / self.freq_array - 1
        dV0 = comoving_voxel_volume(Zs[self.ref_chan], dnu, om)
        for fi in range(self.Nfreqs):
            dV = comoving_voxel_volume(Zs[fi], dnu, om)
            s = sig * np.sqrt(dV0 / dV)
            self.data[:, :, fi] = np.random.normal(0.0, s, (self.Nskies, self.Npix))
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
                if k == 'freq_array':
                    if chan_range is not None:
                        self.freq_array = infile[k][c0:c1]
                    else:
                        self.freq_array = infile[k][()]
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
