
# Code for making, reading, and writing sky models
# skymodel = Npix, Nside, Nfreqs, and data arrays
#            data = either a shared array or numpy array shape (Npix, Nfreqs)
#            (optionally) pixels = (Npix,) array of pixel numbers (for incomplete maps)
#

from __future__ import absolute_import, division, print_function

import numpy as np
import h5py
from astropy.cosmology import Planck15 as cosmo

from .utils import mparray, comoving_voxel_volume, comoving_distance

# Moving a lot from eorsky to here.

f21 = 1420e6

class skymodel(object):

    Npix = None
    Nside = None
    Nskies = 1
    Nfreqs = None
    indices = None
    ref_chan = None
    pspec_amp = None
    freq_array = None
    Z_array = None
    data = None
    _updated = []

    def __init__(self, **kwargs):

        params = ['Npix', 'Nside', 'Nskies, Nfreqs', 'indices', 'ref_chan', 'pspec_amp', 'freq_array', 'data']

        for k, v in kwargs.items():
            if k in params:
                setattr(self, k, v)
        self._update()

    def __setattr__(self, name, value):
        if self._updated is None:
            self._updated = []
        self._updated.append(name)
        super(skymodel, self).__setattr__(name, value)

    def _update(self):
        """
            Assume that whatever parameter was just changed has priority over others.
            If two parameters from the same group change at the same time, confirm that they're consistent.
        """

        hpx_params = ['Nside', 'Npix', 'data', 'indices']
        z_params = ['Z_array', 'freq_array', 'Nfreqs', 'r_mpc']
        ud = np.unique(self._updated)
        for p in ud:
            if p == 'freq_array':
                self.Z_array = f21 / self.freq_array - 1.
                self.r_mpc = comoving_distance(self.Z_array)
                self.Nfreqs = self.freq_array.size
            if p == 'Nside':
                if self.Npix is None:
                    self.Npix = 12 * self.Nside**2
                if 'indices' not in ud:
                    if self.indices is None:
                        self.indices = np.arange(self.Npix)
            if p == 'indices':
                self.Npix = self.indices.size
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
        self._update()

    def read_hdf5(self, filename, chan_range=None):
        print('Reading: ', filename)
        infile = h5py.File(filename)
        skymodel = 'spectral_info' in infile.keys()
        transpose = False
        if skymodel:
            freqs_key = 'spectral_info/freq'
            shell_key = 'spectral_info/spectrum'
        else:
            freqs_key = 'specinfo/freqs'
            shell_key = 'skyinfo/surfaces'
            transpose = True
        if chan_range is None:
            chan_range = [0, infile[freqs_key].shape[0]]
        c0, c1 = chan_range
        self.freq_array = infile[freqs_key][c0:c1] / 1e6  # Hz -> MHz
        if not transpose:
            self.shell = infile[shell_key][:, c0:c1]
        else:
            self.shell = infile[shell_key][c0:c1, :].T
        self.Npix, self.Nfreq = self.shell.shape
        self.indices = np.arange(self.Npix)
        self.Nside = hp.npix2nside(self.Npix)
        infile.close()
        self._update()

    def write_hdf5(self, filename):
        with h5py.File(filename, 'w') as fileobj:
            freqs_Hz = self.freq_array
            hdr_grp = fileobj.create_group('header')
            hdr_grp['units'] = "K"
            hdr_grp['is_healpix'] = 1
            spec_group = fileobj.create_group('spectral_info')
            freq_dset = spec_group.create_dataset('freq', data=freqs_Hz, compression='gzip', compression_opts=9)
            freq_dset.attrs['units'] = 'Hz'
            spectrum_dset = spec_group.create_dataset('spectrum', data=self.shell, compression='gzip', compression_opts=9)
