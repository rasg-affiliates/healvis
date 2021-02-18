# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

from __future__ import absolute_import, division, print_function

import numpy as np
import os
import healpy as hp
from astropy.cosmology import Planck15

from healvis import sky_model, utils
from healvis.data import DATA_PATH
from healvis.tests import TESTDATA_PATH
import tempfile


def verify_update(sky_obj):
    """
    Check that the frequencies/redshifts/distances are consistent.
    If healpix, check that Npix/indices/Nside are consistent
    """

    if sky_obj.data is not None:
        Nside = sky_obj.Nside
        indices = sky_obj.indices
        Npix = sky_obj.Npix

        if Npix == 12 * Nside**2:
            assert all(indices == np.arange(Npix))
        else:
            assert Npix == indices.size
    if sky_obj.freqs is not None:
        freqs = sky_obj.freqs
        Nfreq = sky_obj.Nfreqs
        Z = sky_obj.Z_array
        r_mpc = sky_obj.r_mpc

        Zcheck = 1420. / freqs - 1.
        Rcheck = Planck15.comoving_distance(Zcheck).to("Mpc").value
        assert all(Zcheck == Zcheck)
        assert all(Rcheck == Rcheck)
    assert len(sky_obj._updated) == 0  # The _updated list should be cleared


def test_update():
    Nfreqs = 100
    freq_array = np.linspace(167.0, 177.0, Nfreqs)
    sky = sky_model.SkyModel(Nside=64, Nskies=1, freqs=freq_array)
    verify_update(sky)
    sky.ref_chan = 50
    sky.make_flat_spectrum_shell(sigma=2.0)
    verify_update(sky)


def test_flat_spectrum():
    Nfreqs = 100
    freq_array = np.linspace(167.0e6, 177.0e6, Nfreqs)
    sigma = 2.0
    Nside = 32
    Npix = hp.nside2npix(Nside)
    Nskies = 1
    maps = sky_model.flat_spectrum_noise_shell(sigma, freq_array, Nside, Nskies)
    assert maps.shape == (Nskies, Npix, Nfreqs)
    # any other checks?


def test_write_read():
    dr = tempfile.mkdtemp()
    testfilename = os.path.join(dr, 'test_out.hdf5')
    Nfreqs = 10
    freq_array = np.linspace(167.0, 177.0, Nfreqs)
    sky = sky_model.SkyModel(Nside=64, Nskies=1, freqs=freq_array, ref_chan=5)
    sky.make_flat_spectrum_shell(sigma=2.0)
    sky.write_hdf5(testfilename)
    sky2 = sky_model.SkyModel()
    sky3 = sky_model.SkyModel()
    sky2.read_hdf5(testfilename, shared_memory=True)
    assert isinstance(sky2.data, utils.mparray)
    sky3.read_hdf5(testfilename)
    sky.history, sky2.history, sky3.history, = '', '', ''
    assert sky == sky2
    assert sky2 == sky3
    os.remove(testfilename)


def test_fewchannel_read():
    sky = sky_model.SkyModel()
    chans = np.arange(4)
    sky.read_hdf5(os.path.join(DATA_PATH, "gsm_nside32.hdf5"), freq_chans=chans)
    assert sky.freqs.size == 4
    sky = sky_model.SkyModel()
    sky.read_hdf5(os.path.join(DATA_PATH, "gsm_nside32.hdf5"), freq_chans=chans, shared_memory=True)
    assert sky.freqs.size == 4


def test_freqselect_read():
    # Using existing frequencies as selection on read
    sky = sky_model.SkyModel()
    sky.read_hdf5(os.path.join(DATA_PATH, "gsm_nside32.hdf5"))
    subfreqs = sky.freqs[::2]
    sky = sky_model.SkyModel()
    sky.freqs = subfreqs
    sky.read_hdf5(os.path.join(DATA_PATH, "gsm_nside32.hdf5"), do_not_overwrite_freqs=True)
    assert sky.freqs.size == subfreqs.size
