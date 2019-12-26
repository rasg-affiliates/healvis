# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

from __future__ import absolute_import, division, print_function

import numpy as np
import healpy as hp
import os
import copy
from astropy.cosmology import WMAP9

from healvis import beam_model
from healvis.data import DATA_PATH
import healvis.tests as simtest


def test_PowerBeam():
    # load it
    beam_path = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    P = beam_model.PowerBeam(beam_path)
    freqs = np.arange(120e6, 160e6, 4e6)
    Nfreqs = len(freqs)

    # test frequency interpolation
    P2 = copy.deepcopy(P)
    P3 = P2.interp_freq(freqs, inplace=False, kind='linear')
    P2.interp_freq(freqs, inplace=True, kind='linear')
    # assert inplace and not inplace are consistent
    assert P2 == P3
    # assert correct frequencies
    np.testing.assert_array_almost_equal(freqs, P3.freq_array[0])
    assert (P3.bandpass_array.shape[1] == P3.Nfreqs == Nfreqs)

    # get beam value
    Npix = 20
    az = np.linspace(0, 2 * np.pi, Npix, endpoint=False)
    za = np.linspace(0, 1, Npix, endpoint=False)
    b = P.beam_val(az, az, freqs, pol='XX')
    # check shape and rough value check (i.e. interpolation is near zenith as expected)
    assert b.shape == (Npix, Nfreqs)
    assert np.isclose(b.max(), 1.0, atol=1e-3)

    # shift frequnecies by a delta and assert beams are EXACTLY the same (i.e. no freq interpolation)
    # delta must be larger than UVBeam._inter_freq tol, but small enough
    # to keep the same freq nearest neighbors
    b2 = P.beam_val(az, az, freqs + 1e6, pol='XX')
    np.testing.assert_array_almost_equal(b, b2)

    # smooth the beam
    SP = P.smooth_beam(freqs, inplace=False, freq_ls=2.0, noise=1e-10)
    assert SP.Nfreqs == len(freqs)


def test_AnalyticBeam():
    freqs = np.arange(120e6, 160e6, 4e6)
    Nfreqs = len(freqs)
    Npix = 20
    az = np.linspace(0, 2 * np.pi, Npix, endpoint=False)
    za = np.linspace(0, 1, Npix, endpoint=False)

    # Gaussian
    A = beam_model.AnalyticBeam('gaussian', gauss_width=15.0)
    b = A.beam_val(az, za, freqs)
    assert b.shape == (Npix, Nfreqs)  # assert array shape
    assert np.isclose(b[0, :], 1.0).all()  # assert peak normalized

    # Chromatic Gaussian
    A = beam_model.AnalyticBeam('gaussian', gauss_width=15.0, ref_freq=freqs[0], spectral_index=-1.0)
    b = A.beam_val(az, za, freqs)
    assert b.shape == (Npix, Nfreqs)  # assert array shape
    assert np.isclose(b[0, :], 1.0).all()  # assert peak normalized

    # Uniform
    A = beam_model.AnalyticBeam('uniform')
    b = A.beam_val(az, za, freqs)
    assert b.shape == (Npix, Nfreqs)  # assert array shape
    assert np.isclose(b, 1.0).all()  # assert peak normalized

    # Airy
    A = beam_model.AnalyticBeam('airy', diameter=15.0)
    b = A.beam_val(az, za, freqs)
    assert b.shape == (Npix, Nfreqs)  # assert array shape
    assert np.isclose(b[0, :], 1.0).all()  # assert peak normalized

    # custom
    A = beam_model.AnalyticBeam(beam_model.airy_disk)
    b2 = A.beam_val(az, za, freqs, diameter=15.0)
    assert b2.shape == (Npix, Nfreqs)  # assert array shape
    assert np.isclose(b2[0, :], 1.0).all()  # assert peak normalized
    np.testing.assert_array_almost_equal(b, b2)  # assert its the same as airy

    # exceptions
    A = beam_model.AnalyticBeam("uniform")
    simtest.assert_raises_message(NotImplementedError, 'Beam type foo not available yet.', beam_model.AnalyticBeam, "foo")
    simtest.assert_raises_message(KeyError, 'gauss_width required for gaussian beam', beam_model.AnalyticBeam, 'gaussian')
    simtest.assert_raises_message(KeyError, 'Dish diameter required for airy beam', beam_model.AnalyticBeam, 'airy')
