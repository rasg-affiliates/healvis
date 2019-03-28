# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

from __future__ import absolute_import, division, print_function

import numpy as np
from astropy.time import Time
import nose.tools as nt

from healvis import observatory, sky_model, utils, cosmology

latitude = -30.7215277777
longitude = 21.4283055554


def test_pspec_amp():
    # Construct a flat-spectrum shell, simulate visibilities,
    #  confirm power spectrum amplitude.

    ant1_enu = np.array([0, 0, 0])
    ant2_enu = np.array([0.0, 14.6, 0])

    bl = observatory.Baseline(ant1_enu, ant2_enu)

    Ntimes = 20
    Nfreqs = 200
    freqs = np.linspace(100e6, 150e6, Nfreqs)

    fov = 50  # Deg

    # Longitude/Latitude in degrees.

    nside = 64

    obs = observatory.Observatory(latitude, longitude, array=[bl], freqs=freqs)
    t0 = Time('J2000').jd
    obs.times_jd = np.linspace(t0, t0 + 0.5, Ntimes)    # Half a day
    obs.set_pointings(obs.times_jd)

    obs.set_fov(fov)
    obs.set_beam('uniform')

    skysig = 0.031

    sky = sky_model.SkyModel(Nside=nside, freqs=freqs, ref_chan=Nfreqs // 2)
    sky.make_flat_spectrum_shell(skysig, shared_mem=True)
    pix_area = sky.pix_area_sr

    visibs, times, bls = obs.make_visibilities(sky, Nprocs=3)

    vis_jy = visibs[:, 0, :]  # (Nblts, Nskies, Nfreqs)
    vis_Ksr = utils.jy2Tsr(freqs) * vis_jy

    _vis = np.fft.ifft(vis_Ksr, axis=1)
    Nkpar = Nfreqs // 2
    _vis = _vis[:, :Nkpar]  # Keeping only positive k_par modes

    dspec_instr = np.abs(_vis)**2

    za, az = obs.calc_azza(sky.Nside, obs.pointing_centers[0])
    beam_sq_int = np.sum(obs.beam.beam_val(az, za, freqs)**2, axis=0) * pix_area
    beam_sq_int = beam_sq_int[0]    # Constant across frequencies

    Bandwidth = freqs[-1] - freqs[0]
    scalar = cosmology.X2Y(sky.Z_array[sky.ref_chan]) * (Bandwidth / beam_sq_int)

    dspec_I = np.mean(dspec_instr * scalar, axis=0)

    # Theoretical pspec amp
    dnu = np.diff(freqs)[0]
    Z_sel = sky.Z_array[sky.ref_chan]
    amp_theor = skysig**2 * cosmology.comoving_voxel_volume(Z_sel, dnu, sky.pix_area_sr)

    tolerance = (amp_theor / float(Ntimes))   # assuming independent fields
    print(amp_theor, np.mean(dspec_I))
    print(tolerance)
    nt.assert_true(np.isclose(amp_theor, np.mean(dspec_I), atol=2 * tolerance))   # Close to within twice the sample variance
