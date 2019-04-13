# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

from __future__ import absolute_import, division, print_function

import numpy as np
from astropy.time import Time
import nose.tools as nt

from healvis import observatory, sky_model, utils, cosmology, pspec_tools

f21 = cosmology.f21

latitude = -30.7215277777
longitude = 21.4283055554

# TODO Test estimator on non-flat power spectrum


def test_sim_pspec_amp():
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
    obs.set_beam('gaussian', gauss_width=7.37)

    skysig = 0.031

    sky = sky_model.SkyModel(Nside=nside, freqs=freqs, ref_chan=Nfreqs // 2)
    sky.make_flat_spectrum_shell(skysig, shared_memory=True)
    pix_area = sky.pix_area_sr

    visibs, times, bls = obs.make_visibilities(sky, Nprocs=3)

    vis_jy = visibs[:, 0, :]  # (Nblts, Nskies, Nfreqs)
    vis_Ksr = utils.jy2Tsr(freqs) * vis_jy

    _vis = np.fft.ifft(vis_Ksr, axis=1)
    Nkpar = Nfreqs // 2
    _vis = _vis[:, :Nkpar]  # Keeping only positive k_par modes

    dspec_instr = np.abs(_vis)**2

    za, az = obs.calc_azza(sky.Nside, obs.pointing_centers[0])
    beam_sq_int = np.mean(obs.beam_sq_int(freqs, nside, obs.pointing_centers[0]))

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


def test_gaussian_box_fft():
    """
    Check the power spectrum of a gaussian box.
    """

    N = 256  # Vox per side
    L = 300.  # Mpc
    sig = 2.0
    mu = 0.0
    box = np.random.normal(mu, sig, (N, N, N))

    kbins, pk = pspec_tools.box_dft_pspec(box, L, use_cosmo=True)

    dV = (L / float(N))**3

    tol = np.sqrt(np.var(pk))

    nt.assert_true(np.allclose(np.mean(pk), sig**2 * dV, atol=tol))


def test_single_projected_dft():
    Nside = 128
    Npix = 12 * Nside**2
    Omega = 4 * np.pi / float(Npix)

    Nfreq = 100
    freqs = np.linspace(167.0, 177.0, Nfreq)
    dnu = np.diff(freqs)[0]
    Z = f21 / freqs - 1.

    sig = 0.031
    mu = 0.0
    shell = np.random.normal(mu, sig, (Npix, Nfreq))
    center = [np.sqrt(2) / 2., np.sqrt(2) / 2., 0]

    box = pspec_tools.cartesian_project(shell, center, 20, degrees=True)
    Nx, Ny, Nz = box.shape
    Lx = cosmology.dL_dth(Z[Nfreq // 2]) * np.sqrt(Omega) * Nx
    Ly = Lx
    Lz = cosmology.dL_df(Z[Nfreq // 2]) * dnu * Nz

    dV = cosmology.comoving_voxel_volume(Z[Nfreq // 2], dnu, Omega)

    r_mpc = cosmology.comoving_distance(Z)
    kbins, pk = pspec_tools.box_dft_pspec(box, [Lx, Ly, Lz], r_mpc=r_mpc, use_cosmo=True)

    tol = np.sqrt(np.var(pk))
    theor_amp = sig**2 * dV
    print(tol / theor_amp, theor_amp)
    nt.assert_true(np.allclose(np.mean(pk), sig**2 * dV, atol=tol))


def test_shell_pspec_dft():
    sky = sky_model.SkyModel()
    skysig = 0.031

    fov = 5  # deg
    Npatches = 20

    Nfreqs = 308//2
    sky.Nside = 512
    Npix = 256**2 * 12 // 2
    freqs = np.linspace(100e6, 130e6, Nfreqs)
    sky.freqs = freqs
    sky.ref_chan = Nfreqs // 2

    sky.make_flat_spectrum_shell(skysig, Npix = Npix)

    # Centers = along a latitude of 90 - 2*fov, covering longitudes spaced by 360/Npatches
    lats = np.ones(Npatches) * (90.0 - 2*fov)
    lons = np.linspace(0, 360, Npatches)
    centers = np.column_stack((lats, lons))
    hpx_inds = np.arange(Npix)
    kbins, pk = pspec_tools.shell_pspec(sky, Npatches=Npatches, radius=fov // 2, hpx_inds=hpx_inds, centers=centers)
    Z_sel = sky.Z_array[sky.ref_chan]
    dnu = sky.freqs[1] - sky.freqs[0]
    amp_theor = skysig**2 * cosmology.comoving_voxel_volume(Z_sel, dnu, sky.pix_area_sr)

    print(np.mean(pk / amp_theor))
    nt.assert_true(np.isclose(np.mean(pk), amp_theor, rtol=0.05))   # Within 5%
