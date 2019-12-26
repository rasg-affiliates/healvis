# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

from __future__ import absolute_import, division, print_function

import numpy as np
import os
import healpy as hp
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, ICRS, Angle

from healvis import observatory, sky_model, beam_model, utils
from healvis.data import DATA_PATH

# TODO
# Test a skymodel that isn't a complete-sky shell (ie., use the sky.indices key)

# HERA site
latitude = -30.7215277777
longitude = 21.4283055554


def test_Observatory():
    # TODO: fill out test coverage

    # setup
    ant1 = np.array([15.0, 0, 0])
    ant2 = np.array([0.0, 0, 0])
    bl = observatory.Baseline(ant1, ant2)
    Npix, Nfreqs = 10, 100
    az = np.linspace(0, 2 * np.pi, Npix)
    za = np.linspace(0, np.pi, Npix)
    freqs = np.linspace(100e6, 200e6, Nfreqs)
    obs = observatory.Observatory(latitude, longitude, array=[bl], freqs=freqs)

    # test Analytic set beam
    for beam in ['uniform', 'gaussian', 'airy', beam_model.airy_disk]:
        obs.set_beam(beam, gauss_width=10, diameter=10)
        assert isinstance(obs.beam, beam_model.AnalyticBeam)
        b = obs.beam.beam_val(az, za, freqs, pol='xx')
        assert b.shape == (Npix, Nfreqs)

    # test Power set beam
    obs.set_beam(os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits"))
    assert isinstance(obs.beam, beam_model.PowerBeam)
    b = obs.beam.beam_val(az, za, freqs, pol='xx')
    assert b.shape == (Npix, Nfreqs)


def test_Baseline():
    # TODO: fill out test coverage

    # setup
    ant1 = np.array([15.0, 0, 0])
    ant2 = np.array([0.0, 0, 0])
    bl = observatory.Baseline(ant1, ant2)
    Npix, Nfreqs = 10, 100
    az = np.linspace(0, 2 * np.pi, Npix)
    za = np.linspace(0, np.pi, Npix)
    freqs = np.linspace(100e6, 200e6, Nfreqs)

    # test fringe is right dimensions
    fringe = bl.get_fringe(az, za, freqs)
    assert fringe.shape == (Npix, Nfreqs)

    # test fringe at zenith is 1.0 across all freqs
    assert np.isclose(fringe[0, :], 1.0).all()


####################
# Validation Tests #
####################

def test_pointings():
    """
    Test pointings

    In the ICRS, RAs should be close to the LST for times near J2000.

    This test checks that the pointing centers shift in RA at near the sky rotation rate,
    and that DEC stays close to the observatory latitude.
    """

    t0 = Time('J2000').jd
    dt_min = 20.0
    dt_days = dt_min * 1 / 60. * 1 / 24.  # 20 minutes in days

    time_arr = np.arange(20) * dt_days + t0
    obs = observatory.Observatory(latitude, longitude)

    obs.set_pointings(time_arr)

    ras = np.array([c[0] for c in obs.pointing_centers])
    decs = np.array([c[1] for c in obs.pointing_centers])
    if np.any(np.diff(ras) < 0):
        ind = np.where(np.diff(ras) < 0)[0][0]
        ras[ind + 1:] += 360.   # Deal with 360 degree wrap
    degperhour_sidereal = 360. / 23.9344
    dts = np.diff(ras) / degperhour_sidereal
    dts *= 60.  # Minutes
    assert np.allclose(dts, dt_min, atol=1e-2)  # Within half a second.
    assert np.allclose(decs, latitude, atol=1e-1)  # Within 6 arcmin


def test_az_za():
    """
    Check the calculated azimuth and zenith angle of a point exactly 5 deg east on the sphere (az = 90d, za = 5d)
    """
    Nside = 128
    obs = observatory.Observatory(latitude, longitude)
    center = [0, 0]
    lon, lat = [5, 0]
    ind0 = hp.ang2pix(Nside, lon, lat, lonlat=True)
    lon, lat = hp.pix2ang(Nside, ind0, lonlat=True)
    cvec = hp.ang2vec(center[0], center[1], lonlat=True)
    radius = np.radians(10.)
    obs.set_fov(20)
    pix = hp.query_disc(Nside, cvec, radius)
    za, az = obs.calc_azza(Nside, center)
    ind = np.where(pix == ind0)
    print(np.degrees(za[ind]), np.degrees(az[ind]))
    print(lon, lat)
    # lon = longitude of the source, which is set to 5deg off zenith (hence, zenith angle)
    assert np.isclose(np.degrees(za[ind]), lon)
    assert np.isclose(np.degrees(az[ind]), 90.)


def test_vis_calc():
    # Construct a shell with a single point source at the zenith and confirm against analytic calculation.

    ant1_enu = np.array([0, 0, 0])
    ant2_enu = np.array([0.0, 14.6, 0])

    bl = observatory.Baseline(ant1_enu, ant2_enu)

    freqs = np.array([1e8])
    nfreqs = 1

    fov = 20  # Deg

    # Longitude/Latitude in degrees.

    nside = 128
    ind = 10
    center = list(hp.pix2ang(nside, ind, lonlat=True))
    centers = [center]
    npix = nside**2 * 12
    shell = np.zeros((npix, nfreqs))
    pix_area = 4 * np.pi / float(npix)
    shell[ind] = 1  # Jy/pix
    shell[ind] *= utils.jy2Tsr(freqs[0], bm=pix_area)  # K

    obs = observatory.Observatory(latitude, longitude, array=[bl], freqs=freqs)
    obs.pointing_centers = centers
    obs.times_jd = np.array([1])
    obs.set_fov(fov)
    obs.set_beam('uniform')

    sky = sky_model.SkyModel(Nside=nside, freqs=freqs, data=shell[np.newaxis, :, :])

    visibs, times, bls = obs.make_visibilities(sky)
    print(visibs)
    assert np.isclose(np.real(visibs), 1.0).all()  # Unit point source at zenith


def test_offzenith_vis():
    # Construct a shell with a single point source a known position off from zenith.
    #   Similar to test_vis_calc, but set the pointing center 5deg off from the zenith and adjust analytic calculation

    freqs = [1.0e8]
    Nfreqs = 1
    fov = 60
    ant1_enu = np.array([10.0, 0, 0])
    ant2_enu = np.array([0.0, 140.6, 0])

    bl = observatory.Baseline(ant1_enu, ant2_enu)

    # Set pointing center to ra/dec = 0/0
    center = [0, 0]

    # Make a shell and place a 1 Jy/pix point source at ra/dec of 5/0
    Nside = 128
    Npix = Nside**2 * 12
    shell = np.zeros((Npix, Nfreqs))
    pix_area = 4 * np.pi / float(Npix)
    ind = hp.ang2pix(Nside, 5, 0.0, lonlat=True)
    shell[ind] = 1  # Jy/pix
    shell[ind] *= utils.jy2Tsr(freqs[0], bm=pix_area)  # K

    obs = observatory.Observatory(0, 0, array=[bl], freqs=freqs)
    obs.pointing_centers = [center]
    obs.times_jd = np.array([1])
    obs.set_fov(fov)
    obs.set_beam('uniform')

    sky = sky_model.SkyModel(Nside=Nside, freqs=np.array(freqs), data=shell)

    vis_calc, times, bls = obs.make_visibilities(sky)

    ra_deg, dec_deg = hp.pix2ang(Nside, ind, lonlat=True)

    src_az = np.radians(90.0)
    src_za = np.radians(ra_deg)

    src_l = np.sin(src_az) * np.sin(src_za)
    src_m = np.cos(src_az) * np.sin(src_za)
    src_n = np.cos(src_za)
    u, v, w = bl.get_uvw(freqs[0])

    vis_analytic = (1) * np.exp(2j * np.pi * (u * src_l + v * src_m + w * src_n))

    print(vis_analytic)
    print(vis_calc)
    print(vis_calc[0, 0, 0] - vis_analytic)
    vis_calc = vis_calc[0, 0, 0]
    # nt.assert_true(np.isclose(vis_calc, vis_analytic, atol=1e-3).all())
    assert np.isclose(vis_calc.real, vis_analytic.real)
    assert np.isclose(vis_calc.imag, vis_analytic.imag)


def test_gsm_pointing():
    # test that PyGSM visibility sim peaks at times when galactic center transits
    ant1_enu = np.array([0.0, 0.0, 0.0])
    ant2_enu = np.array([15.0, 0, 0])
    bl = observatory.Baseline(ant1_enu, ant2_enu)

    freqs = np.linspace(100e6, 200e6, 2)
    nfreqs = len(freqs)
    # for 2458000, galactic center at RA of 266 degrees transits at 2458000.2272949447
    times = np.linspace(2458000.2272949447 - 0.2, 2458000.2272949447 + 0.2, 11)
    fov = 20  # Deg

    nside = 64
    gsm = sky_model.gsm_shell(nside, freqs)[None, :, :]

    obs = observatory.Observatory(latitude, longitude, array=[bl], freqs=freqs)
    obs.set_pointings(times)
    obs.set_fov(fov)
    obs.set_beam('airy', diameter=15)

    sky = sky_model.SkyModel(Nside=nside, freqs=freqs, data=gsm)
    visibs, times, bls = obs.make_visibilities(sky)

    # make sure peak is at time index 5, centered at transit time
    assert np.argmax(np.abs(visibs[:, 0, 0])) == 5


def test_az_za_astropy():
    """
    Check the calculated azimuth and zenith angle for a selection
    of HEALPix pixels against the corresponding astropy calculation.
    """

    Nside = 128

    altitude = 0.0
    loc = EarthLocation.from_geodetic(longitude, latitude, altitude)

    obs = observatory.Observatory(latitude, longitude)

    t0 = Time(2458684.453187554, format='jd')
    obs.set_fov(180)

    zen = AltAz(alt=Angle('90d'), az=Angle('0d'), obstime=t0, location=loc)

    zen_radec = zen.transform_to(ICRS)
    center = [zen_radec.ra.deg, zen_radec.dec.deg]
    northloc = EarthLocation.from_geodetic(lat='90.d', lon='0d', height=0.0)
    north_radec = AltAz(alt='90.0d', az='0.0d', obstime=t0, location=northloc).transform_to(ICRS)
    yvec = np.array([north_radec.ra.deg, north_radec.dec.deg])
    za, az, inds = obs.calc_azza(Nside, center, yvec, return_inds=True)

    ra, dec = hp.pix2ang(Nside, inds, lonlat=True)

    altaz_astropy = ICRS(ra=Angle(ra, unit='deg'), dec=Angle(dec, unit='deg')).transform_to(AltAz(obstime=t0, location=loc))

    za0 = altaz_astropy.zen.rad
    az0 = altaz_astropy.az.rad

#    Restore the lines below for visualization, if desired.
#    hmap = np.zeros(12*Nside**2) + hp.UNSEEN
#    hmap[inds] = np.unwrap(az0 - az)
#    import IPython; IPython.embed()
    print(np.degrees(za0 - za))
    assert np.allclose(za0, za, atol=1e-4)
    assert np.allclose(np.unwrap(az0 - az), 0.0, atol=3e-4)   # About 1 arcmin precision. Worst is at the southern horizon.
