from healvis import observatory, sky_model, beam_model, utils, cosmology
from astropy.cosmology import WMAP9
from astropy.time import Time
import nose.tools as nt
import numpy as np
import healpy as hp
from healvis.data import DATA_PATH
import os
import copy

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
        nt.assert_true(isinstance(obs.beam, beam_model.AnalyticBeam))
        b = obs.beam.beam_val(az, za, freqs, pol='xx')
        nt.assert_equal(b.shape, (Npix, Nfreqs))

    # test Power set beam
    obs.set_beam(os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits"))
    nt.assert_true(isinstance(obs.beam, beam_model.PowerBeam))
    b = obs.beam.beam_val(az, za, freqs, pol='xx')
    nt.assert_equal(b.shape, (Npix, Nfreqs))


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
    nt.assert_equal(fringe.shape, (Npix, Nfreqs))

    # test fringe at zenith is 1.0 across all freqs
    nt.assert_true(np.isclose(fringe[0, :], 1.0).all())


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
    nt.assert_true(np.allclose(dts, dt_min, atol=1e-2))  # Within half a second.
    nt.assert_true(np.allclose(decs, latitude, atol=1e-1))  # Within 6 arcmin


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
    nt.assert_true(np.isclose(np.degrees(za[ind]), lon))
    nt.assert_true(np.isclose(np.degrees(az[ind]), (lat - 90) % 360))


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
    shell[ind] *= cosmo_funcs.jy2Tsr(freqs[0], bm=pix_area)  # K

    obs = observatory.Observatory(latitude, longitude, array=[bl], freqs=freqs)
    obs.pointing_centers = centers
    obs.times_jd = np.array([1])
    obs.set_fov(fov)
    obs.set_beam('uniform')

    sky = sky_model.SkyModel(Nside=nside, freqs=freqs, data=shell[np.newaxis, :, :])

    visibs, times, bls = obs.make_visibilities(sky)
    print(visibs)
    nt.assert_true(np.isclose(np.real(visibs), 1.0).all())  # Unit point source at zenith


def test_offzenith_vis():
    # Construct a shell with a single point source a known position off from zenith.
    #   Similar to test_vis_calc, but set the pointing center 5deg off from the zenith and adjust analytic calculation

    freqs = [1.0e8]
    Nfreqs = 1
    fov = 60
    ant1_enu = np.array([0, 0, 0])
    ant2_enu = np.array([0.0, 140.6, 0])

    bl = observatory.Baseline(ant1_enu, ant2_enu)

    Nside = 128
    ind = 9081
    center = list(hp.pix2ang(Nside, ind, lonlat=True))
    centers = [center]
    Npix = Nside**2 * 12
    pix_area = 4 * np.pi / float(Npix)
    shell = np.zeros((Npix, Nfreqs))

    # Choose an index 5 degrees off from the pointing center
    phi, theta = hp.pix2ang(Nside, ind, lonlat=True)
    ind = hp.ang2pix(Nside, phi, theta - 5, lonlat=True)
    shell[ind] = 1  # Jy/pix
    shell[ind] *= cosmo_funcs.jy2Tsr(freqs[0], bm=pix_area)  # K

    obs = observatory.Observatory(latitude, longitude, array=[bl], freqs=freqs)
    obs.pointing_centers = [[phi, theta]]
    obs.times_jd = np.array([1])
    obs.set_fov(fov)
    resol = np.sqrt(pix_area)
    obs.set_beam('uniform')

    sky = sky_model.SkyModel(Nside=Nside, freqs=np.array(freqs), data=shell)

    vis_calc, times, bls = obs.make_visibilities(sky)

    phi_new, theta_new = hp.pix2ang(Nside, ind, lonlat=True)
    src_az, src_za = np.radians(phi - phi_new), np.radians(theta - theta_new)
    src_l = np.sin(src_az) * np.sin(src_za)
    src_m = np.cos(src_az) * np.sin(src_za)
    src_n = np.cos(src_za)
    u, v, w = bl.get_uvw(freqs[0])

    vis_analytic = (1) * np.exp(2j * np.pi * (u * src_l + v * src_m + w * src_n))

    print(vis_analytic)
    print(vis_calc)

    nt.assert_true(np.isclose(vis_calc, vis_analytic, atol=1e-3).all())


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
    nt.assert_equal(np.argmax(np.abs(visibs[:, 0, 0])), 5)
