from healvis import visibility
from astropy.cosmology import WMAP9
import nose.tools as nt
import numpy as np
import healpy as hp
from healvis.data import DATA_PATH
import os
import copy

# HERA site
latitude = -30.7215277777
longitude = 21.4283055554


@nt.nottest
def test_pointings():
    t0 = 2451545.0  # Start at J2000 epoch
    dt_min = 20.0
    dt_days = dt_min * 1 / 60. * 1 / 24.  # 20 minutes in days

    time_arr = np.arange(20) * dt_days + t0
    obs = visibility.Observatory(latitude, longitude)

    obs.set_pointings(time_arr)

    ras = np.array([c[0] for c in obs.pointing_centers])
    decs = np.array([c[1] for c in obs.pointing_centers])
    ind = np.where(np.diff(ras) < 0)[0][0]
    ras[ind + 1:] += 360.   # Deal with 360 degree wrap

    dts = np.diff(ras) / 15.04106 * 60.
    nt.assert_true(np.allclose(dts, dt_min, atol=1e-1))  # Within 6 seconds. Not great...
    nt.assert_true(np.allclose(decs, latitude, atol=1e-1))


def test_az_za():
    """
    Check the calculated azimuth and zenith angle of a point exactly 5 deg east on the sphere (az = 90d, za = 5d)
    """
    Nside = 128
    obs = visibility.Observatory(latitude, longitude)
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

    bl = visibility.Baseline(ant1_enu, ant2_enu)

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
    shell[ind] *= visibility.jy2Tstr(freqs[0], bm=pix_area)  # K

    obs = visibility.Observatory(latitude, longitude, array=[bl], freqs=freqs)
    obs.pointing_centers = centers
    obs.times_jd = np.array([1])
    obs.set_fov(fov)
    obs.set_beam('uniform')

    visibs, times, bls = obs.make_visibilities(shell)
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

    bl = visibility.Baseline(ant1_enu, ant2_enu)

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
    shell[ind] *= visibility.jy2Tstr(freqs[0], pix_area)  # K

    obs = visibility.Observatory(latitude, longitude, array=[bl], freqs=freqs)
    obs.pointing_centers = [[phi, theta]]
    obs.times_jd = np.array([1])
    obs.set_fov(fov)
    resol = np.sqrt(pix_area)
    obs.set_beam('uniform')

    vis_calc, times, bls = obs.make_visibilities(shell)

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


def test_PowerBeam():
    # load it
    beam_path = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    P = visibility.PowerBeam(beam_path)
    freqs = np.arange(120e6, 160e6, 4e6)
    Nfreqs = len(freqs)

    # test frequency interpolation
    P2 = copy.deepcopy(P)
    P3 = P2.interp_freq(freqs, inplace=False, kind='linear')
    P2.interp_freq(freqs, inplace=True, kind='linear')
    # assert inplace and not inplace are consistent
    nt.assert_equal(P2, P3)
    # assert correct frequencies
    np.testing.assert_array_almost_equal(freqs, P3.freq_array[0])
    nt.assert_true(P3.bandpass_array.shape[1] == P3.Nfreqs == Nfreqs)

    # get beam value
    Npix = 20
    az = np.linspace(0, 2*np.pi, Npix, endpoint=False)
    za = np.linspace(0, 1, Npix, endpoint=False)
    b = P.beam_val(az, az, freqs, pol='XX')
    # check shape and rough value check (i.e. interpolation is near zenith as expected)
    nt.assert_equal(b.shape, (Npix, Nfreqs))
    nt.assert_true(np.isclose(b.max(), 1.0, atol=1e-3))

    # shift frequnecies by a delta and assert beams are EXACLTY the same (i.e. no freq interpolation)
    # delta must be larger than UVBeam._inter_freq tol, but small enough
    # to keep the same freq nearest neighbors
    b2 = P.beam_val(az, az, freqs + 1e6, pol='XX')
    np.testing.assert_array_almost_equal(b, b2)


def test_AnalyticBeam():
    freqs = np.arange(120e6, 160e6, 4e6)
    Nfreqs = len(freqs)
    Npix = 20
    az = np.linspace(0, 2*np.pi, Npix, endpoint=False)
    za = np.linspace(0, 1, Npix, endpoint=False)

    # Gaussian
    A = visibility.AnalyticBeam('gaussian', sigma=15.0)
    b = A.beam_val(az, za, freqs)
    nt.assert_equal(b.shape, (Npix, Nfreqs))  # assert array shape
    nt.assert_true(np.isclose(b[0, :], 1.0).all())  # assert peak normalized

    # Uniform
    A = visibility.AnalyticBeam('uniform')
    b = A.beam_val(az, za, freqs)
    nt.assert_equal(b.shape, (Npix, Nfreqs))  # assert array shape
    nt.assert_true(np.isclose(b, 1.0).all())

    # Airy
    A = visibility.AnalyticBeam('airy', diameter=15.0)
    b = A.beam_val(az, za, freqs)
    nt.assert_equal(b.shape, (Npix, Nfreqs))  # assert array shape
    nt.assert_true(np.isclose(b[0, :], 1.0).all())  # assert peak normalized

    # custom
    A = visibility.AnalyticBeam(visibility.airy_disk)
    b2 = A.beam_val(az, za, freqs, diameter=15.0)
    nt.assert_equal(b2.shape, (Npix, Nfreqs))  # assert array shape
    nt.assert_true(np.isclose(b2[0, :], 1.0).all())  # assert peak normalized
    np.testing.assert_array_almost_equal(b, b2)  # assert its the same as airy

    # exceptions
    A = visibility.AnalyticBeam("uniform")
    nt.assert_raises(NotImplementedError, visibility.AnalyticBeam, "foo")
    nt.assert_raises(KeyError, visibility.AnalyticBeam, "gaussian")
    nt.assert_raises(KeyError, visibility.AnalyticBeam, "airy")
