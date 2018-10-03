from eorsky import pspec_funcs, comoving_voxel_volume, comoving_radial_length, comoving_transverse_length
from eorsky import visibility
from astropy.cosmology import WMAP9
import nose.tools as nt
import numpy as np
import healpy as hp

# HERA site
latitude  = -30.7215277777
longitude =  21.4283055554

def test_pointings():
    t0 = 2451545.0      #Start at J2000 epoch
    dt_min = 20.0
    dt_days = dt_min * 1/60. * 1/24.  # 20 minutes in days

    time_arr = np.arange(20) * dt_days + t0
    obs = visibility.observatory(latitude, longitude)

    obs.set_pointings(time_arr)

    ras = np.array([ c[0] for c in obs.pointing_centers ])
    decs = np.array([ c[1] for c in obs.pointing_centers ])
    ind = np.where(np.diff(ras)<0)[0][0]
    ras[ind+1:] += 360.   # Deal with 360 degree wrap
    
    dts = np.diff(ras)/15.04106 * 60.
    nt.assert_true(np.allclose(dts, dt_min, atol=1e-1))   #Within 6 seconds. Not great...
    nt.assert_true(np.allclose(decs, latitude, atol=1e-1))   # Close enough for my purposes, for now.

def test_az_za():
    """
    Check the calculated azimuth and zenith angle of a point exactly 5 deg east on the sphere (az = 90d, za = 5d)
    """
    Nside=128
    obs = visibility.observatory(latitude, longitude)
    center = [0, 0]
    lon, lat = [5,0]
    ind0 = hp.ang2pix(Nside, lon, lat, lonlat=True)
    lon, lat = hp.pix2ang(Nside, ind0, lonlat=True)
    cvec = hp.ang2vec(center[0],center[1], lonlat=True)
    radius = np.radians(10.)
    obs.set_fov(20)
    pix = hp.query_disc(Nside, cvec, radius)
    za, az = obs.calc_azza(Nside, center)
    ind = np.where(pix == ind0)
    print(np.degrees(za[ind]), np.degrees(az[ind]))
    print(lon, lat)
    nt.assert_true(np.isclose(np.degrees(za[ind]), lon))
    nt.assert_true(np.isclose(np.degrees(az[ind]), (lat-90)%360))

def test_vis_calc():
    # Construct a shell with a single point source at the zenith and confirm against analytic calculation.

    ant1_enu = np.array([0, 0, 0])
    ant2_enu = np.array([0.0, 14.6, 0])
    
    bl = visibility.baseline(ant1_enu, ant2_enu)

    freqs = np.array([1e8])
    nfreqs = 1

    fov=20  #Deg

    ## Longitude/Latitude in degrees.

    nside=128
    ind = 10
    center = list(hp.pix2ang(nside, ind, lonlat=True))
    centers = [center]
    npix = nside**2 * 12
    shell = np.zeros((npix, nfreqs))
    pix_area = 4* np.pi / float(npix)
    shell[ind] = 1  # Jy/pix
    shell[ind] *= visibility.jy2Tstr(freqs[0], bm=pix_area)  # K

    obs = visibility.observatory(latitude, longitude, array=[bl], freqs=freqs)
    obs.pointing_centers = centers
    obs.set_fov(fov)
#    resol = np.sqrt(4*np.pi/float(npix))
    obs.set_beam('uniform')

    visibs = obs.make_visibilities(shell)
    print visibs

    nt.assert_true(np.real(visibs) == 1.0)   #Unit point source at zenith

def test_offzenith_vis():
    # Construct a shell with a single point source a known position off from zenith.
    #   Similar to test_vis_calc, but set the pointing center 5deg off from the zenith and adjust analytic calculation

    freqs = [1.0e8]
    Nfreqs = 1
    fov = 60
    ant1_enu = np.array([0, 0, 0])
    ant2_enu = np.array([0.0, 140.6, 0])
    
    bl = visibility.baseline(ant1_enu, ant2_enu)

    Nside=128
    ind = 9081
    center = list(hp.pix2ang(Nside, ind, lonlat=True))
    centers = [center]
    Npix = Nside**2 * 12
    pix_area = 4*np.pi/float(Npix)
    shell = np.zeros((Npix, Nfreqs))

    # Choose an index 5 degrees off from the pointing center
    phi, theta = hp.pix2ang(Nside, ind, lonlat=True)
    ind = hp.ang2pix(Nside, phi, theta-5, lonlat=True)
    shell[ind] = 1  # Jy/pix
    shell[ind] *= visibility.jy2Tstr(freqs[0], pix_area)  # K

    obs = visibility.observatory(latitude, longitude, array=[bl], freqs=freqs)
    obs.pointing_centers = [[phi, theta]]
    obs.set_fov(fov)
    resol = np.sqrt(pix_area)
    obs.set_beam('uniform')

    vis_calc = obs.make_visibilities(shell)

    phi_new, theta_new = hp.pix2ang(Nside, ind, lonlat=True)
    src_az, src_za = np.radians(phi - phi_new), np.radians(theta-theta_new)
    src_l = np.sin(src_az) * np.sin(src_za)
    src_m = np.cos(src_az) * np.sin(src_za)
    src_n = np.cos(src_za)
    u, v, w = bl.get_uvw(freqs[0])

    vis_analytic = (1) * np.exp(2j * np.pi * (u*src_l + v*src_m + w*src_n))

    print(vis_analytic)
    print(vis_calc)

    nt.assert_true(np.isclose(vis_analytic, vis_calc, atol=1e-3))


def test_hera_beam():
    beam_path = '/users/alanman/data/alanman/NickFagnoniBeams/HERA_NicCST_fullfreq.uvbeam'
    beam = visibility.powerbeam(beam_path)

    Nside=64
    center = [0,0]
    obs = visibility.observatory(latitude, longitude)
    obs.set_fov(40)
    za, az, inds= obs.calc_azza(Nside, center, return_inds=True)
    bv = beam.beam_val(az, za)
    beam.to_healpix(Nside)
    pix2 = hp.query_disc(Nside, [0,0,0], np.radians(20))
    map1 = np.zeros(12*Nside**2)
    map1[pix2] = beam.data_array[0,0,0,0][pix2]

    nt.assert_equal(np.around(np.sum(map1)), np.around(np.sum(map0)))   # Close to nearest integer

    ## Question --- Which beam component is the right one? (Can I construct pseudo-stokes I?)
#    import IPython; IPython.embed()


if __name__ == '__main__':
    #est_offzenith_vis()
    test_hera_beam()
    #test_vis_calc()
    #test_az_za()
