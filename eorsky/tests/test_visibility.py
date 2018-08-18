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

def test_vis_calc():
    # Construct a shell with a single point source at the zenith and confirm against analytic calculation.

    ant1_enu = np.array([0, 0, 0])
    ant2_enu = np.array([0.0, 14.6, 0])
    
    bl = visibility.baseline(ant1_enu, ant2_enu)

    freqs = np.array([1e8])
    nfreqs = 1

    fov=20  #Deg
    centers = [[0,0]]

    nside=64
    npix = nside**2 * 12
    shell = np.zeros((npix, nfreqs))
    ind = hp.ang2pix(nside, centers[0][1], centers[0][0], lonlat=True)
    shell[ind] = 1

    obs = visibility.observatory(latitude, longitude, array=[bl], freqs=freqs)
    obs.pointing_centers = centers
    obs.set_fov(fov)
    resol = np.sqrt(4*np.pi/float(npix))
    obs.calc_azza(resol)
    obs.set_beam('uniform')

    visibs = obs.make_visibilities(shell)
    print visibs

    nt.assert_true(np.real(visibs) == 1.0)   #Unit point source at zenith
     
#!!! More tests:
#        Conirm the fringe against an analytic value
#        Check visibility for off-zenith point source


# scripts:
#   Script to calculate visibilities from a gaussian sky given gaussian beams of different widths. --- Confirm the relationship between beam width and covariance matrices
#   Get overlap between fields of view and primary beams for different centers --- how does that relate with correlation?
#   Look at effect of resolution and time cadence, with varying beam width
#   Covariance binning of random visibilities

