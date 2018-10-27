
"""
    Generate visibilities for a single baseline.
"""
import numpy as np
from pspec_funcs import orthoslant_project
from astropy.constants import c
from astropy.time import Time
from astropy.coordinates import Angle, AltAz, EarthLocation, ICRS
import healpy as hp
from numba import jit

from pyuvdata import UVBeam

from line_profiler import LineProfiler
import atexit
import __builtin__ as builtins

# Line profiling
prof = LineProfiler()
builtins.__dict__['profile'] = prof
ofile = open('time_profiling.out', 'w')
atexit.register(ofile.close)
atexit.register(prof.print_stats, stream=ofile)

c_ms = c.to('m/s').value

def jy2Tstr(f, bm = 1.0):
    '''Return [K sr] / [Jy] vs. frequency (in Hz)
        Arguments:
            f = frequencies (Hz)
            bm = Reference area (defaults to 1 steradian)
    '''
    c_cmps = c_ms * 100.   # cm/s
    k_boltz = 1.380658e-16   # erg/K
    lam = c_cmps / f   #cm
    return 1e-23 * lam**2 / (2 * k_boltz * bm)

class powerbeam(UVBeam):
    """
    Interface for using beamfits files here.
    """

    ### TODO Develop and test!
    def __init__(self, beamfits_path=None):
        super(powerbeam, self).__init__()
        if beamfits_path is not None:
            self.read_beamfits(beamfits_path)

    def read_beamfits(self, beamfits_path):
        super(powerbeam, self).read_beamfits(beamfits_path)
        if not self.beam_type == 'power':
            self.efield_to_power(calc_cross_pols=False)
        self.interpolation_function = 'az_za_simple'

    def beam_val(self, az, za, freq_Hz=None):
        """
        az, za = radians
        """
        az = np.array(az)
        za = np.array(za)
        if isinstance(az, float):
            az = np.array([az])
        if isinstance(za, float):
            za = np.array([za])
        if freq_Hz is None:
            freq_Hz = self.freq_array[0,0]
        if isinstance(freq_Hz, float):
            freq_Hz = np.array([freq_Hz])
        interp_beam, interp_basis = self.interp(az_array = az, za_array = za, freq_array = freq_Hz, reuse_spline=True)
        return interp_beam[0,0,0,:]  #  pol XX

class analyticbeam(object):

    def __init__(self, beam_type, sigma=None):
        if beam_type not in ['uniform', 'gaussian']:
            raise NotImplementedError("Beam type " + str(beam_type) + " not available yet.")
        self.beam_type = beam_type
        if beam_type == 'gaussian':
            if sigma is None:
                raise KeyError("Sigma required for gaussian beam")
            self.sigma = sigma * np.pi / 180.  # deg -> radians

    def plot_beam(self, az, za):
        import pylab as pl
        fig = pl.figure()
        pl.imshow(self.beam_val(az, za))
        pl.show()

    def beam_val(self, az, za, freq_Hz=None):
        """
        az, za = radians

        """
        if self.beam_type == 'uniform':
            if isinstance(az, np.ndarray):
                return np.ones_like(az)
            return 1
        if self.beam_type == 'gaussian':
            return np.exp(-(za**2) / (2 * self.sigma**2))  # Peak normalized
@jit
def make_fringe(az, za, freq, enu):
    """
    az, za = Azimuth, zenith angle, radians
    freq = frequeny in Hz
    """
    pos_l = np.sin(az) * np.sin(za)
    pos_m = np.cos(az) * np.sin(za)
    pos_n = np.cos(za)
    lmn = np.array([pos_l, pos_m, pos_n])
    uvw = np.outer(enu, 1/(c_ms / freq))  # In wavelengths
    udotl = np.einsum("jk,jl->kl", lmn, uvw)
    fringe = np.cos(2 * np.pi * udotl) + (1j) * np.sin( 2 * np.pi * udotl)  # This is weirdly faster than np.exp
    return fringe

class baseline(object):

    def __init__(self, ant1_enu, ant2_enu):
        self.enu = ant1_enu - ant2_enu
        assert(self.enu.size == 3)

    def get_uvw(self, freq_Hz):
        return self.enu / (c_ms / float(freq_Hz))
    @profile
    def get_fringe(self, az, za, freq_Hz, degrees=False):
        if degrees:
            az *= np.pi/180.
            za *= np.pi/180.
        freq_Hz = freq_Hz.astype(float)
        return make_fringe(az, za, freq_Hz, self.enu)

    def plot_fringe(self, az, za, freq=None, degrees=False, pix=None, Nside=None):
        import pylab as pl
        if len(az.shape) == 1:
            # Healpix mode
            if pix is None or Nside is None:
                raise ValueError("Need to provide healpix indices and Nside")
            map0 = np.zeros(12*Nside**2)
            if isinstance(freq, np.ndarray):
                freq = np.array(freq[0])
            if isinstance(freq, float):
                freq = np.array(freq)

            vecs = hp.pixelfunc.pix2vec(Nside, pix)
            mean_vec = (np.mean(vecs[0]), np.mean(vecs[1]), np.mean(vecs[2]))    
            dt, dp = hp.rotator.vec2dir(mean_vec,lonlat=True)
            map0[pix] = self.get_fringe(az,za, freq, degrees=degrees)[:,0]
            hp.mollview(map0, rot=(dt, dp, 0))
            pl.show()
        else:
            fig = pl.figure()
            pl.imshow(self.get_fringe(az, za, freq=freq, degrees=degrees))
            pl.show()


class observatory:
    """
    Baseline, time, frequency, location (lat/lon), beam
    Assumes the shell lat/lon are ra/dec.
        Init time and freq structures.
        From times, get pointing centers

    """

    def __init__(self, latitude, longitude, array=None, freqs=None):
        """
        array = list of baseline objects (just one for now)
        """
        self.lat = latitude
        self.lon = longitude
        self.array = array
        self.pointings = None
        self.fov = None
        self.freqs = freqs
        if freqs is not None:
            self.Nfreqs = len(freqs)

    def set_pointings(self, time_arr):
        """
        Set the pointing centers (in ra/dec) based on array location and times.
            Dec = self.latitude
        RA  = What RA is at zenith at a given JD?
        """

        telescope_location = EarthLocation.from_geodetic(self.lon, self.lat)
        self.times_jd = time_arr
        centers = []
        for t in Time(time_arr, scale='utc', format='jd'):
            zen = AltAz(alt=Angle('90d'), az=Angle('0d'), obstime=t, location=telescope_location)
            zen_radec = zen.transform_to(ICRS)
            centers.append([zen_radec.ra.deg, zen_radec.dec.deg])
        self.pointing_centers = centers

    @profile
    def calc_azza(self, Nside, center, return_inds=False):
        """
        Set the az/za arrays.
            Center = lon/lat in degrees
            radius = selection radius in degrees
            return_inds = Return the healpix indices too
        """
        if self.fov is None:
            raise AttributeError("Need to set a field of view in degrees")
        radius = self.fov * np.pi / 180. * 1 / 2.
        cvec = hp.ang2vec(center[0], center[1], lonlat=True)
        pix = hp.query_disc(Nside, cvec, radius)
        vecs = hp.pix2vec(Nside, pix)
        vecs = np.array(vecs).T  # Shape (Npix_use, 3)

        colat = np.radians(90. - center[1])  # Colatitude, radians.
        xvec = [-cvec[1], cvec[0], 0] * 1 / np.sin(colat)  # From cross product
        yvec = np.cross(cvec, xvec)
        sdotx = np.array([np.dot(s, xvec) for s in vecs])
        sdotz = np.array([np.dot(s, cvec) for s in vecs])
        sdoty = np.array([np.dot(s, yvec) for s in vecs])
        za_arr = np.arccos(sdotz)
        az_arr = (np.arctan2(sdotx, sdoty) + np.pi) % (2 * np.pi)  # xy plane is tangent. Increasing azimuthal angle eastward, zero at North (y axis)
        if return_inds:
            return za_arr, az_arr, pix
        return za_arr, az_arr

    def set_fov(self, fov):
        """
        fov = field of view in degrees
        """
        self.fov = fov

    def set_beam(self, beam_type='uniform', **kwargs):
        self.beam = analyticbeam(beam_type, **kwargs)

    def get_observed_region(self, Nside):
        """
        Just as a check, get the pixels sampled by each snapshot.
        Returns a list of arrays of pixel numbers

        """
        try:
            assert self.pointing_centers is not None
            assert self.fov is not None
        except AssertionError:
            raise AssertionError("Pointing centers and FoV must be set.")

#        Npix, Nfreqs = shell.shape
#        Nside = np.sqrt(Npix/12)

        pixels = []
        for cent in self.pointing_centers:
            cent = hp.ang2vec(cent[0], cent[1], lonlat=True)
            hpx_inds = hp.query_disc(Nside, cent, 2 * np.sqrt(2) * np.radians(self.fov))
            pixels.append(hpx_inds)

        return pixels

    @profile
    def make_visibilities(self, shell):
        """
        Orthoslant project sections of the shell (fov=radius, looping over centers)
        Make beam cube and fringe cube, multiply and sum.
        shell (Npix, Nfreq) = healpix shell

        Takes a shell in Kelvin
        Returns visibility in Jy
        """
        if len(shell.shape) == 3:
            Nskies, Npix, Nfreqs = shell.shape
        else:
            Npix, Nfreqs = shell.shape

        assert Nfreqs == self.Nfreqs
        Nside = hp.npix2nside(Npix)
        bl = self.array[0]  # Make into a set of baselines and loop over them
        pix_area_sr = 4*np.pi/float(Npix)
        freqs = self.freqs
        conv_fact = jy2Tstr(np.array(freqs), bm = pix_area_sr)
        visibilities = []
        for c in self.pointing_centers:
            za_arr, az_arr = self.calc_azza(Nside, c)
            beam_cube = np.ones(az_arr.shape + (self.Nfreqs,))
            fringe_cube = np.ones_like(beam_cube, dtype=np.complex128)
            beam_one = self.beam.beam_val(az_arr, za_arr)
            beam_cube = np.repeat(beam_one[...,np.newaxis], Nfreqs, axis=1)
            fringe_cube = bl.get_fringe(az_arr, za_arr, np.array(freqs))
#            for fi in range(self.Nfreqs):
#                fringe_cube[..., fi] = bl.get_fringe(az_arr, za_arr, freqs[fi])
            radius = self.fov * np.pi / 180. * 1 / 2.
            cvec = hp.ang2vec(c[0], c[1], lonlat=True)
            pix = hp.query_disc(Nside, cvec, radius)
            ogrid = shell[..., pix, :]
            ## The last two axes are(pixels, freqs).
            # Optionally --- First axis = ensemble
            visibilities.append(np.sum(ogrid * beam_cube * fringe_cube, axis=-2))
        return np.array(visibilities, dtype=np.complex128)/conv_fact
