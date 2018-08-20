
"""
    Generate visibilities for a single baseline.

    Choose a random point at a given latitude as a start. For each time, rotate by 15deg/hour.
        orthoslant_project returns a box (Nx, Ny, Nz) where each xy pixel is equal area to the Healpix pixels and each df is the channel width.
        Need: A fringe, a beam (altaz), resolution (obtain from the healpix shell)
"""
import numpy as np
from pspec_funcs import orthoslant_project
from astropy.constants import c
from astropy.time import Time
from astropy.coordinates import Angle, AltAz, EarthLocation, ICRS
import healpy as hp

c_ms = c.to('m/s').value

class analyticbeam(object):

    def __init__(self, beam_type, sigma=None):
        if beam_type not in ['uniform', 'gaussian']:
            raise NotImplementedError("Beam type "+str(beam_type)+" not available yet.")
        self.beam_type = beam_type
        if beam_type == 'gaussian':
            if sigma is None:
                raise KeyError("Sigma required for gaussian beam")
            self.sigma = sigma * np.pi/180.  # deg -> radians

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
            return 1
#            try:
#                return np.ones((az.shape[0], az.shape[1], freq_Hz.shape[0]))
#            except AttributeError:
#                return 1.
        if self.beam_type == 'gaussian':
            return np.exp(-(za**2)/(2*self.sigma**2))  #Peak normalized


class baseline(object):

    def __init__(self, ant1_enu, ant2_enu):
        self.enu = ant1_enu - ant2_enu
        assert(self.enu.size == 3)

    def get_uvw(self, freq_Hz):
        return self.enu / ( c_ms/ float(freq_Hz))

    def get_fringe(self, az, za, freq_Hz, degrees=False):
#        if degrees:
#            az *= np.pi/180.
#            za *= np.pi/180.
#        pos_l = np.sin(az * np.sin(za))
#        pos_m = np.cos(az * np.sin(za))
#        pos_n = np.cos(za)
#        lmn = np.array([pos_l, pos_m, pos_n])
#        uvw = np.outer(self.enu, 1/(c_ms / freq_Hz.astype(float)))  # In wavelengths
#        udotl = np.einsum("ijk,il->jkl", lmn, uvw)
#        return np.exp(2j * np.pi * udotl)
        pos_l = np.sin(az) * np.sin(za)
        pos_m = np.cos(az) * np.sin(za)
        pos_n = np.cos(za)
        self.lmn = np.array([pos_l, pos_m, pos_n])
        uvw = self.enu / (c_ms / float(freq_Hz))
        print 'uvw: ', uvw
        return np.exp(2j * np.pi * (pos_l * uvw[0] + pos_m * uvw[1] + pos_n * uvw[2]))


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
        self.az_arr = None
        self.za_arr = None
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

    def calc_azza(self, resol):
        """
        From the field of view, resolution, calculate the az and za matrices

        resol = pixel size in radians
        """
        if self.fov is None:
            raise AttributeError("Need to set a field of view (degrees)")
        radius = self.fov * np.pi/180.  #deg to rad
        extent = 2*np.floor(radius/resol).astype(int)   #Exactly the same as in pspec_funcs
        pixind = np.arange(-extent/2, extent/2)
#        import IPython; IPython.embed()
        x_arr, y_arr = np.meshgrid(pixind*resol, pixind*resol)
        self.za_arr = np.sqrt(x_arr**2 + y_arr**2)
        self.az_arr = (np.arctan2(y_arr, x_arr) - np.pi/2.)%(2*np.pi)

    def set_fov(self, fov):
        """
        fov = field of view in degrees
        """
        self.fov = fov

    def set_beam(self, beam_type = 'uniform', **kwargs):
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
            hpx_inds = hp.query_disc(Nside, cent, 2*np.sqrt(2)*np.radians(self.fov))
            pixels.append(hpx_inds)

        return pixels


    def make_visibilities(self, shell):
        """
        Orthoslant project sections of the shell (fov=radius, looping over centers)
        Make beam cube and fringe cube, multiply and sum.
	shell (Npix, Nfreq) = healpix shell

        """
	Npix, Nfreqs = shell.shape
        assert Nfreqs == self.Nfreqs

        if self.az_arr is None:
            self.calc_azza(np.sqrt(4*np.pi/float(Npix)))
        extent = self.az_arr.shape[0]

	bl = self.array[0]
        beam_cube = np.ones((extent, extent, self.Nfreqs))
        fringe_cube = np.ones_like(beam_cube, dtype=np.complex128)
        az_arr = self.az_arr
        za_arr = self.za_arr
        freqs = self.freqs
        for fi in range(self.Nfreqs):
            beam_cube[:,:,fi] = self.beam.beam_val(az_arr, za_arr, freqs[fi])
            fringe_cube[:,:,fi] = bl.get_fringe(az_arr, za_arr, freqs[fi])

        visibilities = []
        for c in self.pointing_centers:
            ogrid = orthoslant_project(shell, c, self.fov, degrees=True)
#            import IPython; IPython.embed()
            visibilities.append(np.sum( ogrid * beam_cube * fringe_cube, axis=(0,1)))
        return np.array(visibilities)
