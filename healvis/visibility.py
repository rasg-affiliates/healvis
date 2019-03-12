"""
Generate visibilities from a HEALPix shell.
"""
from __future__ import print_function

import numpy as np
from astropy.constants import c
from astropy.time import Time
from astropy.coordinates import Angle, AltAz, EarthLocation, ICRS
import healpy as hp
from numba import jit
import multiprocessing as mp
import sys
import resource
import time
from scipy.special import j1
import copy

from pyuvdata import UVBeam
from pyuvsim.utils import progsteps

from .skymodel import skymodel

# Line profiling
#from line_profiler import LineProfiler
#import atexit
#import __builtin__ as builtins
#prof = LineProfiler()
#builtins.__dict__['profile'] = prof
#ofile = open('time_profiling.out', 'w')
# atexit.register(ofile.close)
#atexit.register(prof.print_stats, stream=ofile)

c_ms = c.to('m/s').value


def jy2Tstr(f, bm=1.0):
    '''Return [K sr] / [Jy] vs. frequency (in Hz)
        Arguments:
            f = frequencies (Hz)
            bm = Reference area (defaults to 1 steradian)
    '''
    c_cmps = c_ms * 100.   # cm/s
    k_boltz = 1.380658e-16   # erg/K
    lam = c_cmps / f  # cm
    return 1e-23 * lam**2 / (2 * k_boltz * bm)


def airy_disk(za_array, freqs, diameter=15.0):
    """
    Airy disk function for an antenna of specified diameter.

    Args:
        za_array: 1D ndarray, zenith angle [radians]
        freqs: 1D array, observing frequencies [Hz]
        diameter: float, antenna diameter [meters]

    Returns:
        beam: 2D array of shape (Nfreqs, Nza_array)
    """
    Nfreqs, Nza_array = len(freqs), len(za_array)
    xvals = diameter / 2. * np.sin(za_array.reshape(1, -1)) * 2. * np.pi * freqs.reshape(-1, 1) / c.value
    zeros = np.isclose(xvals, 0.0)
    beam = (2.0 * np.true_divide(j1(xvals), xvals, where=~zeros))**2.0
    beam[zeros] = 1.0

    return beam


class PowerBeam(UVBeam):
    """
    Interface for using beamfits files.
    """

    def __init__(self, beamfits=None):
        """
        Initialize a PowerBeam object

        Args:
            beamfits : str or UVBeam
                A path to beamfits file or a UVBeam object
            interp_mode : str
                Interpolation method. See pyuvdata.UVBeam
        """
        # initialize
        super(PowerBeam, self).__init__()

        # read a beamfits
        if beamfits is not None:
            self.read_beamfits(beamfits)

        # convert to power if efield
        if self.beam_type == 'efield':
            self.efield_to_power()

    def interp_freq(self, freqs, inplace=False, kind='linear', run_check=True):
        """
        Interpolate object across frequency.

        Args:
            freqs: 1D frequency array [Hz]
            inplace: bool, if True edit data in place, otherwise return a new UVBeam
            kind: str, interpolation method. See scipy.interpolate.interp1d
            run_check: bool, if True run attribute check on output object

        Returns:
            If not inplace, returns UVBeam object with interpolated frequencies
        """
        # make a new object
        if inplace:
            new_beam = self
        else:
            new_beam = copy.deepcopy(self)

        interp_data, interp_bp = super(PowerBeam, self)._interp_freq(freqs, kind=kind)
        new_beam.data_array = interp_data
        new_beam.Nfreqs = interp_data.shape[3]
        new_beam.freq_array = freqs.reshape(1, -1)
        new_beam.bandpass_array = interp_bp
        if hasattr(new_beam, 'saved_interp_functions'):
            delattr(new_beam, 'saved_interp_functions')

        if run_check:
            new_beam.check()

        if not inplace:
            return new_beam

    def beam_val(self, az, za, freqs, pol='pI', **kwargs):
        """
        Fast interpolation of power beam across the sky,
        using UVBeam "simple" interpolation methods.

        Note: this does not interpolate frequencies, instead
        it just takes the nearest neighbor of each requested
        frequency for speed. To interpolate the frequency axis
        ahead of time, see self.interp_freq().

        Args:
            az : float or ndarray, azimuth angle [radian], must have len(za)
            za : float or ndarray, zenith angle [radian], must have len(az)
            freqs : float or ndarray, frequencies [Hz]
            pol : str, requested visibility polarization, Ex: 'XX' or 'pI'.

        Returns:
            beam_value : ndarray of beam power, with shape (Nfreqs, Naz_array)
        """
        # type checks
        assert self.beam_type == 'power', "beam_type must be power. See efield_to_power()"
        if isinstance(az, (float, np.float, int, np.int)):
            az = np.array([az])
        if isinstance(za, (float, np.float, int, np.int)):
            za = np.array([za])
        az = np.asarray(az)
        za = np.asarray(za)

        if self.pixel_coordinate_system == 'az_za':
            self.interpolation_function = 'az_za_simple'
        elif self.pixel_coordinate_system == 'healpix':
            self.interpolation_function = 'healpix_simple'

        if freqs is None:
            freqs = self.freq_array[0]
        else:
            # get nearest neighbor of each reqeuested frequency
            if isinstance(freqs, (float, np.float, int, np.int)):
                freqs = np.array([freqs])
            freqs = np.asarray(freqs)
            assert freqs.ndim == 1, "input freqs array must be 1-dimensional"
            freq_dists = np.abs(self.freq_array - freqs.reshape(-1, 1))
            nearest_dist = np.min(freq_dists, axis=1)
            nearest_inds = np.argmin(freq_dists, axis=1)
            freqs = self.freq_array[0, nearest_inds]

        assert isinstance(pol, str), "requested polarization must be a single string"

        # interpolate
        if self.pixel_coordinate_system == 'az_za':
            # azimuth - zenith angle interpolation
            interp_beam, interp_basis = self._interp_az_za_rect_spline(az_array=az, za_array=za, freq_array=freqs, reuse_spline=True, polarizations=[pol])

        elif self.pixel_coordinate_system == 'healpix':
            # healpix interpolation
            interp_beam, interp_basis = self._interp_healpix_bilinear(az_array=az, za_array=za, freq_array=freqs, polarizations=[pol])

        return interp_beam[0, 0, 0]


class AnalyticBeam(object):

    def __init__(self, beam_type, sigma=None, diameter=None):
        """
        Instantiate an analytic beam model.

        Currently this class only supports single-polarization beam models.

        Args:
            beam_type : str or callable, type of beam to use. options=['uniform', 'gaussian', 'airy', callable]
            sigma : float, standard deviation [degrees] for gaussian beam
            diameter : float, dish diameter [meter] used for airy beam

        Notes:
            Uniform beam is a flat-top beam across the entire sky.
            Gaussian beam is a frequency independent, peak normalized Gaussian
                with respect to zenith angle.
            Airy beam uses the dish diameter to set the airy beam width as a function of frequency.
            callable is any function that takes (za_array, freqs) in units [radians, Hz] respectively
                and returns an ndarray of shape (Nfreqs, Nza_array) with power beam values.
        """
        if beam_type not in ['uniform', 'gaussian', 'airy'] and not callable(beam_type):
            raise NotImplementedError("Beam type " + str(beam_type) + " not available yet.")
        self.beam_type = beam_type
        if beam_type == 'gaussian':
            if sigma is None:
                raise KeyError("Sigma required for gaussian beam")
            self.sigma = sigma * np.pi / 180.  # deg -> radians
        elif beam_type == 'airy':
            if diameter is None:
                raise KeyError("Dish diameter required for airy beam")
            self.diameter = diameter

    def plot_beam(self, az, za, freqs, **kwargs):
        # TODO: needs development, and testing coverage?
        fig = pl.figure()
        pl.imshow(self.beam_val(az, za, freqs, **kwargs))
        pl.show()

    def beam_val(self, az, za, freqs, **kwargs):
        """
        Evaluation of an analytic beam model.

        Args:
            az : float or ndarray, azimuth angle [radian], must have len(za)
            za : float or ndarray, zenith angle [radian], must have len(az)
            freqs : float or ndarray, frequencies [Hz]
            kwargs : keyword arguments to pass if self.beam_type is callable

        Returns:
            beam_value : ndarray of beam power, with shape (Nfreqs, Naz_array)
        """
        if self.beam_type == 'uniform':
            if isinstance(az, np.ndarray):
                beam_value = np.ones((len(freqs), len(az)), dtype=np.float)
            else:
                beam_value = 1.0
        elif self.beam_type == 'gaussian':
            beam_value = np.exp(-(za**2) / (2 * self.sigma**2))  # Peak normalized
            beam_value = np.repeat(beam_value[None], len(freqs), axis=0)
        elif self.beam_type == 'airy':
            beam_value = airy_disk(za, freqs, diameter=self.diameter)
        elif callable(self.beam_type):
            beam_value = self.beam_type(za, freqs, **kwargs)

        return beam_value


@jit
def make_fringe(az, za, freq, enu):
    """
    az, za = Azimuth, zenith angle, radians
    freq = frequeny in Hz
    enu = baseline vector in meters
    """
    pos_l = np.sin(az) * np.sin(za)
    pos_m = np.cos(az) * np.sin(za)
    pos_n = np.cos(za)
    lmn = np.vstack((pos_l, pos_m, pos_n))
    uvw = np.outer(enu, 1 / (c_ms / freq))  # In wavelengths
    udotl = np.einsum("jk,jl->kl", lmn, uvw)
    fringe = np.cos(2 * np.pi * udotl) + (1j) * np.sin(2 * np.pi * udotl)  # This is weirdly faster than np.exp
    return fringe


class Baseline(object):

    def __init__(self, ant1_enu, ant2_enu):
        if not isinstance(ant1_enu, np.ndarray):
            ant1_enu = np.array(ant1_enu)
            ant2_enu = np.array(ant2_enu)
        self.enu = ant1_enu - ant2_enu
        assert(self.enu.size == 3)

    def get_uvw(self, freq_Hz):
        return self.enu / (c_ms / float(freq_Hz))

    def get_fringe(self, az, za, freq_Hz, degrees=False):
        if degrees:
            az *= np.pi / 180.
            za *= np.pi / 180.
        freq_Hz = freq_Hz.astype(float)
        return make_fringe(az, za, freq_Hz, self.enu)

    def plot_fringe(self, az, za, freq=None, degrees=False, pix=None, Nside=None):
        import pylab as pl
        if len(az.shape) == 1:
            # Healpix mode
            if pix is None or Nside is None:
                raise ValueError("Need to provide healpix indices and Nside")
            map0 = np.zeros(12 * Nside**2)
            if isinstance(freq, np.ndarray):
                freq = np.array(freq[0])
            if isinstance(freq, float):
                freq = np.array(freq)

            vecs = hp.pixelfunc.pix2vec(Nside, pix)
            mean_vec = (np.mean(vecs[0]), np.mean(vecs[1]), np.mean(vecs[2]))
            dt, dp = hp.rotator.vec2dir(mean_vec, lonlat=True)
            map0[pix] = self.get_fringe(az, za, freq, degrees=degrees)[:, 0]
            hp.mollview(map0, rot=(dt, dp, 0))
            pl.show()
        else:
            fig = pl.figure()
            pl.imshow(self.get_fringe(az, za, freq=freq, degrees=degrees))
            pl.show()


class Observatory:
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
            Dec = self.lat
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
        sdotx = np.tensordot(vecs, xvec, 1)
        sdotz = np.tensordot(vecs, cvec, 1)
        sdoty = np.tensordot(vecs, yvec, 1)
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
        self.beam = AnalyticBeam(beam_type, **kwargs)

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

    def vis_calc(self, pcents, tinds, shell, vis_array, Nfin):
        if len(pcents) == 0:
            return
        for count, c in enumerate(pcents):
            memory_usage_GB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            za_arr, az_arr, pix = self.calc_azza(self.Nside, c, return_inds=True)
            beam_cube = self.beam.beam_val(az_arr, za_arr, self.freqs)
            beam_cube = np.moveaxis(beam_cube, -1, 0)
            for bi, bl in enumerate(self.array):
                fringe_cube = bl.get_fringe(az_arr, za_arr, self.freqs)
                vis = np.sum(shell[..., pix, :] * beam_cube * fringe_cube, axis=-2)
                vis_array.put((tinds[count], bi, vis.tolist()))
            with Nfin.get_lock():
                Nfin.value += 1
            if mp.current_process().name == 1:
                #        print('Mem: {}GB'.format(memory_usage_GB))
                #        sys.stdout.flush()
                print('Finished {:d}, Elapsed {:.2f}sec, MaxRSS {}GB '.format(Nfin.value, time.time() - self.time0, memory_usage_GB))
                sys.stdout.flush()

    def make_visibilities(self, shell, Nprocs=1):
        """
        Orthoslant project sections of the shell (fov=radius, looping over centers)
        Make beam cube and fringe cube, multiply and sum.
        shell (Npix, Nfreq) = healpix shell, as an mparray (multiprocessing shared array)

        Takes a shell in Kelvin
        Returns visibility in Jy
        """
        if len(shell.shape) == 3:
            Nskies, Npix, Nfreqs = shell.shape
        else:
            Npix, Nfreqs = shell.shape

        assert Nfreqs == self.Nfreqs
        self.time0 = time.time()
        Nside = hp.npix2nside(Npix)
        Nbls = len(self.array)
        self.Nside = Nside
        pix_area_sr = 4 * np.pi / float(Npix)
        self.freqs = np.array(self.freqs)
        conv_fact = jy2Tstr(np.array(self.freqs), bm=pix_area_sr)
        self.Ntimes = len(self.pointing_centers)
        pcenter_list = np.array_split(self.pointing_centers, Nprocs)
        time_inds = np.array_split(range(self.Ntimes), Nprocs)
        procs = []
        man = mp.Manager()
        vis_array = man.Queue()
        Nfin = mp.Value('i', 0)
        prog = progsteps(maxval=self.Ntimes)
        for pi in range(Nprocs):
            p = mp.Process(name=pi, target=self.vis_calc, args=(pcenter_list[pi], time_inds[pi], shell, vis_array, Nfin))
            p.start()
            procs.append(p)
        while (Nfin.value < self.Ntimes) and np.any([p.is_alive() for p in procs]):
            prog.update(Nfin.value)
        prog.finish()
        visibilities = []
        time_inds, baseline_inds = [], []

        for (ti, bi, varr) in iter(vis_array.get, None):
            visibilities.append(varr)
            N = len(varr)
            time_inds += [ti]
            baseline_inds += [bi]
            if vis_array.empty():
                break

        srt = np.lexsort((time_inds, baseline_inds))
        time_inds = np.array(time_inds)[srt]
        visibilities = np.array(visibilities)[srt]      # Shape (Nblts, Nskies, Nfreqs)
        time_array = self.times_jd[time_inds]
        baseline_array = np.array(baseline_inds)[srt]

        # Time and baseline arrays are now Nblts
        return visibilities / conv_fact, time_array, baseline_array
