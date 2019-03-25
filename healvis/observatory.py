"""
Generate visibilities from a HEALPix shell.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from astropy.time import Time
from astropy.constants import c
from astropy.coordinates import Angle, AltAz, EarthLocation, ICRS
from astropy import units
import healpy as hp
from numba import jit
import multiprocessing as mp
import sys
import resource
import time
import copy

from .beam_model import PowerBeam, AnalyticBeam
from .utils import jy2Tstr

# Line profiling
# from line_profiler import LineProfiler
# import atexit
# import __builtin__ as builtins
# prof = LineProfiler()
# builtins.__dict__['profile'] = prof
# ofile = open('time_profiling.out', 'w')
# atexit.register(ofile.close)
# atexit.register(prof.print_stats, stream=ofile)


c_ms = c.to('m/s').value


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
        telescope_location = EarthLocation.from_geodetic(self.lon * units.degree, self.lat * units.degree)
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

    def set_beam(self, beam='uniform', **kwargs):
        """
        Set the beam of the array.

        Args:
            beam : str
                Input to PowerBeam or AnalyticBeam. If beam is
                a viable input to AnalyticBeam, then instantiates
                an AnalyticBeam, otherwise assumes beam is a filepath
                to a beamfits and instantiates a PowerBeam.

            kwargs : keyword arguments
                kwargs to pass to AnalyticBeam instantiation.
        """
        if beam in ['uniform', 'gaussian', 'airy'] or callable(beam):
            self.beam = AnalyticBeam(beam, **kwargs)

        else:
            self.beam = PowerBeam(beam)

    def beam_sq_int(self, freqs, Nside, pointing, beam_pol='pI'):
        """
        Get the integral of the squared antenna primary beam power across the sky.

        Args:
            freqs : 1D ndarray
                Frequencies [Hz]
            Nside : int
                Nside of healpix map to use in integral
            pointing : len-2 list
                Pointing center [Dec, RA] in J2000 degrees
        """
        za, az = self.calc_azza(Nside, pointing)
        beam_sq_int = np.sum(self.beam.beam_val(az, za, freqs, pol=beam_pol)**2, axis=0)
        om = 4 * np.pi / (12.0 * Nside)
        beam_sq_int = beam_sq_int * om

        return beam_sq_int


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

    def vis_calc(self, pcents, tinds, shell, vis_array, Nfin, beam_pol='pI'):
        if len(pcents) == 0:
            return
        for count, c in enumerate(pcents):
            memory_usage_GB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            za_arr, az_arr, pix = self.calc_azza(self.Nside, c, return_inds=True)
            beam_cube = self.beam.beam_val(az_arr, za_arr, self.freqs, pol=beam_pol)
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

    def make_visibilities(self, shell, Nprocs=1, beam_pol='pI'):
        """
        Make beam cube and fringe cube, multiply and sum.
        shell (Npix, Nfreq) = healpix shell, as an mparray (multiprocessing shared array)

        Takes a shell in Kelvin
        Returns visibility in Jy
        """

        Nskies = shell.Nskies
        Nside = shell.Nside
        Npix = shell.Npix
        Nfreqs = shell.Nfreqs
        pix_area_sr = shell.pix_area_sr

        assert Nfreqs == self.Nfreqs

        self.time0 = time.time()
        Nbls = len(self.array)
        self.Nside = Nside
        self.freqs = np.array(self.freqs)
        conv_fact = jy2Tstr(np.array(self.freqs), bm=pix_area_sr)
        self.Ntimes = len(self.pointing_centers)

        pcenter_list = np.array_split(self.pointing_centers, Nprocs)
        time_inds = np.array_split(range(self.Ntimes), Nprocs)
        procs = []
        man = mp.Manager()
        vis_array = man.Queue()
        Nfin = mp.Value('i', 0)
        for pi in range(Nprocs):
            p = mp.Process(name=pi, target=self.vis_calc, args=(pcenter_list[pi], time_inds[pi], shell.data, vis_array, Nfin), kwargs=dict(beam_pol=beam_pol))
            p.start()
            procs.append(p)
        while (Nfin.value < self.Ntimes) and np.any([p.is_alive() for p in procs]):
            continue
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
