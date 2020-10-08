# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

from __future__ import absolute_import, division, print_function

import numpy as np
import multiprocessing as mp
import sys
import resource
import warnings
import time
import copy
import healpy as hp
from astropy.time import Time
from astropy.constants import c
from astropy.coordinates import Angle, AltAz, EarthLocation, ICRS
from astropy import units

from .beam_model import PowerBeam, AnalyticBeam
from .utils import jy2Tsr, mparray
from .cosmology import c_ms

# -----------------------
# Classes and methods to calculate visibilities from HEALPix maps.
# -----------------------


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

    def __init__(self, ant1_enu=None, ant2_enu=None, ant1=None, ant2=None, enu_vec=None):
        self.ant1 = ant1	# Antenna indexes, for indexing beam list if necessary
        self.ant2 = ant2        # Must be numbers from 0 ..., so the right beams are found.
        if enu_vec is not None:
            self.enu = enu_vec
        else:
            if not isinstance(ant1_enu, np.ndarray):
                ant1_enu = np.array(ant1_enu)
                ant2_enu = np.array(ant2_enu)
            self.enu = ant2_enu - ant1_enu
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


class Observatory(object):
    """
    Baseline, time, frequency, location (lat/lon), beam
    Assumes the shell lat/lon are ra/dec.
        Init time and freq structures.
        From times, get pointing centers

    """

    def __init__(self, latitude, longitude, height, array=None, freqs=None, pix_area_sr=None):
        """
        array = list of baseline objects
        """
        self.lat = latitude
        self.lon = longitude
        self.height = height
        self.array = array
        self.antennas = set()	# Accumulate the antenna numbers
        for baseline in array:
            self.antennas.add(baseline.ant1)
            self.antennas.add(baseline.ant2)
        self.freqs = freqs

        self.beam = None        # Primary beam. Set by `set_beam`
        self.times_jd = None     # Observation times. Set by `set_pointings` function
        self.fov = None         # Diameter of sky region, centered on pointing_centers, to select from the shell.
        self.pointing_centers = None    # List of [ra, dec] positions. One for each time. `set_pointings` sets this to zenith.
        self.north_poles = None     # [ra,dec] ICRS position of the Earth's north pole. Set by `set_pointings`.
        self.telescope_location = EarthLocation.from_geodetic(self.lon * units.degree, self.lat * units.degree, 
                                    self.height)

        self.do_horizon_taper = False
        self.pix_area_sr = pix_area_sr  # If doing horizon taper, need to set pixel area

        if freqs is not None:
            self.Nfreqs = len(freqs)

    def set_pointings(self, time_arr):
        """
        Set the pointing centers (in ra/dec) based on array location and times.
            Dec = self.lat
        Also sets the north pole positions in ICRS.
        RA  = What RA is at zenith at a given JD?
        """
        self.times_jd = time_arr
        centers = []
        north_poles = []
        for t in Time(time_arr, scale='utc', format='jd'):
            zen = AltAz(alt=Angle('90d'), az=Angle('0d'), obstime=t, location=self.telescope_location)
            north = AltAz(alt=Angle('0d'), az=Angle('0d'), obstime=t, location=self.telescope_location)
            zen_radec = zen.transform_to(ICRS)
            north_radec = north.transform_to(ICRS)
            centers.append([zen_radec.ra.deg, zen_radec.dec.deg])
            north_poles.append([north_radec.ra.deg, north_radec.dec.deg])
        self.pointing_centers = centers
        self.north_poles = north_poles

    def calc_azza(self, Nside, center, north=None, return_inds=False):
        """

        Calculate azimuth/altitude of sources given the pointing center.

        Parameters:
            Center = lon/lat in degrees
            radius = selection radius in degrees
            return_inds = Return the healpix indices too
            north = The direction of North in the ICRS frame (ra,dec)
                    Defines the origin of azimuth.
                    By default, assumes North is at ra/dec of 0, 90.

                    NB -- This is a bad assumption, in general, and will affect the
                    azimuth angles returned. Providing the north position fixes
                    this.

        Returns:
            zenith angles (radians)
            azimuth angles (radians)
            indices (if return_inds)
        """
        if self.fov is None:
            raise AttributeError("Need to set a field of view in degrees")
        radius = self.fov * np.pi / 180. * 1 / 2.
        if self.do_horizon_taper:
            radius += np.sqrt(self.pix_area_sr)     # Allow parts of pixels to be above the horizon.
        cvec = hp.ang2vec(center[0], center[1], lonlat=True)

        if north is None:
            north = np.array([0, 90.])
        nvec = hp.ang2vec(north[0], north[1], lonlat=True)
        pix = hp.query_disc(Nside, cvec, radius)
        vecs = hp.pix2vec(Nside, pix)
        vecs = np.array(vecs).T  # Shape (Npix, 3)

        colat = np.arccos(np.dot(cvec, nvec))  # Should be close to 90d
        xvec = np.cross(nvec, cvec) * 1 / np.sin(colat)
        yvec = np.cross(cvec, xvec)
        sdotx = np.tensordot(vecs, xvec, 1)
        sdotz = np.tensordot(vecs, cvec, 1)
        sdoty = np.tensordot(vecs, yvec, 1)
        za_arr = np.arccos(sdotz)
        az_arr = (np.arctan2(sdotx, sdoty)) % (2 * np.pi)  # xy plane is tangent. Increasing azimuthal angle eastward, zero at North (y axis). x is East.
        if return_inds:
            return za_arr, az_arr, pix
        return za_arr, az_arr

    def set_fov(self, fov):
        """
        fov = field of view in degrees
        """
        self.fov = fov

    def set_beam(self, beam='uniform', freq_interp_kind='linear', **kwargs):
        """
        Set the beam of the array.

        Args:
            beam : str, class or list of classes
                If beam is a string: if it is a viable input to AnalyticBeam, 
	        then instantiates an AnalyticBeam, otherwise assumes beam is 
                a filepath to a beamfits and instantiates a PowerBeam.
                If beam is a class or list of classes: assume these are beam
                objects with a "beam_val" method, and save them.
            freq_interp_kind : str
                For PowerBeam, frequency interpolation option.

            kwargs : keyword arguments
                kwargs to pass to AnalyticBeam instantiation.
        """
        if isinstance(beam, str):
            if beam in ['uniform', 'gaussian', 'airy'] or callable(beam):
                self.beam = AnalyticBeam(beam, **kwargs)
            else:
                self.beam = PowerBeam(beam)
                self.beam.interp_freq(self.freqs, inplace=True, kind=freq_interp_kind)
                self.beam.freq_interp_kind = freq_interp_kind
        else: self.beam = beam

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
        if isinstance(beam, list):
            raise RuntimeError("beam_sq_int not implemented for multiple beams")
        za, az = self.calc_azza(Nside, pointing)
        beam_sq_int = np.sum(self.beam.beam_val(az, za, freqs, pol=beam_pol)**2, axis=0)
        om = 4 * np.pi / (12.0 * Nside**2)
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

        pixels = []
        for cent in self.pointing_centers:
            cent = hp.ang2vec(cent[0], cent[1], lonlat=True)
            hpx_inds = hp.query_disc(Nside, cent, 2 * np.sqrt(2) * np.radians(self.fov))
            pixels.append(hpx_inds)

        return pixels

    def _horizon_taper(self, za_arr):
        """
        For pixels near the edge of the FoV downweight flux
        by what fraction of the pixel is below the horizon.

        (Allow pixels to "set")
        """
        res = np.sqrt(self.pix_area_sr)
        max_za = np.radians(self.fov) / 2.
        fracs = 0.5 * (1 - (za_arr - max_za) / res)
        fracs[fracs > 1] = 1.0    # Do not weight pixels fully above the horizon.

        return fracs

    def _vis_calc(self, pcents, tinds, shell, vis_array, Nfin, beam_pol='pI'):
        """
        Function sent to subprocesses. Called by make_visibilities.

        pcents : Pointing centers to evaluate.
        tinds : Array of indices in the time array (and correspondingly in pointings/north_poles)
        shell : SkyModel data array
        vis_array : Output array for placing results.
        Nfin : Number of finished tasks. A variable shared among subprocesses.
        """
        if len(pcents) == 0:
            return
        
        # Check for North Pole attribute.
        haspoles = True
        if self.north_poles is None:
            warnings.warn('North pole positions not set. Azimuths may be inaccurate.')
            haspoles = False

        for count, c in enumerate(pcents):
            memory_usage_GB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            if haspoles:
                north = self.north_poles[tinds[count]]
            else:
                north = None
            za_arr, az_arr, pix = self.calc_azza(self.Nside, c, north, return_inds=True)
 
            if isinstance(self.beam, list):
                # Adds another dimension to beam_cube: the baselines
                beam_cube = [ None for i in range(len(self.array)) ]
                beam_val = [ None for i in range(len(self.antennas)) ]
                for i in self.antennas: 
                    bv = self.beam[i].beam_val(az_arr, za_arr, self.freqs, pol=beam_pol)
                    bv[np.argwhere(za_arr>np.pi/2)[:, 0], :] = 0    # Sources below horizon
                    beam_val[i] = bv
                for bi, bl in enumerate(self.array):
                    # Multiply beam correction for the two antennas in each baseline
                    beam_cube[bi] = beam_val[bl.ant1]*beam_val[bl.ant2]
            else:
                beam_cube = self.beam.beam_val(az_arr, za_arr, self.freqs, pol=beam_pol) # (Npix, Nfreq)
                if not isinstance(self.beam, PowerBeam): beam_cube *= beam_cube
                beam_cube[np.argwhere(za_arr>np.pi/2)[:, 0], :] = 0    # Sources below horizon
            if self.do_horizon_taper:
                horizon_taper = self._horizon_taper(za_arr).reshape(1, za_arr.size, 1)
            else:
                horizon_taper = 1.0
            for bi, bl in enumerate(self.array):
                fringe_cube = bl.get_fringe(az_arr, za_arr, self.freqs)
                vis = np.sum(shell[..., pix, :] * horizon_taper * 
                             (beam_cube[bi] if isinstance(beam_cube, list) else beam_cube) *
                             fringe_cube, axis=-2)
                vis_array.put((tinds[count], bi, vis.tolist()))
            with Nfin.get_lock():
                Nfin.value += 1
            if mp.current_process().name == '0':
                if Nfin.value > 0:
                    dt = (time.time() - self.time0)
                    sys.stdout.write('Finished: {:d}, Elapsed {:.2f}min, Remain {:.3f}hour, MaxRSS {}GB\n'.format(
                        Nfin.value, dt / 60., (1 / 3600.) * (dt / float(Nfin.value)) * (self.Ntimes - Nfin.value), memory_usage_GB))
                    sys.stdout.flush()

    def make_visibilities(self, shell, Nprocs=1, times_jd=None, beam_pol='pI'):
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
        self.pix_area_sr = pix_area_sr

        assert Nfreqs == self.Nfreqs

        self.time0 = time.time()
        Nbls = len(self.array)
        self.Nside = Nside
        self.freqs = np.array(self.freqs)
        conv_fact = jy2Tsr(np.array(self.freqs), bm=pix_area_sr)

        if self.pointing_centers is None and times_jd is None:
            raise ValueError("Observatory.pointing_centers must be set using set_pointings() before simulation can begin.")

        if times_jd is not None:
            if self.pointing_centers is not None:
                warnings.warn("Overwriting existing pointing centers")
            self.set_pointings(times_jd)

        self.Ntimes = len(self.pointing_centers)
        pcenter_list = np.array_split(self.pointing_centers, Nprocs)
        time_inds = np.array_split(range(self.Ntimes), Nprocs)
        procs = []
        man = mp.Manager()
        vis_array = man.Queue()
        Nfin = mp.Value('i', 0)

        if Nprocs > 1 and not isinstance(shell.data, mparray):
            warnings.warn("Caution: SkyModel data array is not in shared memory. With Nprocs > 1, this will cause duplication.")

        for pi in range(Nprocs):
            p = mp.Process(name=str(pi), target=self._vis_calc, args=(pcenter_list[pi], time_inds[pi], shell.data, vis_array, Nfin), kwargs=dict(beam_pol=beam_pol))
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

        srt = np.lexsort((baseline_inds, time_inds))
        time_inds = np.array(time_inds)[srt]
        visibilities = np.array(visibilities)[srt]      # Shape (Nblts, Nskies, Nfreqs)
        if self.times_jd is not None:
            time_array = self.times_jd[time_inds]
        else:
            time_array = None
        baseline_array = np.array(baseline_inds)[srt]

        # Time and baseline arrays are now Nblts
        return visibilities / conv_fact, time_array, baseline_array
