# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

import numpy as np
import multiprocessing as mp
import sys
import resource
import warnings
import time
import copy
from astropy_healpix import healpy as hp
from astropy_healpix import HEALPix
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


class Baseline(object):

    def __init__(self, ant1_enu=None, ant2_enu=None, ant1=None, ant2=None, enu_vec=None):
        if enu_vec is not None:
            self.enu = enu_vec
        else:
            ant1_enu = np.asarray(ant1_enu)
            ant2_enu = np.asarray(ant2_enu)
            self.enu = ant2_enu - ant1_enu

        assert self.enu.size == 3, f"Wronge enu vector shape {self.enu.shape}"

        # Antenna indexes, for indexing beam list if necessary.
        # Must be numbers from 0 ..., so the right beams are found.
        self.ant1 = ant1        
        self.ant2 = ant2       


    def get_uvw(self, freq_Hz):
        return np.outer(self.enu, 1 / (c_ms / freq_Hz))  # In wavelengths

    def get_fringe(self, az, za, freq_Hz, degrees=False):
        if degrees:
            az *= np.pi / 180
            za *= np.pi / 180
        freq_Hz = freq_Hz.astype(float)

        pos_l = np.sin(az) * np.sin(za)
        pos_m = np.cos(az) * np.sin(za)
        pos_n = np.cos(za)
        lmn = np.vstack((pos_l, pos_m, pos_n))
        self.uvw = self.get_uvw(freq_Hz)
        udotl = np.einsum("jk,jl->kl", lmn, self.uvw)
        fringe = np.cos(2 * np.pi * udotl) + (1j) * np.sin(
            2 * np.pi * udotl
        )  # This is weirdly faster than np.exp

        return fringe


class Observatory(object):
    """
    Representation of the observing instrument.

    Parameters
    ----------
    latitude, longitude: float
        Decimal degrees position of the observatory on Earth.
    height: float
        Decimal meters height of the observatory on Earth.
    fov: float
        Field of view in degrees (Defaults to 180 deg for horizon to horizon).
    baseline_array: array_like of Baseline instances
        The set of baselines in the observatory.
    freqs: array of float
        Array of frequencies, in Hz
    nside: int
        Nside parameter for the input map (optional).
    array: array_like of Baseline instances
        Alias for baseline_array, for backwards compatibility.
    """

    def __init__(
        self,
        latitude,
        longitude,
        height=0.0,
        fov=None,
        baseline_array=None,
        freqs=None,
        nside=None,
        array=None,
    ):
        if baseline_array is None and array is not None:
            baseline_array = array
        self.array = baseline_array
        self.freqs = freqs

        if fov is None:
            fov = 180  # Degrees

        self.fov = fov

        if nside is None:
            self.healpix = None
        else:
            self.healpix = HEALPix(nside=nside)
            self._set_vectors()

        self.beam = None  # Primary beam. Set by `set_beam`
        self.times_jd = None  # Observation times. Set by `set_pointings` function
        self.pointing_centers = None  # List of [ra, dec] positions. One for each time. `set_pointings` sets this to zenith.
        self.north_poles = None  # [ra,dec] ICRS position of the Earth's north pole. Set by `set_pointings`.
        self.telescope_location = EarthLocation.from_geodetic(
            longitude * units.degree, latitude * units.degree, height 
        )

        self.do_horizon_taper = False

        if freqs is not None:
            self.Nfreqs = len(freqs)

    def _set_vectors(self):
        """
        Set the unit vectors to pixel centers for the whole shell, in a shared memory array.

        Sets the attribute _vecs.
        """
        vecs = hp.pix2vec(self.healpix.nside, np.arange(self.healpix.npix))
        vecs = np.array(vecs).T  # Shape (Npix, 3)
        self._vecs = mparray(vecs.shape, dtype=float)
        self._vecs[()] = vecs[()]

    def set_pointings(self, time_arr):
        """
        Set the pointing centers (in ra/dec) based on array location and times.
            Dec = self.lat
            RA  = What RA is at zenith at a given JD?
        Also sets the north pole positions in ICRS.
        """
        self.times_jd = time_arr
        centers = []
        north_poles = []
        for t in Time(time_arr, scale="utc", format="jd"):
            zen = AltAz(
                alt=Angle("90d"),
                az=Angle("0d"),
                obstime=t,
                location=self.telescope_location,
            )
            north = AltAz(
                alt=Angle("0d"),
                az=Angle("0d"),
                obstime=t,
                location=self.telescope_location,
            )

            zen_radec = zen.transform_to(ICRS())
            north_radec = north.transform_to(ICRS())
            centers.append([zen_radec.ra.deg, zen_radec.dec.deg])
            north_poles.append([north_radec.ra.deg, north_radec.dec.deg])
        self.pointing_centers = centers
        self.north_poles = north_poles

    def calc_azza(self, center, north=None, return_inds=False):
        """
        Calculate azimuth/altitude of sources given the pointing center.

        Parameters
        ----------
        center: array_like of float
            [lon, lat] of pointing center in degrees
        radius: float
            Selection radius in degrees
        north: array_like of float
            [ra, dec] in degrees of the ICRS North pole.
            This is used to define the origin of azimuth.
            Defaults to [0, 90].
            NB -- This is a bad assumption, in general, and will affect the
            azimuth angles returned. Providing the north position fixes this.
        return_inds: bool
            Return the healpix indices (Default False)

        Returns
        -------
        zenith_angles: array of float
            zenith angles in radians.
        azimuth_angles: array of float
            azimuth angles in radians same shape as zenith_angles)
        indices: array of int
            healpix indices of chosen pixels
            (If return_inds is True)
        """
        if self.fov is None:
            raise AttributeError("Need to set a field of view in degrees")
        if self.healpix is None:
            raise AttributeError("Need to set HEALPix instance attribute")

        radius = self.fov * np.pi / 180.0 * 1 / 2.0
        if self.do_horizon_taper:
            radius += self.healpix.pixel_resolution.to_value(
                "rad"
            )  # Allow parts of pixels to be above the horizon.

        cvec = hp.ang2vec(center[0], center[1], lonlat=True)

        if north is None:
            north = np.array([0, 90.0])
        nvec = hp.ang2vec(north[0], north[1], lonlat=True)
        colat = np.arccos(np.dot(cvec, nvec))  # Should be close to 90d
        xvec = np.cross(nvec, cvec) * 1 / np.sin(colat)
        yvec = np.cross(cvec, xvec)
        sdotx = np.tensordot(self._vecs, xvec, 1)
        sdotz = np.tensordot(self._vecs, cvec, 1)
        sdoty = np.tensordot(self._vecs, yvec, 1)
        za_arr = np.arccos(sdotz)
        az_arr = (np.arctan2(sdotx, sdoty)) % (
            2 * np.pi
        )  # xy plane is tangent. Increasing azimuthal angle eastward, zero at North (y axis). x is East.
        pix = za_arr <= radius  # Horizon cut.
        if return_inds:
            return za_arr[pix], az_arr[pix], np.arange(self.healpix.npix)[pix]
        return za_arr[pix], az_arr[pix]

    def set_fov(self, fov):
        """
        fov = field of view in degrees
        """
        self.fov = fov

    def set_beam(self, beam="uniform", freq_interp_kind="linear", **kwargs):
        """
        Set the beam of the array.

        Args:
            beam : str, or list of beam objects
                str: If it is a viable input to AnalyticBeam, 
	            then instantiates an AnalyticBeam, otherwise assumes beam is 
                    a filepath to a beamfits and instantiates a PowerBeam.
                list: List of beam objects. This allows for external beams to be 
                    used, and different beams for each antenna. They should not be 
                    power beams. Each beam mustshould have an interp method: 
                    interp(self, az_array, za_array, freq_array)
            freq_interp_kind : str
                For PowerBeam, frequency interpolation option.

            kwargs : keyword arguments
                kwargs to pass to AnalyticBeam instantiation.
        """

        if isinstance(beam, list): self.beam = beam
        elif beam in ['uniform', 'gaussian', 'airy'] or callable(beam):
                self.beam = AnalyticBeam(beam, **kwargs)
        else:
            self.beam = PowerBeam(beam)
            self.beam.interp_freq(self.freqs, inplace=True, kind=freq_interp_kind)
            self.beam.freq_interp_kind = freq_interp_kind

    def beam_sq_int(self, freqs, Nside, pointing, beam_pol="pI"):
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

        if isinstance(self.beam, list):
            raise RuntimeError("beam_sq_int not implemented for multiple antenna beams")
        za, az = self.calc_azza(pointing)
        beam_sq_int = np.sum(
            self.beam.beam_val(az, za, freqs, pol=beam_pol) ** 2, axis=0
        )
        om = 4 * np.pi / (12.0 * Nside ** 2)
        beam_sq_int = beam_sq_int * om

        return beam_sq_int

    def external_beam_val(self, beam, az_arr, za_arr, freqs, pol="XX"):
        """
        Call interp() on a beam, and provide results in the right format.
        """
        interp_data, interp_basis_vector = beam.interp(az_arr, za_arr, freqs)
        return interp_data[0, 0, 1].T   # just want Npix, Nfreq

    def _horizon_taper(self, za_arr):
        """
        For pixels near the edge of the FoV downweight flux
        by what fraction of the pixel is below the horizon.

        (Allow pixels to "set")
        """
        res = self.healpix.pixel_resolution.to_value("rad")
        max_za = np.radians(self.fov) / 2.0
        fracs = 0.5 * (1 - (za_arr - max_za) / res)
        fracs[fracs > 1] = 1.0  # Do not weight pixels fully above the horizon.

        return fracs

    def _vis_calc(self, pcents, tinds, shell, vis_array, Nfin, beam_pol="pI"):
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
            warnings.warn("North pole positions not set. Azimuths may be inaccurate.")
            haspoles = False

        for count, c_ in enumerate(pcents):
            memory_usage_GB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            north = self.north_poles[tinds[count]] if haspoles else None
            za_arr, az_arr, pix = self.calc_azza(c_, north, return_inds=True)
            if isinstance(self.beam, list):
                # Adds another dimension to beam_cube: the baselines.
                # Beams may be different for each antenna, and they are
                # not power beams.
                # Multiplies the beams for the 2 antennas in a baseline.

                # Accumulate the antenna numbers
                antennas = set()
                for bi, baseline in enumerate(self.array):
                    assert baseline.ant1 is not None and baseline.ant2 is not None,  \
                            "Antenna number not set for baseline "+str(bi)
                    antennas.add(baseline.ant1)
                    antennas.add(baseline.ant2)
                assert len(antennas) == len(self.beam), "Number of beams does not match number of antennas"

                beam_cube = [ None for i in range(len(self.array)) ]
                beam_val = [ None for i in range(len(antennas)) ]
                for i in antennas:
                    bv = self.external_beam_val(self.beam[i], az_arr, za_arr, self.freqs, pol=beam_pol)
                    bv[np.argwhere(za_arr>np.pi/2)[:, 0], :] = 0    # Sources below horizon
                    beam_val[i] = bv
                for bi, bl in enumerate(self.array):
                    # Multiply beam correction for the two antennas in each baseline
                    beam_cube[bi] = beam_val[bl.ant1]*beam_val[bl.ant2]
            else:
                beam_cube = self.beam.beam_val(az_arr, za_arr, self.freqs, pol=beam_pol)
                beam_cube[np.argwhere(za_arr>np.pi/2)[:, 0], :] = 0    # Sources below horizon

            if self.do_horizon_taper:
                horizon_taper = self._horizon_taper(za_arr).reshape(1, za_arr.size, 1)
            else:
                horizon_taper = 1.0
            if isinstance(self.beam, list):
                # Beams are possibly different for each baseline
                sky = shell[..., pix, :] * horizon_taper 
                for bi, bl in enumerate(self.array):
                    fringe_cube = bl.get_fringe(az_arr, za_arr, self.freqs)
                    vis = np.sum(sky * fringe_cube * beam_cube[bi], axis=-2)
                    vis_array.put((tinds[count], bi, vis.tolist()))
            else:
                sky = shell[..., pix, :] * horizon_taper * beam_cube
                for bi, bl in enumerate(self.array):
                    fringe_cube = bl.get_fringe(az_arr, za_arr, self.freqs)
                    vis = np.sum(sky * fringe_cube, axis=-2)
                    vis_array.put((tinds[count], bi, vis.tolist()))
            with Nfin.get_lock():
                Nfin.value += 1
            if mp.current_process().name == "0" and Nfin.value > 0:
                dt = time.time() - self.time0
                sys.stdout.write(
                    "Finished: {:d}, Elapsed {:.2f}min, Remain {:.3f}hour, MaxRSS {}GB\n".format(
                        Nfin.value,
                        dt / 60.0,
                        (1 / 3600.0)
                        * (dt / float(Nfin.value))
                        * (self.Ntimes - Nfin.value),
                        memory_usage_GB,
                    )
                )
                sys.stdout.flush()

    def make_visibilities(self, shell, Nprocs=1, times_jd=None, beam_pol="pI"):
        """
        Make beam cube and fringe cube, multiply and sum.
        shell (Npix, Nfreq) = healpix shell, as an mparray (multiprocessing shared array)

        Takes a shell in Kelvin
        Returns visibility in Jy
        """

        self.healpix = HEALPix(nside=shell.Nside)
        self._set_vectors()
        Nfreqs = shell.Nfreqs

        assert Nfreqs == self.Nfreqs

        self.time0 = time.time()
        self.freqs = np.asarray(self.freqs)
        conv_fact = jy2Tsr(self.freqs, bm=self.healpix.pixel_area.to_value("sr"))

        if self.pointing_centers is None and times_jd is None:
            raise ValueError(
                "Observatory.pointing_centers must be set using set_pointings() before simulation can begin."
            )

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
        Nfin = mp.Value("i", 0)

        if Nprocs > 1 and not isinstance(shell.data, mparray):
            warnings.warn(
                "Caution: SkyModel data array is not in shared memory. With Nprocs > 1, "
                "this will cause duplication."
            )

        for pi in range(Nprocs):
            p = mp.Process(
                name=str(pi),
                target=self._vis_calc,
                args=(pcenter_list[pi], time_inds[pi], shell.data, vis_array, Nfin),
                kwargs={"beam_pol": beam_pol},
            )
            p.start()
            procs.append(p)
        while (Nfin.value < self.Ntimes) and np.any([p.is_alive() for p in procs]):
            continue
        visibilities = []
        time_inds, baseline_inds = [], []
        for (ti, bi, varr) in iter(vis_array.get, None):
            visibilities.append(varr)
            time_inds += [ti]
            baseline_inds += [bi]
            if vis_array.empty():
                break

        srt = np.lexsort((baseline_inds, time_inds))
        time_inds = np.array(time_inds)[srt]
        visibilities = np.array(visibilities)[srt]  # Shape (Nblts, Nskies, Nfreqs)
        time_array = self.times_jd[time_inds] if self.times_jd is not None else None
        baseline_array = np.array(baseline_inds)[srt]

        # Time and baseline arrays are now Nblts
        return visibilities / conv_fact, time_array, baseline_array
