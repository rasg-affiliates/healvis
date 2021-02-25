# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

import numpy as np
import yaml
import sys
import os
import ast
import copy
import warnings

from pyuvdata import UVData, UVBeam
from pyuvdata import utils as uvutils

from . import observatory, version, beam_model, sky_model, utils

# -----------------------
# Methods to parse configuration files and setup/run simulation.
# -----------------------


def _parse_layout_csv(layout_csv):
    """ Interpret the layout csv file """

    with open(layout_csv, "r") as fhandle:
        header = fhandle.readline()

    header = [h.strip() for h in header.split()]
    dt = np.format_parser(
        ["U10", "i4", "i4", "f8", "f8", "f8"],
        ["name", "number", "beamid", "e", "n", "u"],
        header,
    )

    return np.genfromtxt(layout_csv, autostrip=True, skip_header=1, dtype=dt.dtype)


def parse_telescope_params(tele_params):
    """
    Parse the "telescope" section of a healvis obsparam.

    Args:
        tele_params: Dictionary of telescope parameters

    Returns:
        dict of array properties:
            |  Nants_data: Number of antennas
            |  Nants_telescope: Number of antennas
            |  antenna_names: list of antenna names
            |  antenna_numbers: corresponding list of antenna numbers
            |  antenna_positions: Array of ECEF antenna positions
            |  telescope_location: ECEF array center location
            |  telescope_config_file: Path to configuration yaml file
            |  antenna_location_file: Path to csv layout file
            |  telescope_name: observatory name
    """
    layout_csv = tele_params["array_layout"]
    if not os.path.exists(layout_csv):
        if not os.path.exists(layout_csv):
            raise ValueError(
                "layout_csv file from yaml does not exist: {}".format(layout_csv)
            )

    ant_layout = _parse_layout_csv(layout_csv)
    if isinstance(tele_params["telescope_location"], str):
        tloc = tele_params["telescope_location"][1:-1]  # drop parens
        tloc = list(map(float, tloc.split(",")))
    else:
        tloc = list(tele_params["telescope_location"])
    tloc[0] *= np.pi / 180.0
    tloc[1] *= np.pi / 180.0  # Convert to radians
    tloc_xyz = uvutils.XYZ_from_LatLonAlt(*tloc)

    E, N, U = ant_layout["e"], ant_layout["n"], ant_layout["u"]
    antnames = ant_layout["name"]
    return_dict = {}

    return_dict["Nants_data"] = antnames.size
    return_dict["Nants_telescope"] = antnames.size
    return_dict["antenna_names"] = np.array(antnames.tolist())
    return_dict["antenna_numbers"] = np.array(ant_layout["number"])
    antpos_enu = np.vstack((E, N, U)).T
    return_dict["antenna_positions"] = (
        uvutils.ECEF_from_ENU(antpos_enu, *tloc) - tloc_xyz
    )
    return_dict["array_layout"] = layout_csv
    return_dict["telescope_location"] = tloc_xyz
    return_dict["telescope_name"] = tele_params["telescope_name"]

    return return_dict


def parse_frequency_params(freq_params):
    """
    Parse the "freq" section of obsparam.

    Frequency arrays are defined to be reflective of channel center.

    If the 'freq_array' key is present in freq_params, it supercedes all
    other keys. Otherwise, freq_params must contain one of the following
          i) start_freq, Nfreqs & channel_width
         ii) start_freq, Nfreqs & bandwidth
        iii) start_freq, Nfreqs & end_freq
         iv) start_freq, channel_width & end_freq

    This function will look for the key combinations in that order, and take
    the first combination with all keys present.

    Args:
        freq_params: Dictionary of frequency parameters

    Returns:
        dict of array properties:
            |  channel_width: (float) Frequency channel spacing in Hz
            |  Nfreqs: (int) Number of frequency channels
            |  freq_array: (dtype float, ndarray, shape=(Nspws, Nfreqs)) Frequency channel centers in Hz
            |  bandwidth : (float) Full "observers" bandwidth of the data: i.e. channel_width * Nfreqs
    """
    freq_keywords = [
        "freq_array",
        "start_freq",
        "end_freq",
        "Nfreqs",
        "channel_width",
        "bandwidth",
    ]
    fa, sf, ef, nf, cw, bw = [fk in freq_params for fk in freq_keywords]

    # look for freq_array
    if fa:
        freq_arr = np.array(freq_params["freq_array"])
        Nfreqs = freq_arr.size
        if Nfreqs > 1:
            channel_width = np.diff(freq_arr)[0]
        elif "channel_width" not in freq_params:
            raise ValueError(
                "Channel width must be specified " "if passed freq_arr has length 1"
            )
        bandwidth = channel_width * Nfreqs

    # look for other key combinations in the order described above
    else:
        if nf and cw and sf:
            Nfreqs = freq_params["Nfreqs"]
            channel_width = freq_params["channel_width"]
            start_freq = freq_params["start_freq"]
            bandwidth = Nfreqs * channel_width

        elif nf and bw and sf:
            Nfreqs = freq_params["Nfreqs"]
            bandwidth = freq_params["bandwidth"]
            start_freq = freq_params["start_freq"]
            channel_width = float(bandwidth / Nfreqs)

        elif nf and sf and ef:
            start_freq = freq_params["start_freq"]
            end_freq = freq_params["end_freq"]
            Nfreqs = freq_params["Nfreqs"]
            channel_width = float(end_freq - start_freq) / np.clip(
                Nfreqs - 1, 1, np.inf
            )
            bandwidth = Nfreqs * channel_width

        elif cw and sf and ef:
            start_freq = freq_params["start_freq"]
            end_freq = freq_params["end_freq"]
            channel_width = freq_params["channel_width"]
            Nfreqs = float(end_freq - start_freq) / channel_width + 1
            if not np.isclose(Nfreqs % 1, 0.0):
                raise ValueError(
                    "end_freq - start_freq must be evenly divisible by channel_width"
                )
            Nfreqs = int(Nfreqs)
            bandwidth = channel_width * Nfreqs

        else:
            raise KeyError(
                "Couldn't find any proper combination of keys in freq_params"
            )

        # Only supports Nspw = 1 because pyuvdata only supports this
        freq_arr = np.linspace(
            start_freq, start_freq + bandwidth, Nfreqs, endpoint=False
        ).reshape(1, -1)

        # Check that the freq_array is consistent with channel_width
        assert np.allclose(np.diff(freq_arr), np.ones(Nfreqs - 1) * channel_width)

    if "freq_chans" in freq_params:
        chans = ast.literal_eval(freq_params["freq_chans"])
        freq_arr = freq_arr[:, slice(*chans)]
        Nfreqs = freq_arr.size

    return_dict = {}
    return_dict["Nfreqs"] = Nfreqs
    return_dict["freq_array"] = freq_arr
    return_dict["channel_width"] = channel_width
    return_dict["bandwidth"] = bandwidth

    return return_dict


def parse_time_params(time_params):
    """
    Parse the "time" section of obsparam.

    Time arrays are defined to be reflective of bin center.

    If 'time_array' key is present in time_params, it supercedes all other keys.
    Otherwise, the following key combinations will be searched for in this order:
          i) start_time, Ntimes & time_cadence
         ii) start_time, Ntimes & duration
        iii) start_time, Ntimes & end_time
         iv) start_time, end_time & time_cadence

    This function will look for the key combinations in that order, and take
    the first combination with all keys present.

    Args:
        time_params: Dictionary of time parameters

    Returns:
        dict of array properties:
            |  time_cadence: (float) Time step size on seconds
            |  Ntimes: (int) Number of time steps
            |  time_array: (dtype float, ndarray, shape=(Ntimes)) Time step centers in Julian Date.
            |  duration: (float) Time duration in Julian Date.
    """
    time_keywords = [
        "start_time",
        "end_time",
        "Ntimes",
        "time_cadence",
        "duration",
        "duration_hours",
        "duration_days",
        "time_array",
    ]
    st, et, nt, tc, du, dh, dd, ta = [tk in time_params for tk in time_keywords]
    daysperhour = 1 / 24.0
    hourspersec = 1 / 60.0 ** 2
    dayspersec = daysperhour * hourspersec

    # parse possible keys related to duration
    if du:
        duration = time_params["duration"]
    else:
        if dh:
            assert not dd, "Cannot feed duration_hours and duration_days"
            duration = time_params["duration_hours"] * daysperhour
            du = True
        elif dd:
            assert not du, "Cannot feed duration_hours and duration_days"
            duration = time_params["duration_days"]
            du = True

    # look for time_array
    if ta:
        time_array = time_params["time_array"]
        Ntimes = time_array.size
        if Ntimes > 1:
            time_cadence = np.diff(time_array)[0] / dayspersec
        elif "time_cadence" not in time_params:
            raise ValueError("time_cadence must be specified " "if Ntimes == 1.")
        duration = time_cadence * Ntimes

    # look for other key combinations in the order described above
    else:
        if st and nt and tc:
            start_time = time_params["start_time"]
            Ntimes = time_params["Ntimes"]
            time_cadence = time_params["time_cadence"]
            duration = time_cadence * Ntimes * dayspersec

        elif st and nt and du:
            start_time = time_params["start_time"]
            Ntimes = time_params["Ntimes"]
            time_cadence = float(duration) / Ntimes / dayspersec

        elif st and nt and et:
            start_time = time_params["start_time"]
            end_time = time_params["end_time"]
            Ntimes = time_params["Ntimes"]
            time_cadence = (
                float(end_time - start_time)
                / np.clip(Ntimes - 1, 1, np.inf)
                / dayspersec
            )
            duration = time_cadence * Ntimes * dayspersec

        elif st and et and tc:
            start_time = time_params["start_time"]
            end_time = time_params["end_time"]
            time_cadence = time_params["time_cadence"]
            Ntimes = float(end_time - start_time) / (time_cadence * dayspersec) + 1
            if not np.isclose(Ntimes % 1, 0.0, atol=1e-4):
                raise ValueError(
                    "end_time - start_time must be evenly divisible by time_cadence"
                )
            Ntimes = int(Ntimes)
            duration = time_cadence * Ntimes * dayspersec

        else:
            raise KeyError(
                "Couldn't find any proper combination of keys in time_params."
            )

        time_array = np.linspace(
            start_time, start_time + duration, Ntimes, endpoint=False
        )

        # Check that the time_array is consistent with time_cadence
        assert np.allclose(
            np.diff(time_array), np.ones(Ntimes - 1) * time_cadence * dayspersec
        )

    return_dict = {}
    return_dict["time_array"] = time_array
    return_dict["duration"] = duration
    return_dict["time_cadence"] = time_cadence
    return_dict["Ntimes"] = Ntimes

    return return_dict


def complete_uvdata(uv_obj, run_check=True):
    """
    Given a UVData object lacking Nblts-length arrays, fill out the rest.

    Args:
        uv_obj : UVData object to finish.
        run_check: Run the standard UVData checks.
    """
    bl_array = uv_obj.baseline_array  # (Nbls,)
    time_array = uv_obj.time_array  # (Ntimes,)

    uv_obj.baseline_array = np.tile(bl_array, uv_obj.Ntimes)
    uv_obj.ant_1_array, uv_obj.ant_2_array = uv_obj.baseline_to_antnums(
        uv_obj.baseline_array
    )
    uv_obj.Nbls = np.unique(uv_obj.baseline_array).size
    uv_obj.Nblts = uv_obj.Nbls * uv_obj.Ntimes
    uv_obj.time_array = np.repeat(time_array, uv_obj.Nbls)
    uv_obj.integration_time = np.zeros(uv_obj.Nblts, dtype=float)
    uv_obj.Nants_data = np.unique(
        uv_obj.ant_1_array.tolist() + uv_obj.ant_2_array.tolist()
    ).size
    uv_obj.set_lsts_from_time_array()

    # fill in data
    uv_obj.data_array = np.zeros(
        (uv_obj.Nblts, uv_obj.Nspws, uv_obj.Nfreqs, uv_obj.Npols), dtype=np.complex128
    )
    uv_obj.flag_array = np.zeros(
        (uv_obj.Nblts, uv_obj.Nspws, uv_obj.Nfreqs, uv_obj.Npols), dtype=bool
    )
    uv_obj.nsample_array = np.ones(
        (uv_obj.Nblts, uv_obj.Nspws, uv_obj.Nfreqs, uv_obj.Npols), dtype=np.float64
    )

    # Other attributes
    uv_obj.set_uvws_from_antenna_positions()
    if run_check:
        uv_obj.check()

    return uv_obj


def setup_uvdata(
    array_layout=None,
    telescope_location=None,
    telescope_name=None,
    Nfreqs=None,
    start_freq=None,
    bandwidth=None,
    freq_array=None,
    Ntimes=None,
    time_cadence=None,
    start_time=None,
    time_array=None,
    bls=None,
    anchor_ant=None,
    antenna_nums=None,
    no_autos=True,
    pols=["xx"],
    make_full=False,
    redundancy=None,
    run_check=True,
):
    """
    Setup a UVData object for simulating.

    Args:
        array_layout : str
            Filepath to array layout in ENU coordinates [meters]
        telescope_location : len-3 tuple
            Telescope location on Earth in LatLonAlt coordinates [deg, deg, meters]
        telescope_name : str
            Name of telescope
        Nfreqs : int
            Number of frequency channels
        start_freq : float
            Starting frequency [Hz]
        bandwidth : float
            Total frequency bandwidth of spectral window [Hz]
        freq_array : ndarray
            frequency array [Hz], cannot be specified if start_freq, Nfreqs or bandwidth is specified
        Ntimes : int
            Number of integration bins
        time_cadence : float
            Cadence of time bins [seconds]
        start_time : float
            Time of the first integration bin [Julian Date]
        time_array : ndarray
            time array [Julian Date], cannot be specified if start_time, Ntimes and time_cadence is specified
        bls : list
            List of antenna-pair tuples for baseline selection
        anchor_ant: int
            Selects baselines such that one of the pair is a specified antenna number
        redundancy: float
            Redundant baseline selection tolerance for selection
        antenna_nums : list
            List of antenna numbers to keep in array
        make_full : Generate the full UVData object, includes arrays of length Nblts.
                    Default behavior creates an invalid UVData object, where baseline_array has length Nbls, etc.
                    This is to avoid excessive memory usage up front when it's not necessary.

    Returns:
        UVData object with zeroed data_array
    """

    # get antenna information
    tele_dict = parse_telescope_params(
        dict(
            array_layout=array_layout,
            telescope_location=telescope_location,
            telescope_name=telescope_name,
        )
    )
    lat, lon, alt = uvutils.LatLonAlt_from_XYZ(tele_dict["telescope_location"])
    antpos = tele_dict["antenna_positions"]
    enu = uvutils.ENU_from_ECEF(
        tele_dict["antenna_positions"] + tele_dict["telescope_location"], lat, lon, alt
    )
    anums = tele_dict["antenna_numbers"]
    antnames = tele_dict["antenna_names"]
    Nants = tele_dict["Nants_data"]

    # setup object and fill in basic info
    uv_obj = UVData()
    uv_obj.telescope_location = tele_dict["telescope_location"]
    uv_obj.telescope_location_lat_lon_alt = (lat, lon, alt)
    uv_obj.telescope_location_lat_lon_alt_degrees = (
        np.degrees(lat),
        np.degrees(lon),
        alt,
    )
    uv_obj.antenna_numbers = anums
    uv_obj.antenna_names = antnames
    uv_obj.antenna_positions = antpos
    uv_obj.Nants_telescope = Nants

    # fill in frequency and time info: wait to fill in len-Nblts arrays until later
    if freq_array is not None:
        if Nfreqs is not None or start_freq is not None or bandwidth is not None:
            raise ValueError(
                "Cannot specify freq_array as well as Nfreqs, start_freq or bandwidth"
            )
        if freq_array.ndim == 1:
            freq_array = freq_array.reshape(1, -1)
            Nfreqs = freq_array.size
    else:
        freq_dict = parse_frequency_params(
            dict(Nfreqs=Nfreqs, start_freq=start_freq, bandwidth=bandwidth)
        )
        freq_array = freq_dict["freq_array"]

    if time_array is not None:
        if Ntimes is not None or start_time is not None or time_cadence is not None:
            raise ValueError(
                "Cannot specify time_array as well as Ntimes, start_time or time_cadence"
            )
        Ntimes = time_array.size
    else:
        time_dict = parse_time_params(
            dict(Ntimes=Ntimes, start_time=start_time, time_cadence=time_cadence)
        )
        time_array = time_dict["time_array"]

    uv_obj.freq_array = freq_array
    uv_obj.Nfreqs = Nfreqs
    uv_obj.Ntimes = Ntimes

    # fill in other attributes
    uv_obj.spw_array = np.array([0], dtype=int)
    uv_obj.Nspws = 1
    uv_obj.polarization_array = np.array(
        [uvutils.polstr2num(pol) for pol in pols], dtype=int
    )
    uv_obj.Npols = uv_obj.polarization_array.size
    if uv_obj.Nfreqs > 1:
        uv_obj.channel_width = np.diff(uv_obj.freq_array[0])[0]
    else:
        uv_obj.channel_width = 1.0
    uv_obj._set_drift()
    uv_obj.telescope_name = tele_dict["telescope_name"]
    uv_obj.instrument = "simulator"
    uv_obj.object_name = "zenith"
    uv_obj.vis_units = "Jy"
    uv_obj.history = ""

    if redundancy is not None:
        red_tol = redundancy
        reds, vec_bin_centers, lengths = uvutils.get_antenna_redundancies(
            anums, enu, tol=red_tol, include_autos=not bool(no_autos)
        )
        bls = [r[0] for r in reds]
        bls = [uvutils.baseline_to_antnums(bl_ind, Nants) for bl_ind in bls]

    # Setup array and implement antenna/baseline selections.
    bl_array = []
    _bls = [(a1, a2) for a1 in anums for a2 in anums if a1 <= a2]
    if bls is not None:
        if isinstance(bls, str):
            bls = ast.literal_eval(bls)
        bls = [bl for bl in _bls if bl in bls]
    else:
        bls = _bls
    if anchor_ant is not None:
        bls = [bl for bl in bls if anchor_ant in bl]

    if bool(no_autos):
        bls = [bl for bl in bls if bl[0] != bl[1]]
    if antenna_nums is not None:
        if isinstance(antenna_nums, str):
            antenna_nums = ast.literal_eval(antenna_nums)
        if isinstance(antenna_nums, int):
            antenna_nums = [antenna_nums]
        bls = [(a1, a2) for (a1, a2) in bls if a1 in antenna_nums or a2 in antenna_nums]
    bls = sorted(bls)
    for (a1, a2) in bls:
        bl_array.append(uvutils.antnums_to_baseline(a1, a2, 1))
    bl_array = np.asarray(bl_array)
    print("Nbls: {}".format(bl_array.size))
    if bl_array.size == 0:
        raise ValueError("No baselines selected.")
    uv_obj.time_array = time_array  # Keep length Ntimes
    uv_obj.baseline_array = bl_array  # Length Nbls

    if make_full:
        uv_obj = complete_uvdata(uv_obj, run_check=run_check)

    return uv_obj


def setup_observatory_from_uvdata(
    uv_obj,
    fov=180,
    set_pointings=True,
    beam=None,
    beam_kwargs={},
    beam_freq_interp="cubic",
    smooth_beam=False,
    smooth_scale=2.0,
    freq_chans=None,
    apply_horizon_taper=False,
    pointings=None,
):
    """
    Setup an Observatory object from a UVData object.

    Args:
        uv_obj : UVData object with metadata
        fov : float
            Field of View (diameter) in degrees
        set_pointings : bool
            If True, use time_array to set Observatory pointing centers
        beam : str or UVBeam or PowerBeam or AnalyticBeam
            Filepath to beamfits or a UVBeam object to use as primary beam model.
        beam_kwargs : dictionary
            Beam keyword arguments to pass to Observatory.set_beam if beam is a viable
            input to AnalyticBeam
        beam_freq_interp : str
            Intepolation method of beam across frequency if PowerBeam, see scipy.interp1d for details
        smooth_beam : bool
            If True, and beam is PowerBeam, smooth it across frequency with a Gaussian Process
        smooth_scale : float
            If smoothing the beam, smooth it at this frequency scale [MHz]
        freq_chans : integer 1D array
            Frequency channel indices to use from uv_obj when setting observatory freqs.
        apply_horizon_taper : bool
            When simulating, weight pixels near horizon by the fraction of the pixel area that is up.

    Returns:
        Observatory object
    """
    # setup Baselines
    antpos, ants = uv_obj.get_ENU_antpos()
    antpos_d = dict(zip(ants, antpos))
    bls = []
    for bl in np.unique(uv_obj.baseline_array):
        ap = uv_obj.baseline_to_antnums(bl)
        bls.append(observatory.Baseline(antpos_d[ap[0]], antpos_d[ap[1]]))

    lat, lon, alt = uv_obj.telescope_location_lat_lon_alt_degrees
    if freq_chans is None:
        freq_chans = slice(None)
    obs = observatory.Observatory(
        lat, lon, fov=fov, baseline_array=bls, freqs=uv_obj.freq_array[0, freq_chans]
    )

    # Horizon taper flag
    obs.do_horizon_taper = apply_horizon_taper

    if pointings is not None:
        obs.pointing_centers = pointings

    # set pointings
    if set_pointings:
        obs.set_pointings(np.unique(uv_obj.time_array))

    # set beam
    if beam is not None:
        if isinstance(beam, UVBeam):
            obs.beam = copy.deepcopy(beam)
            obs.beam.__class__ = beam_model.PowerBeam
            obs.beam.interp_freq(obs.freqs, inplace=True, kind=beam_freq_interp)

        elif isinstance(beam, str) or callable(beam):
            obs.set_beam(beam, freq_interp_kind=beam_freq_interp, **beam_kwargs)

        elif isinstance(beam, beam_model.PowerBeam):
            obs.beam = beam.interp_freq(obs.freqs, inplace=False, kind=beam_freq_interp)

        elif isinstance(beam, beam_model.AnalyticBeam):
            obs.beam = beam

    # smooth the beam
    if isinstance(obs.beam, beam_model.PowerBeam) and smooth_beam:
        obs.beam.smooth_beam(obs.freqs, inplace=True, freq_ls=smooth_scale)

    return obs


def run_simulation(param_file, Nprocs=1, sjob_id=None, add_to_history=""):
    """
    Parse input parameter file, construct UVData and SkyModel objects, and run simulation.
    """
    # parse parameter dictionary
    if isinstance(param_file, str):
        with open(param_file, "r") as yfile:
            param_dict = yaml.safe_load(yfile)
    else:
        param_dict = copy.deepcopy(param_file)

    sys.stdout.flush()
    freq_dict = parse_frequency_params(param_dict["freq"])
    freq_array = freq_dict["freq_array"][0]

    time_dict = parse_time_params(param_dict["time"])
    time_array = time_dict["time_array"]
    filing_params = param_dict["filing"]

    # ---------------------------
    # Extra parameters required for healvis
    # ---------------------------
    skyparam = param_dict["skyparam"].copy()
    skyparam["freqs"] = freq_array

    Nskies = 1 if "Nskies" not in param_dict else int(param_dict["Nskies"])

    if "Nprocs" in param_dict:
        Nprocs = param_dict["Nprocs"]
    print("Nprocs: ", Nprocs, flush=True)

    # ---------------------------
    # SkyModel
    # ---------------------------
    # construct sky model
    if "Nskies" not in skyparam:
        skyparam["Nskies"] = Nskies
    else:
        Nskies = skyparam["Nskies"]
    sky_type = skyparam.pop("sky_type")
    savepath = None
    if "savepath" in skyparam:
        savepath = skyparam.pop("savepath")

    sky = sky_model.construct_skymodel(sky_type, **skyparam)

    # If loading a healpix map from disk, confirm its frequencies match the obsparam frequencies.
    if sky_type.lower() not in ["flat_spec", "gsm"]:
        try:
            assert np.allclose(freq_array, sky.freqs)
        except AssertionError:
            print(sky.freqs, freq_array)
            raise ValueError("Obsparam frequencies do not match loaded frequencies.")
    else:
        # write to disk if requested
        if savepath is not None:
            sky.write_hdf5(savepath)

    # ---------------------------
    # UVData object
    # ---------------------------
    uvd_dict = {}

    uvd_dict.update(param_dict["telescope"])
    uvd_dict["freq_array"] = freq_array
    uvd_dict["time_array"] = time_array

    beam_attr = param_dict["beam"].copy()
    pols = beam_attr.pop("pols", None)
    if pols is None:
        warnings.warn("No polarization specified. Defaulting to pI")
        pols = ["pI"]

    uvd_dict["pols"] = pols

    if "select" in param_dict:
        uvd_dict.update(param_dict["select"])

    if "make_full" not in uvd_dict:
        uvd_dict["make_full"] = False

    uv_obj = setup_uvdata(**uvd_dict)

    # ---------------------------
    # Observatory
    # ---------------------------
    beam_type = beam_attr.pop("beam_type")

    beam_freq_interp = beam_attr.pop("beam_freq_interp", "cubic")
    smooth_beam = beam_attr.pop("smooth_beam", False)
    smooth_scale = beam_attr.pop("smooth_scale", None)
    apply_horizon_taper = param_dict.pop("do_horizon_taper", False)
    points = param_dict.pop("pointings", None)
    set_pointings = True
    if points is not None:
        print("Setting pointings from list, not times")
        points = ast.literal_eval(points)
        set_pointings = False
    fov = beam_attr.pop("fov")
    obs = setup_observatory_from_uvdata(
        uv_obj,
        fov=fov,
        set_pointings=set_pointings,
        beam=beam_type,
        beam_kwargs=beam_attr,
        beam_freq_interp=beam_freq_interp,
        smooth_beam=smooth_beam,
        smooth_scale=smooth_scale,
        apply_horizon_taper=apply_horizon_taper,
        pointings=points,
    )
    # ---------------------------
    # Run simulation
    # ---------------------------
    print("Running simulation", flush=True)
    visibility = []
    beam_sq_int = {}
    print(f"Nskies: {sky.Nskies}", flush=True)
    for pol in pols:
        # calculate visibility
        visibs, time_array, baseline_inds = obs.make_visibilities(
            sky, Nprocs=Nprocs, beam_pol=pol
        )
        visibility.append(visibs)
        # Average Beam^2 integral across frequency
        beam_sq_int[f"bm_sq_{pol}"] = obs.beam_sq_int(
            sky.ref_freq, sky.Nside, obs.pointing_centers[0], beam_pol=pol
        ).item()

    visibility = np.moveaxis(visibility, 0, -1)

    # ---------------------------
    # Fill in the UVData object and write out.
    # ---------------------------
    param_history = (
        "\nPARAMETER FILE:\nFILING\n{filing}\nSIMULATION\n{tel}\n{beam}\n"
        "SKYPARAM\n{sky}\n".format(
            filing=param_dict["filing"],
            tel=param_dict["telescope"],
            beam=param_dict["beam"],
            sky=param_dict["skyparam"],
        )
    )
    uv_obj.history = version.history_string(notes=add_to_history + param_history)

    if sjob_id is None:
        sjob_id = ""

    del sky.data  # Free up memory.

    uv_obj = complete_uvdata(uv_obj)

    uv_obj.extra_keywords = {"nside": sky.Nside, "slurm_id": sjob_id, "fov": obs.fov}
    uv_obj.extra_keywords.update(beam_sq_int)
    if beam_type == "gaussian":
        fwhm = beam_attr["gauss_width"] * 2.355
        uv_obj.extra_keywords["bm_fwhm"] = fwhm
    elif beam_type == "airy":
        uv_obj.extra_keywords["bm_diam"] = beam_attr["diameter"]

    if sky.pspec_amp is not None:
        uv_obj.extra_keywords["skysig"] = sky.pspec_amp  # Flat spectrum sources

    for si in range(Nskies):
        # get the sky slice
        vis = visibility[:, si]  # vis = (Nblts, Nfreqs, Npols)
        uv_obj.data_array = vis[:, np.newaxis, :, :]  # (Nblts, Nspws, Nfreqs, Npols)

        uv_obj.check()
        if "format" in filing_params:
            out_format = filing_params["format"]
        else:
            out_format = "uvh5"

        if "outfile_suffix" not in filing_params:
            if Nskies > 1:
                filing_params["outfile_suffix"] = f"{si}sky_uv"
            elif out_format == "miriad":
                filing_params["outfile_suffix"] = "uv"

        if "outfile_name" not in filing_params:
            if "outfile_prefix" not in filing_params:
                outfile_name = "healvis"
            else:
                outfile_name = filing_params["outfile_prefix"]
            if beam_type == "gaussian":
                outfile_name += f"_fwhm{fwhm:.3f}"
            elif beam_type == "airy":
                outfile_name += "_diam{:.2f}".format(beam_attr["diameter"])

        else:
            outfile_name = filing_params["outfile_name"]

        if "outfile_suffix" in filing_params:
            outfile_name = outfile_name + "_" + filing_params["outfile_suffix"]

        if out_format == "miriad":
            outfile_name = os.path.join(filing_params["outdir"], outfile_name + ".uv")
        else:
            outfile_name = os.path.join(
                filing_params["outdir"], outfile_name + f".{out_format}"
            )

        # write base directory if it doesn't exist
        dirname = os.path.dirname(outfile_name)
        if dirname != "" and not os.path.exists(dirname):
            os.mkdir(dirname)

        print(f"...writing {outfile_name}")
        if "clobber" not in filing_params:
            filing_params["clobber"] = False
        if out_format == "uvh5":
            uv_obj.write_uvh5(outfile_name, clobber=filing_params["clobber"])
        elif out_format == "miriad":
            uv_obj.write_miriad(outfile_name, clobber=filing_params["clobber"])
        elif out_format == "uvfits":
            uv_obj.write_uvfits(outfile_name, force_phase=True, spoof_nonessential=True)
        filing_params.pop("outfile_suffix", None)


def run_simulation_partial_freq(
    freq_chans,
    uvh5_file,
    skymod_file,
    fov=180,
    beam=None,
    beam_kwargs={},
    beam_freq_interp="linear",
    smooth_beam=True,
    smooth_scale=2.0,
    Nprocs=1,
    add_to_history=None,
):
    """
    Run a healvis simulation on a selected range of frequency channels.

    Requires a pyuvdata.UVH5 file and SkyModel file (HDF5 format) to exist
    on disk with matching frequencies.

    Args:
        freq_chans : integer 1D array
            Frequency channel indices of uvh5_file to simulate
        uvh5_file : str or UVData
            Filepath to a UVH5 file
        skymod_file : str or SkyModel
            Filepath to a SkyModel file
        beam : str, UVbeam, PowerBeam or AnalyticBeam
            Filepath to beamfits, a UVBeam object, or PowerBeam or AnalyticBeam object
        beam_kwargs : dictionary
            If beam is a viable input to AnalyticBeam, these are its keyword arguments
        beam_freq_interp : str
            Interpolation method if beam is PowerBeam. See scipy.interpolate.interp1d fro details.
        smooth_beam : bool
            If True, and beam is PowerBeam, smooth it across frequency with a Gaussian Process
        smooth_scale : float
            If smoothing the beam, smooth it at this frequency scale [MHz]
        Nprocs : int
            Number of processes for this task
        add_to_history : str
            History string to append to file history. Default is no append to history.

    Result:
        Writes simulation result into uvh5_file
    """
    # load UVH5 metadata
    if isinstance(uvh5_file, str):
        uvd = UVData()
        uvd.read_uvh5(uvh5_file, read_data=False)
    pols = [uvutils.polnum2str(pol) for pol in uvd.polarization_array]

    # load SkyModel
    if isinstance(skymod_file, str):
        sky = sky_model.SkyModel()
        sky.read_hdf5(skymod_file, freq_chans=freq_chans, shared_memory=False)

    # Check that chosen freqs are a subset of the skymodel frequencies.
    assert np.isclose(
        sky.freqs, uvd.freq_array[0, freq_chans]
    ).all(), "Frequency arrays in UHV5 file {} and SkyModel file {} don't agree".format(
        uvh5_file, skymod_file
    )

    # setup observatory
    obs = setup_observatory_from_uvdata(
        uvd,
        fov=fov,
        set_pointings=True,
        beam=beam,
        beam_kwargs=beam_kwargs,
        freq_chans=freq_chans,
        beam_freq_interp=beam_freq_interp,
        smooth_beam=smooth_beam,
        smooth_scale=smooth_scale,
    )

    # run simulation
    visibility = []
    for pol in pols:
        # calculate visibility
        visibs, time_array, baseline_inds = obs.make_visibilities(
            sky, Nprocs=Nprocs, beam_pol=pol
        )
        visibility.append(visibs)

    visibility = np.moveaxis(visibility, 0, -1)
    flags = np.zeros_like(visibility, bool)
    nsamples = np.ones_like(visibility, float)

    # write to disk
    print("...writing to {}".format(uvh5_file))
    uvd.write_uvh5_part(
        uvh5_file,
        visibility,
        flags,
        nsamples,
        freq_chans=freq_chans,
        add_to_history=add_to_history,
    )
