from __future__ import absolute_import, division, print_function

import numpy as np
import yaml
import sys
import six
from six.moves import map, range, zip
import os
import ast
import multiprocessing
import copy

from pyuvdata import UVData, UVBeam
from pyuvdata import utils as uvutils

from . import observatory, version, beam_model, sky_model


def _parse_layout_csv(layout_csv):
    """ Interpret the layout csv file """

    with open(layout_csv, 'r') as fhandle:
        header = fhandle.readline()

    header = [h.strip() for h in header.split()]
    if six.PY2:
        str_format_code = 'a'
    else:
        str_format_code = 'U'
    dt = np.format_parser([str_format_code + '10', 'i4', 'i4', 'f8', 'f8', 'f8'],
                          ['name', 'number', 'beamid', 'e', 'n', 'u'], header)

    return np.genfromtxt(layout_csv, autostrip=True, skip_header=1,
                         dtype=dt.dtype)


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
    layout_csv = tele_params['array_layout']
    if not os.path.exists(layout_csv):
        if not os.path.exists(layout_csv):
            raise ValueError('layout_csv file from yaml does not exist')

    ant_layout = _parse_layout_csv(layout_csv)
    if isinstance(tele_params['telescope_location'], (str, np.str)):
        tloc = tele_params['telescope_location'][1:-1]  # drop parens
        tloc = list(map(float, tloc.split(",")))
    else:
        tloc = list(tele_params['telescope_location'])
    tloc[0] *= np.pi / 180.
    tloc[1] *= np.pi / 180.   # Convert to radians
    tloc_xyz = uvutils.XYZ_from_LatLonAlt(*tloc)

    E, N, U = ant_layout['e'], ant_layout['n'], ant_layout['u']
    antnames = ant_layout['name']
    return_dict = {}

    return_dict['Nants_data'] = antnames.size
    return_dict['Nants_telescope'] = antnames.size
    return_dict['antenna_names'] = np.array(antnames.tolist())
    return_dict['antenna_numbers'] = np.array(ant_layout['number'])
    antpos_enu = np.vstack((E, N, U)).T
    return_dict['antenna_positions'] = uvutils.ECEF_from_ENU(antpos_enu, *tloc) - tloc_xyz
    return_dict['array_layout'] = layout_csv
    return_dict['telescope_location'] = tloc_xyz
    return_dict['telescope_name'] = tele_params['telescope_name']

    return return_dict


def parse_freq_params(freq_params):
    """
    Parse the "freq" section of healvis obsparams

    Args:
        freq_params : dictionary

    Returns:
        dictionary
            | Nfreqs : int
            | channel_width : float, [Hz]
            | freq_array : 2D ndarray, shape (1, Nfreqs) [Hz]
    """
    # generate frequencies
    freq_array = np.linspace(freq_params['start_freq'], freq_params['start_freq'] + freq_params['bandwidth'], freq_params['Nfreqs'], endpoint=False).reshape(1, -1)

    # fill return dictionary
    return_dict = {}
    return_dict['Nfreqs'] = freq_params['Nfreqs']
    return_dict['freq_array'] = freq_array
    return_dict['channel_width'] = np.diff(freq_array[0])[0]

    return return_dict


def parse_time_params(time_params):
    """
    Parse the "time" section of healvis obsparams

    Args:
        time_params : dictionary

    Returns:
        dictionary
            | Ntimes : int
            | integration_time : 1D ndarray, shape (Ntimes,) [seconds] all elements set to 1.0
            | time_array : 1D ndarray, shape (Ntimes,) [Julian Date]
            | time_cadence : float, time cadence of integration bins
    """
    # generate times
    time_arr = time_params['start_time'] + np.arange(time_params['Ntimes']) * time_params['time_cadence'] / (24.0 * 3600.0)

    # fill return dictionary
    return_dictionary = {}
    return_dictionary['Ntimes'] = time_params['Ntimes']
    return_dictionary['integration_time'] = np.ones(time_params['Ntimes'], dtype=np.float)
    return_dictionary['time_cadence'] = time_params['time_cadence']
    return_dictionary['time_array'] = time_arr

    return return_dictionary


def setup_uvdata(array_layout=None, telescope_location=None, telescope_name=None,
                 Nfreqs=None, start_freq=None, bandwidth=None, freq_array=None,
                 Ntimes=None, time_cadence=None, start_time=None, time_array=None,
                 bls=None, antenna_nums=None, no_autos=True, pols=['xx'], run_check=True, **kwargs):
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
        antenna_nums : list
            List of antenna numbers to keep in array

    Returns:
        UVData object with zeroed data_array
    """
    # get antenna information
    tele_dict = parse_telescope_params(dict(array_layout=array_layout, telescope_location=telescope_location, telescope_name=telescope_name))
    lat, lon, alt = uvutils.LatLonAlt_from_XYZ(tele_dict['telescope_location'])
    antpos = tele_dict['antenna_positions']
    enu = uvutils.ENU_from_ECEF(tele_dict['antenna_positions'] + tele_dict['telescope_location'], lat, lon, alt)
    anums = tele_dict['antenna_numbers']
    antnames = tele_dict['antenna_names']
    Nants = tele_dict['Nants_data']

    # setup object and fill in basic info
    uv_obj = UVData()
    uv_obj.telescope_location = tele_dict['telescope_location']
    uv_obj.telescope_location_lat_lon_alt = (lat, lon, alt)
    uv_obj.telescope_location_lat_lon_alt_degrees = (np.degrees(lat), np.degrees(lon), alt)
    uv_obj.antenna_numbers = anums
    uv_obj.antenna_names = antnames
    uv_obj.antenna_positions = antpos
    uv_obj.Nants_telescope = Nants

    # fill in frequency and time info: wait to fill in len-Nblts arrays until later
    if freq_array is not None:
        if Nfreqs is not None or start_freq is not None or bandwidth is not None:
            raise ValueError("Cannot specify freq_array as well as Nfreqs, start_freq or bandwidth")
        if freq_array.ndim == 1:
            freq_array = freq_array.reshape(1, -1)
            Nfreqs = freq_array.size
    else:
        freq_dict = parse_freq_params(dict(Nfreqs=Nfreqs, start_freq=start_freq, bandwidth=bandwidth))
        freq_array = freq_dict['freq_array']

    if time_array is not None:
        if Ntimes is not None or start_time is not None or time_cadence is not None:
            raise ValueError("Cannot specify time_array as well as Ntimes, start_time or time_cadence")
        Ntimes = time_array.size
    else:
        time_dict = parse_time_params(dict(Ntimes=Ntimes, start_time=start_time, time_cadence=time_cadence))
        time_array = time_dict['time_array']

    uv_obj.freq_array = freq_array
    uv_obj.Nfreqs = Nfreqs
    uv_obj.Ntimes = Ntimes

    # setup array info
    bl_array = []
    _bls = [(a1, a2) for a1 in anums for a2 in anums if a1 <= a2]
    if bls is not None:
        if isinstance(bls, (str, np.str)):
            bls = ast.literal_eval(bls)
        bls = [bl for bl in _bls if bl in bls]
    else:
        bls = _bls
    if bool(no_autos):
        bls = [bl for bl in bls if bl[0] != bl[1]]
    if antenna_nums is not None:
        if isinstance(antenna_nums, (str, np.str)):
            antenna_nums = ast.literal_eval(antenna_nums)
        if isinstance(antenna_nums, (int, np.int)):
            antenna_nums = [antenna_nums]
        bls = [(a1, a2) for (a1, a2) in bls if a1 in antenna_nums or a2 in antenna_nums]
    bls = sorted(bls)
    for (a1, a2) in bls:
        bl_array.append(uvutils.antnums_to_baseline(a1, a2, 1))
    bl_array = np.asarray(bl_array)

    # fill in Nblts attributes
    uv_obj.baseline_array = np.tile(bl_array, uv_obj.Ntimes)
    uv_obj.ant_1_array, uv_obj.ant_2_array = uv_obj.baseline_to_antnums(uv_obj.baseline_array)
    uv_obj.Nbls = np.unique(uv_obj.baseline_array).size
    uv_obj.Nblts = uv_obj.Nbls * uv_obj.Ntimes
    uv_obj.time_array = np.repeat(time_array, uv_obj.Nbls)
    uv_obj.integration_time = np.repeat(np.ones_like(time_array), uv_obj.Nbls)
    uv_obj.Nants_data = np.unique(bls).size
    uv_obj.set_lsts_from_time_array()

    # fill in other attributes
    uv_obj.spw_array = np.array([0], dtype=np.int)
    uv_obj.Nspws = 1
    uv_obj.polarization_array = np.array([uvutils.polstr2num(pol) for pol in pols], dtype=np.int)
    uv_obj.Npols = uv_obj.polarization_array.size
    uv_obj.set_uvws_from_antenna_positions()
    uv_obj.channel_width = np.diff(uv_obj.freq_array[0])[0]
    uv_obj.set_drift()
    uv_obj.telescope_name = tele_dict['telescope_name']
    uv_obj.instrument = 'simulator'
    uv_obj.object_name = 'zenith'
    uv_obj.vis_units = 'Jy'
    uv_obj.history = ''

    # fill in data
    uv_obj.data_array = np.zeros((uv_obj.Nblts, uv_obj.Nspws, uv_obj.Nfreqs, uv_obj.Npols), dtype=np.complex128)
    uv_obj.flag_array = np.zeros((uv_obj.Nblts, uv_obj.Nspws, uv_obj.Nfreqs, uv_obj.Npols), dtype=np.bool)
    uv_obj.nsample_array = np.ones((uv_obj.Nblts, uv_obj.Nspws, uv_obj.Nfreqs, uv_obj.Npols), dtype=np.float64)

    if run_check:
        uv_obj.check()

    return uv_obj


def setup_observatory_from_uvdata(uv_obj, fov=180, set_pointings=True, beam=None, beam_kwargs={},
                                  beam_freq_interp='cubic', freq_chans=None):
    """
    Setup an Observatory object from a UVData object.

    Args:
        uv_obj : UVData object with metadata
        fov : float
            Field of View (radius) in degrees
        set_pointings : bool
            If True, use time_array to set Observatory pointing centers
        beam : str or UVBeam or PowerBeam or AnalyticBeam
            Filepath to beamfits or a UVBeam object to use as primary beam model.
        beam_kwargs : dictionary
            Beam keyword arguments to pass to Observatory.set_beam if beam is a viable
            input to AnalyticBeam
        beam_freq_interp : str
            Intepolation method of beam across frequency if PowerBeam, see scipy.interp1d for details
        freq_chans : integer 1D array
            Frequency channel indices to use from uv_obj when setting observatory freqs.

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

    lat, lon, alt = uv_obj.telescope_location_lat_lon_alt
    if freq_chans is None:
        freq_chans = slice(None)
    obs = observatory.Observatory(np.degrees(lat), np.degrees(lon), array=bls, freqs=uv_obj.freq_array[0, freq_chans])

    # set FOV
    obs.set_fov(fov)

    # set pointings
    if set_pointings:
        obs.set_pointings(np.unique(uv_obj.time_array))

    # set beam
    if beam is not None:
        if isinstance(beam, UVBeam):
            obs.beam = copy.deepcopy(beam)
            obs.beam.__class__ = beam_model.PowerBeam
            obs.beam.interp_freq(obs.freqs, inplace=True, kind=beam_freq_interp)

        elif isinstance(beam, (str, np.str)) or callable(beam):
            obs.set_beam(beam, **beam_kwargs)

        elif isinstance(beam, beam_model.PowerBeam):
            obs.beam = beam.interp_freq(obs.freqs, inplace=False, kind=beam_freq_interp)

        elif isinstance(beam, beam_model.AnalyticBeam):
            obs.beam = beam

    return obs


def run_simulation(param_file, Nprocs=1, sjob_id=None, add_to_history=''):
    """
    Parse input parameter file, construct UVData and SkyModel objects, and run simulation.

    (Moved code from wrapper to here)
    """
    # parse parameter dictionary
    if isinstance(param_file, (str, np.str)):
        with open(param_file, 'r') as yfile:
            param_dict = yaml.safe_load(yfile)
    else:
        param_dict = param_file

    sys.stdout.flush()
    freq_dict = parse_freq_params(param_dict['freq'])
    freq_array = freq_dict['freq_array'][0]
    filing_params = param_dict['filing']

    # ---------------------------
    # Extra parameters required for healvis
    # ---------------------------
    skyparam = param_dict['skyparam'].copy()
    skyparam['freqs'] = freq_dict['freq_array']

    Nskies = 1 if 'Nskies' not in param_dict else int(param_dict['Nskies'])
    print("Nprocs: ", Nprocs)
    sys.stdout.flush()

    # ---------------------------
    # SkyModel
    # ---------------------------
    # construct sky model
    sky = sky_model.construct_skymodel(skyparam['sky_type'], freqs=freq_array, Nside=skyparam['Nside'],
                                       ref_chan=skyparam['ref_chan'], sigma=skyparam['sigma'], Nskies=Nskies)

    # If loading a healpix map from disk, use those frequencies instead of ones specified in obsparam
    if skyparam['sky_type'].lower() not in ['flat_spec', 'gsm']:
        param_dict['freq'] = {'freq_array': sky.freqs}
    else:
        # write to disk if requested
        if skyparam['savepath'] not in [None, 'None', 'none', '']:
            sky.write_hdf5(os.path.join(filing_params['outdir'], savepath))

    # ---------------------------
    # UVData object
    # ---------------------------
    uvd_dict = {}
    uvd_dict.update(param_dict['telescope'])
    uvd_dict.update(param_dict['freq'])
    uvd_dict.update(param_dict['time'])
    uvd_dict.update(param_dict['beam'])
    uvd_dict.update(param_dict['select'])
    uv_obj = setup_uvdata(**uvd_dict)

    # ---------------------------
    # Observatory
    # ---------------------------
    beam_attr = param_dict['beam'].copy()
    beam_type = beam_attr.pop("beam_type")
    obs = setup_observatory_from_uvdata(uv_obj, fov=param_dict['beam'].pop("fov"), set_pointings=True,
                                        beam=beam_type, beam_kwargs=beam_attr, beam_freq_interp='cubic')

    # ---------------------------
    # Run simulation
    # ---------------------------
    print("Running simulation")
    sys.stdout.flush()
    visibility = []
    beam_sq_int = {}
    for pol in param_dict['beam']['pols']:
        # calculate visibility
        visibs, time_array, baseline_inds = obs.make_visibilities(sky, Nprocs=Nprocs, beam_pol=pol)
        visibility.append(visibs)
        # Average Beam^2 integral across frequency
        beam_sq_int['bm_sq_{}'.format(pol)] = np.asscalar(obs.beam_sq_int(obs.freqs[sky.ref_chan], sky.Nside, obs.pointing_centers[0], beam_pol=pol))

    visibility = np.moveaxis(visibility, 0, -1)

    # ---------------------------
    # Fill in the UVData object and write out.
    # ---------------------------
    param_history = "\nPARAMETER FILE:\nFILING\n{filing}\nSIMULATION\n{tel}\n{beam}\n" \
                    "SKYPARAM\n{sky}\n".format(filing=param_dict['filing'], tel=param_dict['telescope'], beam=param_dict['beam'],
                                               sky=param_dict['skyparam'])
    uv_obj.history = version.history_string(notes=add_to_history + param_history)

    if sjob_id is None:
        sjob_id = ''

    uv_obj.extra_keywords = {'nside': sky.Nside, 'slurm_id': sjob_id}
    uv_obj.extra_keywords.update(beam_sq_int)
    if beam_type == 'gaussian':
        fwhm = beam_attr['sigma'] * 2.355
        uv_obj.extra_keywords['bm_fwhm'] = fwhm
    elif beam_type == 'airy':
        uv_obj.extra_keywords['bm_diam'] = beam_attr['diameter']

    if sky.pspec_amp is not None:
        uv_obj.extra_keywords['skysig'] = sky.pspec_amp   # Flat spectrum sources

    for si in range(Nskies):
        sky_i = slice(si, si + 1)
        uv_obj.data_array = visibility[:, sky_i, :, :]  # (Nblts, Nspws, Nfreqs, Npols)

        uv_obj.check()

        if 'format' in filing_params:
            out_format = filing_params['format']
        else:
            out_format = 'uvh5'

        if Nskies > 1:
            filing_params['outfile_suffix'] = '{}sky_uv'.format(sky_i)
        elif out_format == 'miriad':
            filing_params['outfile_suffix'] = 'uv'

        if 'outfile_name' not in filing_params:
            if 'outfile_prefix' not in filing_params:
                outfile_name = "healvis"
            else:
                outfile_name = filing_params['outfile_prefix']
            if beam_type == 'gaussian':
                outfile_name += '_fwhm{:.3f}'.format(beam_attr['gauss_width'])
            elif beam_type == 'airy':
                outfile_name += '_diam{:.2f}'.format(beam_attr['diameter'])

        else:
            outfile_name = filing_params['outfile_name']
        outfile_name = os.path.join(filing_params['outdir'], outfile_name + ".{}".format(out_format))

        # write base directory if it doesn't exist
        dirname = os.path.dirname(outfile_name)
        if dirname != '' and not os.path.exists(dirname):
            os.mkdir(dirname)

        print("...writing {}".format(outfile_name))
        if out_format == 'uvh5':
            uv_obj.write_uvh5(outfile_name, clobber=filing_params['clobber'])
        elif out_format == 'miriad':
            uv_obj.write_miriad(outfile_name, clobber=filing_params['clobber'])
        elif out_format == 'uvfits':
            uv_obj.write_uvfits(outfile_name)


def run_simulation_partial_freq(freq_chans, uvh5_file, skymod_file, fov=180, beam=None, beam_kwargs={}, Nprocs=1):
    """
    Run a healvis simulation on a selected range of frequency channels.

    Requires a pyuvdata.UVH5 file and SkyModel file (HDF5 format) to exist
    on disk with matching frequencies.

    Args:
        freq_chans : integer 1D array
            Frequency channel indices of uvh5_file to simulate
        uvh5_file : str
            Filepath to a UVH5 file
        skymod_file : str
            Filepath to a SkyModel file
        beam : str, UVbeam, PowerBeam or AnalyticBeam
            Filepath to beamfits, a UVBeam object, or PowerBeam or AnalyticBeam object
        beam_kwargs : dictionary
            If beam is a viable input to AnalyticBeam, these are its keyword arguments
        Nprocs : int
            Number of processes for this task

    Result:
        Writes simulation result into uvh5_file
    """
    # load UVH5 metadata
    uvd = UVData()
    uvd.read_uvh5(uvh5_file, read_data=False)
    pols = [uvutils.polnum2str(pol) for pol in uvd.polarization_array]
 
    # load SkyModel
    sky = sky_model.SkyModel()
    sky.read_hdf5(skymod_file, freq_chans=freq_chans, shared_mem=False)

    assert np.isclose(sky.freqs, uvd.freq_array[0, freq_chans]).all(), "Frequency arrays in UHV5 file {} and SkyModel file {} don't agree".format(uvh5_file, skymod_file)

    # setup observatory
    obs = setup_observatory_from_uvdata(uvd, fov=fov, set_pointings=True, beam=beam, beam_kwargs=beam_kwargs,
                                        freq_chans=freq_chans)

    # run simulation
    visibility = []
    beam_sq_int = {}
    for pol in pols:
        # calculate visibility
        visibs, time_array, baseline_inds = obs.make_visibilities(sky, Nprocs=Nprocs, beam_pol=pol)
        visibility.append(visibs)

    visibility = np.moveaxis(visibility, 0, -1)
    flags = np.zeros_like(visibility, np.bool)
    nsamples = np.ones_like(visibility, np.float)

    # write to disk
    print("...writing to {}".format(uvh5_file))
    uvd.write_uvh5_part(uvh5_file, visibility, flags, nsamples, freq_chans=freq_chans)
