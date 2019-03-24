
from __future__ import absolute_import, division, print_function


import numpy as np
import yaml
import sys
import six
from six.moves import map, range, zip
import os

from pyuvdata import UVData
from pyuvdata import utils as uvutils
import pyuvsim

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
    telescope_config_name = tele_params['telescope_config_name']
    layout_csv = tele_params['array_layout']
    config_path = tele_params['config_dir']
    if not os.path.isdir(config_path):
        config_path = os.path.dirname(config_path)
        if not os.path.isdir(config_path):
            raise ValueError('config_path from yaml is not a directory')
    if not os.path.exists(telescope_config_name):
        telescope_config_name = os.path.join(config_path, telescope_config_name)
        if not os.path.exists(telescope_config_name):
            raise ValueError('telescope_config_name file from yaml does not exist')
    if not os.path.exists(layout_csv):
        layout_csv = os.path.join(config_path, layout_csv)
        if not os.path.exists(layout_csv):
            raise ValueError('layout_csv file from yaml does not exist')

    ant_layout = _parse_layout_csv(layout_csv)
    with open(telescope_config_name, 'r') as yf:
        telconfig = yaml.safe_load(yf)
        tloc = telconfig['telescope_location'][1:-1]  # drop parens
        tloc = list(map(float, tloc.split(",")))
        tloc[0] *= np.pi / 180.
        tloc[1] *= np.pi / 180.   # Convert to radians
        tele_params['telescope_location'] = uvutils.XYZ_from_LatLonAlt(*tloc)

    E, N, U = ant_layout['e'], ant_layout['n'], ant_layout['u']
    antnames = ant_layout['name']
    return_dict = {}

    return_dict['Nants_data'] = antnames.size
    return_dict['Nants_telescope'] = antnames.size
    return_dict['antenna_names'] = np.array(antnames.tolist())
    return_dict['antenna_numbers'] = np.array(ant_layout['number'])
    antpos_enu = np.vstack((E, N, U)).T
    return_dict['antenna_positions'] = uvutils.ECEF_from_ENU(antpos_enu, *tloc) - tele_params['telescope_location']
    return_dict['telescope_config_name'] = telescope_config_name
    return_dict['array_layout'] = layout_csv
    return_dict['telescope_location'] = tele_params['telescope_location']
    return_dict['telescope_name'] = telconfig['telescope_name']

    return return_dict


def run_simulation(param_file, Nprocs=1, sjob_id=None, add_to_history=''):
    """
    Parse input parameter file, construct UVData and SkyModel objects, and run simulation.

    (Moved code from wrapper to here)
    """
    # parse parameter dictionary
    with open(param_file, 'r') as yfile:
        param_dict = yaml.safe_load(yfile)

    print("Making uvdata object")
    sys.stdout.flush()
    tele_dict = parse_telescope_params(param_dict['telescope'].copy())
    freq_dict = pyuvsim.simsetup.parse_frequency_params(param_dict['freq'].copy())
    time_dict = pyuvsim.simsetup.parse_time_params(param_dict['time'].copy())
    filing_params = param_dict['filing']

    # ---------------------------
    # Extra parameters required for healvis
    # ---------------------------
    fov = param_dict['fov']  # Deg
    skyparam = param_dict['skyparam'].copy()
    skyparam['freqs'] = freq_dict['freq_array']
    Nskies = 1 if 'Nskies' not in param_dict else int(param_dict['Nskies'])
    print("Nprocs: ", Nprocs)
    sys.stdout.flush()

    # ---------------------------
    # Observatory
    # ---------------------------
    lat, lon, alt = uvutils.LatLonAlt_from_XYZ(tele_dict['telescope_location'])
    antpos = tele_dict['antenna_positions']
    enu = uvutils.ENU_from_ECEF(tele_dict['antenna_positions'] + tele_dict['telescope_location'], lat, lon, alt)
    anums = tele_dict['antenna_numbers']
    antnames = tele_dict['antenna_names']
    Nants = tele_dict['Nants_data']

    uv_obj = UVData()
    uv_obj.telescope_location = tele_dict['telescope_location']
    uv_obj.telescope_location_lat_lon_alt = (lat, lon, alt)
    uv_obj.telescope_location_lat_lon_alt_degrees = (np.degrees(lat), np.degrees(lon), alt)
    uv_obj.antenna_numbers = anums
    uv_obj.antenna_names = antnames
    uv_obj.antenna_positions = antpos
    uv_obj.Nants_telescope = Nants
    uv_obj.Ntimes = time_dict['Ntimes']
    Ntimes = time_dict['Ntimes']
    uv_obj.freq_array = freq_dict['freq_array']
    uv_obj.Nfreqs = freq_dict['Nfreqs']

    array = []
    bl_array = []
    bls = [(a1, a2) for a2 in anums for a1 in anums if a1 > a2]
    if 'select' in param_dict:
        sel = param_dict['select']
        if 'bls' in sel:
            bls = eval(sel['bls'])
        if 'antenna_nums' in sel:
            antnums = sel['antenna_nums']
            if isinstance(antnums, str):
                antnums = eval(sel['antenna_nums'])
            if isinstance(antnums, int):
                antnums = [antnums]
            bls = [(a1, a2) for (a1, a2) in bls if a1 in antnums or a2 in antnums]
            uv_obj.antenna_nums = antnums
        if 'redundancy' in sel:
            red_tol = sel['redundancy']
            reds, vec_bin_centers, lengths = uvutils.get_antenna_redundancies(anums, enu, tol=red_tol, include_autos=False)
            bls = []
            for rg in reds:
                for r in rg:
                    if r not in bls:
                        bls.append(r)
                        break
    #        bls = [r[0] for r in reds]
            bls = [uvutils.baseline_to_antnums(bl_ind, Nants) for bl_ind in bls]
    uv_obj.Nants_data = np.unique(bls).size
    for (a1, a2) in bls:
        i1, i2 = np.where(anums == a1), np.where(anums == a2)
        array.append(observatory.Baseline(enu[i1], enu[i2]))
        bl_array.append(uvutils.antnums_to_baseline(a1, a2, Nants))
    Nbls = len(bl_array)
    uv_obj.Nbls = Nbls
    uv_obj.Nblts = Nbls * Ntimes

    bl_array = np.array(bl_array)
    freqs = freq_dict['freq_array'][0]  # Hz
    obs = observatory.Observatory(np.degrees(lat), np.degrees(lon), array=array, freqs=freqs)
    obs.set_fov(fov)
    print("Observatory built.")
    print("Nbls: ", Nbls)
    print("Ntimes: ", Ntimes)
    sys.stdout.flush()

    # ---------------------------
    # Pointings
    # ---------------------------
    time_arr = time_dict['time_array']
    obs.set_pointings(time_arr)
    print("Pointings set.")
    sys.stdout.flush()

    # ---------------------------
    # Primary Beam
    # ---------------------------
    beam_attr = param_dict['beam'].copy()
    beam_type = beam_attr.pop("beam_type")
    obs.set_beam(beam_type, **beam_attr)

    # if PowerBeam, interpolate to Observatory frequencies
    if isinstance(obs.beam, beam_model.PowerBeam):
        obs.beam.interp_beam(obs.freqs, inplace=True, kind='cubic')

    # ---------------------------
    # SkyModel
    # ---------------------------
    # construct sky model 
    sky = sky_model.construct_skymodel(skyparam['sky_type'], freqs=obs.freqs, Nside=skyparam['Nside'],
                                       ref_chan=skyparam['ref_chan'], sigma=skyparam['sigma'])
    if not np.isclose(sky.freqs, obs.freqs).all():
        raise ValueError("SkyModel frequencies do not match observation frequencies")
    # write to disk if requested
    if skyparam['savepath'] not in [None, 'None', 'none', ''] and skyparam['sky_type'] not in ['flat_spec', 'gsm']:
        sky.write_hdf5(os.path.join(filing_params['outdir'], savepath))

    # ---------------------------
    # Run simulation
    # ---------------------------
    print("Running simulation")
    sys.stdout.flush()
    visibs, time_array, baseline_inds = obs.make_visibilities(sky, Nprocs=Nprocs)

    # ---------------------------
    # Beam^2 integral
    # ---------------------------
    beam_sq_int = obs.beam_sq_int(freqs, sky.Nside, obs.pointing_centers[0])

    # ---------------------------
    # Fill in the UVData object and write out.
    # ---------------------------
    uv_obj.time_array = time_array
    uv_obj.set_lsts_from_time_array()
    uv_obj.baseline_array = bl_array[baseline_inds]
    uv_obj.ant_1_array, uv_obj.ant_2_array = uv_obj.baseline_to_antnums(uv_obj.baseline_array)

    uv_obj.spw_array = np.array([0])
    uv_obj.Npols = 1
    uv_obj.polarization_array = np.array([1])
    uv_obj.Nspws = 1
    uv_obj.set_uvws_from_antenna_positions()
    uv_obj.channel_width = np.diff(freqs)[0]
    uv_obj.integration_time = np.ones(uv_obj.Nblts) * np.diff(time_arr)[0] * 24 * 3600.  # Seconds
    uv_obj.history = version.history_string(notes=add_to_history + "\n" + yaml.safe_dump(param_dict))
    uv_obj.set_drift()
    uv_obj.telescope_name = 'healvis'
    uv_obj.instrument = 'simulator'
    uv_obj.object_name = 'zenith'
    uv_obj.vis_units = 'Jy'

    if sjob_id is None:
        sjob_id = ''

    uv_obj.extra_keywords = {'nside': sky.Nside, 'slurm_id': sjob_id}
    if beam_type == 'gaussian':
        fwhm = beam_attr['sigma'] * 2.355
        uv_obj.extra_keywords['bm_fwhm'] = fwhm
        uv_obj.extra_keywords['bsq_int'] = beam_sq_int[0]
    elif beam_type == 'airy':
        uv_obj.extra_keywords['bm_diam'] = beam_attr['diameter']
        uv_obj.extra_keywords['bsq_int'] = beam_sq_int[0]

    if sky.pspec_amp is not None:
        uv_obj.extra_keywords['skysig'] = sky.pspec_amp   # Flat spectrum sources

    for sky_i in range(Nskies):
        data_arr = visibs[:, sky_i, :]  # (Nblts, Nskies, Nfreqs)
        data_arr = data_arr[:, np.newaxis, :, np.newaxis]  # (Nblts, Nspws, Nfreqs, Npol)
        uv_obj.data_array = data_arr

        uv_obj.flag_array = np.zeros(uv_obj.data_array.shape).astype(bool)
        uv_obj.nsample_array = np.ones(uv_obj.data_array.shape).astype(float)

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
                filing_params['outfile_prefix'] = \
                    'healvis_{:.2f}hours_Nside{}'.format(Ntimes / (3600. / 11.0), sky.Nside)

            if beam_type == 'gaussian':
                filing_params['outfile_prefix'] += '_fwhm{:.3f}'.format(beam_attr['gauss_width'])
            if beam_type == 'airy':
                filing_params['outfile_prefix'] += '_diam{:.2f}'.format(beam_attr['diameter'])

        while True:
            try:
                pyuvsim.utils.write_uvdata(uv_obj, filing_params, out_format=out_format)  # , run_check=False, run_check_acceptability=False, check_extra=False)
            except ValueError:
                pass
            else:
                break
