
from __future__ import absolute_import, division, print_function


import numpy as np

from pyuvdata import UVData
from pyuvdata import utils as uvutils
import pyuvsim

from . import visibility
from .skymodel import skymodel


def parse_skymodel(param_dict):
    """
    Either load a skymodel from file or construct.
    """

    sky = skymodel()
    if 'filepath' in param_dict:
        sky.read_hdf5(param_dict['filepath'])
        return sky

    # Construct a flat spectrum shell.
    required = ['freq_array', 'ref_chan', 'Nside', 'sigma']]
    missing = [p for p in required if p not in param_dict.keys()]
    if len(missing) > 0:
        raise KeyError("Missing required parameters for shell construction: " + ", ".join(missing))
    sky.Nside = param_dict['Nside']
    sky.freq_array = param_dict['freq_array']
    sky.ref_chan = param_dict['ref_chan']
    sky.make_flat_spectrum_shell(param_dict['sigma'], shared_mem=True)
    return sky


def run_simulation(param_file, Nprocs=None, sjob_id=None):
    """
    Parse input parameter file, construct UVData and skymodel objects, and run simulation.

    (Moved code from wrapper to here)
    """

    with open(param_file, 'r') as yfile:
        param_dict = yaml.safe_load(yfile)

    param_dict['config_path'] = '.'

    print("Making uvdata object")
    sys.stdout.flush()
    tele_dict, beam_list, beam_dict = pyuvsim.simsetup.parse_telescope_params(param_dict['telescope'], param_dict['config_path'])
    freq_dict = pyuvsim.simsetup.parse_frequency_params(param_dict['freq'])
    time_dict = pyuvsim.simsetup.parse_time_params(param_dict['time'])

    filing_params = param_dict['filing']

    beam_list = [pyuvsim.simsetup.beam_string_to_object(b) for b in beam_list]

    # ---------------------------
    # Extra parameters required for healvis
    # ---------------------------

    fov= param_dict['fov']  # Deg
    skyparam = param_dict['skymodel']
    skyparam['freq_array'] = freq_dict['freq_array']
    Nskies = 1 if 'Nskies' not in param_dict else int(param_dict['Nskies'])
    try:
        beam_select = int(param_dict['beam_select'])
    except KeyError:
        beam_select= 0


    beam = beam_list[beam_select]
    beam_type = beam.type
    if beam_type == 'gaussian':
        beam_attr= {'sigma': np.degrees(beam.sigma)}
    elif beam_type == 'airy':
        beam_attr = {'diameter': beam.diameter}
    elif beam_type == 'uniform':
        beam_attr = {}
    else:
        raise ValueError("UVBeam is not yet supported")

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
    bl_array =  []
    bls= [(a1, a2) for a2 in anums for a1 in anums if a1 > a2]
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
        array.append(visibility.baseline(enu[i1], enu[i2]))
        bl_array.append(uvutils.antnums_to_baseline(a1, a2, Nants))
    Nbls = len(bl_array)
    uv_obj.Nbls = Nbls
    uv_obj.Nblts = Nbls * Ntimes

    bl_array = np.array(bl_array)
    freqs= freq_dict['freq_array'][0]  # Hz
    obs = visibility.observatory(np.degrees(lat), np.degrees(lon), array=array, freqs=freqs)
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
    # Primary beam
    # ---------------------------
    obs.set_beam(beam_type, **beam_attr)

    # ---------------------------
    # Beam^2 integral
    # ---------------------------
    za, az = obs.calc_azza(Nside, obs.pointing_centers[0])
    beam_sq_int = np.sum(obs.beam.beam_val(az[: , np.newaxis], za[: , np.newaxis], freqs[np.newaxis, : ])**2, axis=0)
    beam_sq_int = beam_sq_int * om

    # ---------------------------
    # Skymodel
    # ---------------------------

    skymodel = parse_skymodel(skyparam)

    # ---------------------------
    # Run simulation
    # ---------------------------

    print("Running simulation")
    sys.stdout.flush()
    visibs, time_array, baseline_inds = obs.make_visibilities(skymodel, Nprocs=Nprocs)

    # ---------------------------
    # Fill in the UVData object and write out.
    # ---------------------------

    uv_obj.time_array = time_array
    uv_obj.set_lsts_from_time_array()
    uv_obj.baseline_array = bl_array[baseline_inds]
    uv_obj.ant_1_array, uv_obj.ant_2_array = uv_obj.baseline_to_antnums(uv_obj.baseline_array)

    uv_obj.spw_array = np.array([0])
    uv_obj.Npols = 1
    uv_obj.polarization_array= np.array([1])
    uv_obj.Nspws = 1
    uv_obj.set_uvws_from_antenna_positions()
    uv_obj.channel_width = np.diff(freqs)[0]
    uv_obj.integration_time = np.ones(uv_obj.Nblts) * np.diff(time_arr)[0] * 24 * 3600.  # Seconds
    uv_obj.history = 'healvis'
    uv_obj.set_drift()
    uv_obj.telescope_name = 'healvis'
    uv_obj.instrument = 'simulator'
    uv_obj.object_name = 'zenith'
    uv_obj.vis_units = 'Jy'

    if sjob_id is None:
        sjob_id = ''

    if beam_type == 'gaussian':
        fwhm = beam_attr['sigma'] * 2.355
        uv_obj.extra_keywords = {'bsq_int': beam_sq_int[0], 'skysig': sky_sigma, 'bm_fwhm': fwhm, 'nside': Nside, 'slurm_id': sjob_id}
    elif beam_type == 'airy':
        uv_obj.extra_keywords = {'bsq_int': beam_sq_int, 'skysig': sky_sigma, 'nside': Nside, 'slurm_id': sjob_id}
            # Since the beam is frequency dependent, we need to pass along the full set of beam integrals somehow.
    else:
        uv_obj.extra_keywords = {'bsq_int': beam_sq_int[0], 'skysig': sky_sigma, 'nside': Nside, 'slurm_id': sjob_id}

    for sky_i in range(Nskies):

        data_arr = visibs[: , sky_i, : ]  # (Nblts, Nskies, Nfreqs)
        data_arr = data_arr[:, np.newaxis, : , np.newaxis]  # (Nblts, Nspws, Nfreqs, Npol)
        uv_obj.data_array = data_arr

        uv_obj.flag_array = np.zeros(uv_obj.data_array.shape).astype(bool)
        uv_obj.nsample_array = np.ones(uv_obj.data_array.shape).astype(float)

        uv_obj.check()

        if Nskies > 1:
            filing_params['outfile_suffix'] = '{}sky_uv'.format(sky_i)
        else:
            filing_params['outfile_suffix'] = 'uv'
        filing_params['outfile_prefix'] = \
                      'healvis_{:.2f}hours_Nside{}_sigma{:.3f}'.format(Ntimes / (3600. / 11.0), Nside, sky_sigma)

        if beam_type == 'gaussian':
            filing_params['outfile_prefix'] += '_fwhm{:.3f}'.format(fwhm)
        if beam_type == 'airy':
            filing_params['outfile_prefix'] += '_diam{:.2f}'.format(beam.diameter)

        while True:
            try:
                pyuvsim.utils.write_uvdata(uv_obj, filing_params, out_format='uvh5')  # , run_check=False, run_check_acceptability=False, check_extra=False)
            except ValueError:
                pass
            else:
                break
