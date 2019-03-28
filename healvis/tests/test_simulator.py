# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

from __future__ import absolute_import, division, print_function

import numpy as np
import healpy as hp
import os
import yaml
import shutil
import nose.tools as nt
from astropy.cosmology import Planck15

from pyuvdata import UVData

from healvis import sky_model, simulator, observatory, beam_model, utils
from healvis.data import DATA_PATH


def test_setup_uvdata():
    # check it runs through
    uvd = simulator.setup_uvdata(array_layout=os.path.join(DATA_PATH, "configs/HERA65_layout.csv"),
                                 telescope_location=(-30.72152777777791, 21.428305555555557, 1073.0000000093132),
                                 telescope_name="HERA", Nfreqs=10, start_freq=1e8, bandwidth=1e8, Ntimes=60,
                                 time_cadence=100.0, start_time=2458101.0, pols=['xx'], no_autos=True, make_full=True, run_check=True)
    nt.assert_equal(uvd.Nbls, uvd.Nants_data * (uvd.Nants_data - 1) / 2)

    # check selection works
    bls = [(0, 11), (0, 12), (0, 13)]
    uvd = simulator.setup_uvdata(array_layout=os.path.join(DATA_PATH, "configs/HERA65_layout.csv"),
                                 telescope_location=(-30.72152777777791, 21.428305555555557, 1073.0000000093132),
                                 telescope_name="HERA", Nfreqs=10, start_freq=1e8, bandwidth=1e8, Ntimes=60,
                                 time_cadence=100.0, start_time=2458101.0, pols=['xx'], bls=bls, make_full=True, run_check=True)
    nt.assert_equal(uvd.Nbls, len(bls))
    uvd = simulator.setup_uvdata(array_layout=os.path.join(DATA_PATH, "configs/HERA65_layout.csv"),
                                 telescope_location=(-30.72152777777791, 21.428305555555557, 1073.0000000093132),
                                 telescope_name="HERA", Nfreqs=10, start_freq=1e8, bandwidth=1e8, Ntimes=60,
                                 time_cadence=100.0, start_time=2458101.0, pols=['xx'], bls=bls, make_full=True, antenna_nums=[11],
                                 no_autos=False, run_check=True)
    nt.assert_equal(uvd.Nbls, 1)


def test_run_simulation():
    # load parameter file from test directory
    param_file = os.path.join(DATA_PATH, "configs/obsparam_test.yaml")
    with open(param_file, 'r') as _f:
        param_dict = yaml.safe_load(_f)

    # edit
    param_dict['telescope']['array_layout'] = os.path.join(DATA_PATH + '/configs', os.path.basename(param_dict['telescope']['array_layout']))
    param_dict['beam']['beam_type'] = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    param_dict['skyparam']['sky_type'] = os.path.join(DATA_PATH, "gsm_nside32.hdf5")
    param_dict['filing']['outdir'] = os.path.join(DATA_PATH, "sim_testing_out")
    param_dict['filing']['outfile_name'] = 'test_sim'
    param_dict['filing']['format'] = 'uvh5'

    # run simulation
    simulator.run_simulation(param_dict, add_to_history='foo')

    # load result
    uvd = UVData()
    uvd.read(os.path.join(param_dict['filing']['outdir'], "test_sim.uvh5"))

    # basic checks
    nt.assert_equal(uvd.Nfreqs, 10)
    nt.assert_equal(uvd.Ntimes, 5)
    nt.assert_equal(uvd.Nbls, 4)
    nt.assert_equal(uvd.Npols, 2)
    nt.assert_true('foo' in uvd.history)  # check add_to_history was propagated
    nt.assert_true("SKYPARAM" in uvd.history)  # check param_dict was written to history

    # test data_array ordering is correct--i.e. Nbls, Ntimes--by asserting that auto-correlation is purely real for all times
    nt.assert_true(np.isclose(uvd.get_data(0, 0).imag, 0.0).all())

    # erase output directory
    shutil.rmtree(param_dict['filing']['outdir'])


def test_run_simulation_partial_freq():
    # read gsm test file
    skymod_file = os.path.join(DATA_PATH, "gsm_nside32.hdf5")
    sky = sky_model.SkyModel()
    sky.read_hdf5(skymod_file)

    # setup uvdata to match freq of gsm test file
    bls = [(0, 11), (0, 12), (0, 13)]
    uvd = simulator.setup_uvdata(array_layout=os.path.join(DATA_PATH, "configs/HERA65_layout.csv"),
                                 telescope_location=(-30.72152777777791, 21.428305555555557, 1073.0000000093132),
                                 telescope_name="HERA", Ntimes=60, time_cadence=100.0, start_time=2458101.0,
                                 pols=['xx'], bls=bls, make_full=True, run_check=True, freq_array=sky.freqs)
    test_uvh5 = os.path.join(DATA_PATH, "test_freq_parallel_sim.uvh5")
    uvd.write_uvh5(test_uvh5, clobber=True)

    # run simulation
    beamfile = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    beam = beam_model.PowerBeam(beamfile)
    freq_chans = np.arange(3)
    simulator.run_simulation_partial_freq(freq_chans, test_uvh5, skymod_file, fov=180, beam=beam)

    # test that partial frequency sim worked
    uvd2 = UVData()
    uvd2.read_uvh5(test_uvh5)
    d = uvd2.get_data(0, 11)
    nt.assert_false(np.isclose(d[:, freq_chans], 0.0).all())
    nt.assert_true(np.isclose(d[:, freq_chans[-1] + 1:], 0.0).all())

    # clean up
    os.remove(test_uvh5)


def test_setup_light_uvdata():
    skymod_file = os.path.join(DATA_PATH, "gsm_nside32.hdf5")
    sky = sky_model.SkyModel()
    sky.read_hdf5(skymod_file)

    fov = 30
    beam_type = 'uniform'

    # setup uvdata to match freq of gsm test file
    bls = [(0, 11), (0, 12), (0, 13)]
    uvd = simulator.setup_uvdata(array_layout=os.path.join(DATA_PATH, "configs/HERA65_layout.csv"),
                                 telescope_location=(-30.72152777777791, 21.428305555555557, 1073.0000000093132),
                                 telescope_name="HERA", Ntimes=60, time_cadence=100.0, start_time=2458101.0,
                                 pols=['xx'], bls=bls, make_full=False, freq_array=sky.freqs)

    obs = simulator.setup_observatory_from_uvdata(uvd, fov=30, set_pointings=True, beam=beam_type)

    uvd_full = simulator.setup_uvdata(array_layout=os.path.join(DATA_PATH, "configs/HERA65_layout.csv"),
                                      telescope_location=(-30.72152777777791, 21.428305555555557, 1073.0000000093132),
                                      telescope_name="HERA", Ntimes=60, time_cadence=100.0, start_time=2458101.0,
                                      pols=['xx'], bls=bls, make_full=True, run_check=True, freq_array=sky.freqs)

    obs_full = simulator.setup_observatory_from_uvdata(uvd_full, fov=30, set_pointings=True, beam=beam_type)

    nt.assert_true(len(uvd.baseline_array) == len(bls))
    nt.assert_true(len(uvd.time_array) == uvd.Ntimes)       # If the "light" weren't enabled, these would be length Nblts

    # Check that baselines, times, and frequencies all match
    for bi, bl in enumerate(bls):
        nt.assert_true(np.all(obs_full.array[bi].enu == obs.array[bi].enu))

    nt.assert_true(np.all(obs_full.freqs == obs.freqs))


def test_freq_time_params():
    sky = sky_model.SkyModel()
    sky.read_hdf5(os.path.join(DATA_PATH, 'gsm_nside32.hdf5'))
    freqs = sky.freqs
    times = np.linspace(2458570, 2458570 + 0.5, 239)
    time_dict = utils.time_array_to_params(times)
    freq_dict = utils.freq_array_to_params(freqs)
    ftest = simulator.parse_frequency_params(freq_dict)
    ttest = simulator.parse_time_params(time_dict)
    nt.assert_true(np.allclose(ftest['freq_array'], freqs))
    nt.assert_true(np.allclose(ttest['time_array'], times))
