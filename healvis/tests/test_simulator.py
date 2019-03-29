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


def test_parse_freq_params():
    # define global set of keys that all agree
    start_freq = 1.0
    channel_width = 1.0
    end_freq = 10.0
    bandwidth = 10.0
    Nfreqs = 10
    freq_array = np.linspace(start_freq, start_freq + bandwidth, Nfreqs, endpoint=False)
    master_fdict = dict(freq_array=freq_array, Nfreqs=Nfreqs, bandwidth=bandwidth, channel_width=channel_width)

    # now test that parsing of various key combinations all agree
    # case i
    fdict = simulator.parse_frequency_params(dict(start_freq=start_freq, Nfreqs=Nfreqs, channel_width=channel_width))
    for k in master_fdict:
        nt.assert_true(np.all(np.isclose(master_fdict[k], fdict[k])))
    # case ii
    fdict = simulator.parse_frequency_params(dict(start_freq=start_freq, Nfreqs=Nfreqs, bandwidth=bandwidth))
    for k in master_fdict:
        nt.assert_true(np.all(np.isclose(master_fdict[k], fdict[k])))
    # case iii
    fdict = simulator.parse_frequency_params(dict(start_freq=start_freq, Nfreqs=Nfreqs, end_freq=end_freq))
    for k in master_fdict:
        nt.assert_true(np.all(np.isclose(master_fdict[k], fdict[k])))
    # case iv
    fdict = simulator.parse_frequency_params(dict(start_freq=start_freq, end_freq=end_freq, channel_width=channel_width))
    for k in master_fdict:
        nt.assert_true(np.all(np.isclose(master_fdict[k], fdict[k])))

    # test that if freq_array is present, it supercedes!
    _farray = np.linspace(100, 200, 10, endpoint=False)
    fdict = simulator.parse_frequency_params(dict(start_freq=start_freq, end_freq=end_freq, channel_width=channel_width, freq_array=_farray))
    nt.assert_true(np.all(np.isclose(_farray, fdict['freq_array'])))

    # test Nfreqs = 1 exception
    nt.assert_raises(ValueError, simulator.parse_frequency_params, dict(freq_array=np.array([100.0])))

    # test evenly divisible exception
    nt.assert_raises(ValueError, simulator.parse_frequency_params, dict(start_freq=100, end_freq=101, channel_width=0.33))

    # test improper combination KeyError
    nt.assert_raises(KeyError, simulator.parse_frequency_params, dict(start_freq=100.0))


def test_parse_time_params():
    # define global set of keys that all agree
    start_time = 2458101.0
    time_cadence = 100.0
    Ntimes = 10
    duration = Ntimes * time_cadence / (24. * 3600.) 
    time_array = np.linspace(start_time, start_time + duration, Ntimes, endpoint=False)
    end_time = time_array[-1]
    master_tdict = dict(time_array=time_array, Ntimes=Ntimes, duration=duration, time_cadence=time_cadence)

    # now test that parsing of various key combinations all agree
    # case i
    tdict = simulator.parse_time_params(dict(start_time=start_time, Ntimes=Ntimes, time_cadence=time_cadence))
    for k in master_tdict:
        nt.assert_true(np.all(np.isclose(master_tdict[k], tdict[k])))
    # case ii
    tdict = simulator.parse_time_params(dict(start_time=start_time, Ntimes=Ntimes, duration=duration))
    for k in master_tdict:
        nt.assert_true(np.all(np.isclose(master_tdict[k], tdict[k])))
    # case iii
    tdict = simulator.parse_time_params(dict(start_time=start_time, Ntimes=Ntimes, end_time=end_time))
    for k in master_tdict:
        nt.assert_true(np.all(np.isclose(master_tdict[k], tdict[k])))
    # case iv
    tdict = simulator.parse_time_params(dict(start_time=start_time, end_time=end_time, time_cadence=time_cadence))
    for k in master_tdict:
        nt.assert_true(np.all(np.isclose(master_tdict[k], tdict[k])))

    # test that if time_array is present, it supercedes!
    _tarray = np.linspace(2458201, 2458201.10, 10, endpoint=False)
    tdict = simulator.parse_time_params(dict(start_time=start_time, end_time=end_time, time_cadence=time_cadence, time_array=_tarray))
    nt.assert_true(np.all(np.isclose(_tarray, tdict['time_array'])))

    # test Ntimes = 1 exception
    nt.assert_raises(ValueError, simulator.parse_time_params, dict(time_array=np.array([100.0])))

    # test evenly divisible exception
    nt.assert_raises(ValueError, simulator.parse_time_params, dict(start_time=100, end_time=101, time_cadence=0.33))

    # test improper combination KeyError
    nt.assert_raises(KeyError, simulator.parse_time_params, dict(start_time=100.0))

