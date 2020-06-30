# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

import numpy as np
from astropy_healpix import healpy as hp
import os
import pytest
import yaml
import shutil
from astropy.cosmology import Planck15

from pyuvdata import UVData

from healvis import sky_model, simulator, observatory, beam_model, utils
from healvis.data import DATA_PATH
import healvis.tests as simtest


def test_setup_uvdata():
    # check it runs through
    uvd = simulator.setup_uvdata(array_layout=os.path.join(DATA_PATH, "configs/HERA65_layout.csv"),
                                 telescope_location=(-30.72152777777791, 21.428305555555557, 1073.0000000093132),
                                 telescope_name="HERA", Nfreqs=10, start_freq=1e8, bandwidth=1e8, Ntimes=60,
                                 time_cadence=100.0, start_time=2458101.0, pols=['xx'], no_autos=True, make_full=True, run_check=True)
    assert uvd.Nbls == uvd.Nants_data * (uvd.Nants_data - 1) / 2

    # check selection works
    bls = [(0, 11), (0, 12), (0, 13)]
    uvd = simulator.setup_uvdata(array_layout=os.path.join(DATA_PATH, "configs/HERA65_layout.csv"),
                                 telescope_location=(-30.72152777777791, 21.428305555555557, 1073.0000000093132),
                                 telescope_name="HERA", Nfreqs=10, start_freq=1e8, bandwidth=1e8, Ntimes=60,
                                 time_cadence=100.0, start_time=2458101.0, pols=['xx'], bls=bls, make_full=True, run_check=True)
    assert uvd.Nbls == len(bls)
    uvd = simulator.setup_uvdata(array_layout=os.path.join(DATA_PATH, "configs/HERA65_layout.csv"),
                                 telescope_location=(-30.72152777777791, 21.428305555555557, 1073.0000000093132),
                                 telescope_name="HERA", Nfreqs=10, start_freq=1e8, bandwidth=1e8, Ntimes=60,
                                 time_cadence=100.0, start_time=2458101.0, pols=['xx'], bls=bls, make_full=True, antenna_nums=[11],
                                 no_autos=False, run_check=True)
    assert uvd.Nbls == 1


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
    assert uvd.Nfreqs == 10
    assert uvd.Ntimes == 5
    assert uvd.Nbls == 4
    assert uvd.Npols == 2
    assert 'foo' in uvd.history  # check add_to_history was propagated
    assert "SKYPARAM" in uvd.history  # check param_dict was written to history

    # test data_array ordering is correct--i.e. Nbls, Ntimes--by asserting that auto-correlation is purely real for all times
    assert np.isclose(uvd.get_data(0, 0).imag, 0.0).all()

    # erase output directory
    shutil.rmtree(param_dict['filing']['outdir'])

@pytest.mark.skip
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
    test_uvh5 = os.path.join(simtest.TESTDATA_PATH, "test_freq_parallel_sim.uvh5")
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
    assert not (np.isclose(d[:, freq_chans], 0.0).all())
    assert np.isclose(d[:, freq_chans[-1] + 1:], 0.0).all()

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

    assert len(uvd.baseline_array) == len(bls)
    assert len(uvd.time_array) == uvd.Ntimes      # If the "light" weren't enabled, these would be length Nblts

    # Check that baselines, times, and frequencies all match
    for bi, bl in enumerate(bls):
        assert np.all(obs_full.array[bi].enu == obs.array[bi].enu)

    assert np.all(obs_full.freqs == obs.freqs)


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
        assert np.all(np.isclose(master_fdict[k], fdict[k]))
    # case ii
    fdict = simulator.parse_frequency_params(dict(start_freq=start_freq, Nfreqs=Nfreqs, bandwidth=bandwidth))
    for k in master_fdict:
        assert np.all(np.isclose(master_fdict[k], fdict[k]))
    # case iii
    fdict = simulator.parse_frequency_params(dict(start_freq=start_freq, Nfreqs=Nfreqs, end_freq=end_freq))
    for k in master_fdict:
        assert np.all(np.isclose(master_fdict[k], fdict[k]))
    # case iv
    fdict = simulator.parse_frequency_params(dict(start_freq=start_freq, end_freq=end_freq, channel_width=channel_width))
    for k in master_fdict:
        assert np.all(np.isclose(master_fdict[k], fdict[k]))

    # test that if freq_array is present, it supercedes!
    _farray = np.linspace(100, 200, 10, endpoint=False)
    fdict = simulator.parse_frequency_params(dict(start_freq=start_freq, end_freq=end_freq, channel_width=channel_width, freq_array=_farray))
    assert np.all(np.isclose(_farray, fdict['freq_array']))

    # test Nfreqs = 1 exception
    simtest.assert_raises_message(ValueError, 'Channel width must be specified if passed freq_arr has length 1',
                                  simulator.parse_frequency_params, dict(freq_array=np.array([100.0])))

    # test evenly divisible exception
    simtest.assert_raises_message(ValueError, 'end_freq - start_freq must be evenly divisible by channel_width',
                                  simulator.parse_frequency_params, dict(start_freq=100, end_freq=101, channel_width=0.33))

    # test improper combination KeyError
    simtest.assert_raises_message(KeyError, "Couldn't find any proper combination of keys in freq_params",
                                  simulator.parse_frequency_params, dict(start_freq=100.0))


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
        assert np.all(np.isclose(master_tdict[k], tdict[k]))
    # case ii
    tdict = simulator.parse_time_params(dict(start_time=start_time, Ntimes=Ntimes, duration=duration))
    for k in master_tdict:
        assert np.all(np.isclose(master_tdict[k], tdict[k]))
    # case iii
    tdict = simulator.parse_time_params(dict(start_time=start_time, Ntimes=Ntimes, end_time=end_time))
    for k in master_tdict:
        assert np.all(np.isclose(master_tdict[k], tdict[k]))
    # case iv
    tdict = simulator.parse_time_params(dict(start_time=start_time, end_time=end_time, time_cadence=time_cadence))
    for k in master_tdict:
        assert np.all(np.isclose(master_tdict[k], tdict[k]))

    # test that if time_array is present, it supercedes!
    _tarray = np.linspace(2458201, 2458201.10, 10, endpoint=False)
    tdict = simulator.parse_time_params(dict(start_time=start_time, end_time=end_time, time_cadence=time_cadence, time_array=_tarray))
    assert np.all(np.isclose(_tarray, tdict['time_array']))

    # test Ntimes = 1 exception
    simtest.assert_raises_message(ValueError, 'time_cadence must be specified if Ntimes == 1',
                                  simulator.parse_time_params, dict(time_array=np.array([100.0])))

    # test evenly divisible exception
    simtest.assert_raises_message(ValueError, 'end_time - start_time must be evenly divisible by time_cadence',
                                  simulator.parse_time_params, dict(start_time=100, end_time=101, time_cadence=0.33))

    # test improper combination KeyError
    simtest.assert_raises_message(KeyError, "Couldn't find any proper combination of keys in time_params.",
                                  simulator.parse_time_params, dict(start_time=100.0))


def test_redundant_setup():
    # Test selecting redundant baselines.

    redtol = 0.5    # m

    start_time = 2458101.0
    time_cadence = 100.0
    Ntimes = 10

    start_freq = 1.0
    channel_width = 1.0
    Nfreqs = 10

    time_array = np.linspace(start_time, start_time + Ntimes * time_cadence, Ntimes, endpoint=False)
    freq_array = np.linspace(start_freq, start_freq + Nfreqs * channel_width, Nfreqs, endpoint=False)

    telescope_location = (-30.72152777777791, 21.428305555555557, 1073.0000000093132)

    uvd = simulator.setup_uvdata(array_layout=os.path.join(DATA_PATH, "perfect_hex37_14.6m.csv"),
                                 telescope_location=telescope_location, telescope_name='hera',
                                 freq_array=freq_array, time_array=time_array, redundancy=redtol)

    uvd = simulator.complete_uvdata(uvd)

    uvd0 = simulator.setup_uvdata(array_layout=os.path.join(DATA_PATH, "perfect_hex37_14.6m.csv"),
                                  telescope_location=telescope_location, telescope_name='hera',
                                  freq_array=freq_array, time_array=time_array, redundancy=redtol, no_autos=False)

    uvd0 = simulator.complete_uvdata(uvd0)
    assert uvd0.Nbls == 62
    assert uvd.Nbls == 61


def test_freq_time_params():
    sky = sky_model.SkyModel()
    sky.read_hdf5(os.path.join(DATA_PATH, 'gsm_nside32.hdf5'))
    freqs = sky.freqs
    times = np.linspace(2458570, 2458570 + 0.5, 239)
    time_dict = utils.time_array_to_params(times)
    freq_dict = utils.freq_array_to_params(freqs)
    ftest = simulator.parse_frequency_params(freq_dict)
    ttest = simulator.parse_time_params(time_dict)
    assert np.allclose(ftest['freq_array'], freqs)
    assert np.allclose(ttest['time_array'], times)
