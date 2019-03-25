from __future__ import absolute_import, division, print_function

from healvis import sky_model, simulator
from astropy.cosmology import Planck15
import nose.tools as nt
import numpy as np
import os
import healpy as hp
from healvis.data import DATA_PATH
import yaml
import shutil

from pyuvdata import UVData


def test_run_simulation():
    # load parameter file from test directory
    param_file = os.path.join(DATA_PATH, "configs/obsparam_test.yaml")
    with open(param_file, 'r') as _f:
        param_dict = yaml.safe_load(_f)

    # edit
    param_dict['telescope']['config_dir'] = os.path.join(DATA_PATH, 'configs')
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
    nt.assert_equal(uvd.Nbls, 3)
    nt.assert_equal(uvd.Npols, 2)
    nt.assert_true('foo' in uvd.history)  # check add_to_history was propagated
    nt.assert_true("skyparam" in uvd.history)  # check param_dict was written to history

    # erase output directory
    shutil.rmtree(param_dict['filing']['outdir'])

