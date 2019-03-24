from __future__ import absolute_import, division, print_function

from healvis import sky_model, simulator
from astropy.cosmology import Planck15
import nose.tools as nt
import numpy as np
import os
import healpy as hp
from healvis.data import DATA_PATH
import yaml


def test_run_simulation():
    # load parameter file from test directory
    param_file = os.path.join(DATA_PATH, "configs/obsparam_test.yaml")
    with open(param_file, 'r') as _f:
        param_dict = yaml.safe_load(_f)

    # edit
    param_dict['telescope']['config_dir'] = os.path.join(DATA_PATH, 'configs')
    param_dict['beam']['beam_type'] = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    param_dict['skyparam']['sky_type'] = os.path.join(DATA_PATH, "gsm_nside32.hdf5")

    # run simulation
    simulator.run_simulation(param_dict, add_to_history='foo')
