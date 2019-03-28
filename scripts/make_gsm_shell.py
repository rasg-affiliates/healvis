# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

from __future__ import absolute_import, division, print_function


import numpy as np
import argparse
import yaml
import healpy as hp
import os
import sys

try:
    import pygsm
except ImportError:
    raise ImportError("pygsm package not found. This is required to use {}".format(os.path.basename(__file__)))
import pyuvsim

from healvis import sky_model

# -----------------------
# Generate a SkyModel object
# of the Global Sky Model (GSM), for a set of frequencies
# specified in a configuration obsparam file.
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument(dest='param', help='obsparam yaml file')
parser.add_argument('--nside', dest='nside', help='Final Nside to use', type=int, default=128)
args = parser.parse_args()

assert args.nside <= 512

param_file = args.param

with open(param_file, 'r') as yfile:
    param_dict = yaml.safe_load(yfile)

param_dict['config_path'] = '.'

tele_dict, beam_list, beam_dict = pyuvsim.simsetup.parse_telescope_params(param_dict['telescope'], param_dict['config_path'])
freq_dict = pyuvsim.simsetup.parse_frequency_params(param_dict['freq'])

freq_array = freq_dict['freq_array'][0]

sky = sky_model.SkyModel(freq_array=freq_array)
sky.Nside = args.nside

data = sky_model.gsm_shell(sky.Nside, freq_array)
sky.set_data(data)

sky.write_hdf5('gsm_{:.2f}-{:.2f}MHz_nside{}.hdf5'.format(freq_array[0] / 1e6, freq_array[-1] / 1e6, args.nside))
