import numpy as np
import argparse
import yaml
import healpy as hp

import pygsm
import pyuvsim

from healvis import skymodel


"""
For a parameter file, generate a skymodel 
"""

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

sky = skymodel.skymodel(freq_array=freq_array)
sky.Nside = args.nside

maps = pygsm.GlobalSkyModel(freq_unit='Hz', basemap='haslam').generate(freq_array)

sky.Npix = sky.Nside**2 * 12
for fi, f in enumerate(freq_array):
    maps[fi,:sky.Npix] = hp.ud_grade(maps[fi], args.nside)

maps = maps[:,:sky.Npix]

sky.set_data(maps.T)

sky.write_hdf5('gsm_{:.2f}-{:.2f}MHz_nside{}.hdf5'.format(freq_array[0]/1e6,freq_array[-1]/1e6, args.nside))
