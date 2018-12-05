#!/bin/env python

#SBATCH -J eorsky
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=70G
#SBATCH -A jpober-condo
#SBATCH --qos=jpober-condo

"""
Calculate visibilities for:
    > Gaussian beam
    > Single baseline
    > Sky from file or generated on the fly

and save to MIRIAD file
"""

import numpy as np
from eorsky import visibility, utils
import pylab as pl
from scipy.stats import binned_statistic
import os, sys, yaml
import pyuvsim
from pyuvdata import UVData
from pyuvdata import utils as uvutils
from eorsky import comoving_voxel_volume
from itertools import izip
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(dest='param', help='obsparam yaml file')

args = parser.parse_args()

param_file = args.param

with open(param_file, 'r') as yfile:
    param_dict = yaml.safe_load(yfile)

param_dict['config_path'] = '.'
uv_obj, beam_list, beam_dict, beam_ids = pyuvsim.simsetup.initialize_uvdata_from_params(param_dict)

# Reduce uv_obj to only selected parts (reduce metadata)
ant_nums = np.unique(uv_obj.ant_1_array.tolist() +  uv_obj.ant_2_array.tolist())
ant_pos = np.zeros((uv_obj.Nants_data, 3), dtype=float)
ant_num_full = uv_obj.antenna_numbers
ant_names = []
ant_num_ord = []
for i,anum in enumerate(ant_nums):
    inds = np.where(ant_num_full == anum)
    ant_pos[i] = uv_obj.antenna_positions[inds]
    ant_names.append(uv_obj.antenna_names[inds][0])
    ant_num_ord.append(ant_num_full[inds][0])

uv_obj.antenna_names = ant_names
uv_obj.antenna_numbers = np.array(ant_num_ord, dtype=int)
uv_obj.antenna_positions = ant_pos
uv_obj.Nants_telescope = uv_obj.Nants_data

filing_params = param_dict['filing']

beam_list = [pyuvsim.simsetup.beam_string_to_object(b) for b in beam_list]

# ---------------------------
# Extra parameters required for eorsky
# ---------------------------

fov = param_dict['fov']  #Deg
Nside = param_dict['Nside']
Nskies = 1 if 'Nskies' not in param_dict else int(param_dict['Nskies'])
sky_sigma = param_dict['sky_sigma']
try:
    beam_select = int(param_dict['beam_select'])
except KeyError:
    beam_select=0


beam = beam_list[beam_select]
beam_type = beam.type
if beam_type == 'gaussian':
    beam_attr = {'sigma' : np.degrees(beam.sigma)}
elif beam_type == 'airy':
    beam_attr = {'diameter': beam.diameter}
elif beam_type == 'uniform':
    beam_attr = {}
else:
    raise ValueError("UVBeam not currently supported")


# ---------------------------
# Parallelization
# ---------------------------
if 'SLURM_CPUS_PER_TASK' in os.environ:
    Nprocs = int(os.environ['SLURM_CPUS_PER_TASK'])# * int(os.environ['SLURM_NTASKS'])
elif 'Nprocs' in param_dict:
    Nprocs = int(param_dict['Nprocs'])
else:
    Nprocs = 1

sjob_id = None
if 'SLURM_JOB_ID' in os.environ:
    sjob_id = os.environ['SLURM_JOB_ID']

print("Nprocs: ", Nprocs)
sys.stdout.flush()
# ---------------------------
# Observatory
# ---------------------------
lat, lon, alt = uv_obj.telescope_location_lat_lon_alt_degrees
enu, anum = uv_obj.get_ENU_antpos(center=False)

Ntimes = uv_obj.Ntimes
Nbls = uv_obj.Nbls

array = []
bl_array =  []
#uv_obj.Nants_telescope = uv_obj.Nants_data   # Select messes this up
for a1, a2 in izip(uv_obj.ant_1_array[:Nbls], uv_obj.ant_2_array[:Nbls]):
    i1 = np.where(anum == a1)
    i2 = np.where(anum == a2)
    array.append(visibility.baseline(enu[i1], enu[i2]))
    bl_array.append(uv_obj.antnums_to_baseline(a1, a2))
bl_array = np.array(bl_array)
freqs = uv_obj.freq_array[0]        #Hz
obs = visibility.observatory(lat, lon, array=array, freqs=freqs)
obs.set_fov(fov)

print("Observatory built.")
sys.stdout.flush()


# ---------------------------
# Pointings
# ---------------------------
time_arr = np.unique(uv_obj.time_array)
obs.set_pointings(time_arr)

print("Pointings set.")
sys.stdout.flush()

# ---------------------------
# Primary beam
# ---------------------------
obs.set_beam(beam_type, **beam_attr)


# ---------------------------
# Shells
# ---------------------------
Npix = 12*Nside**2
sig = sky_sigma
Nfreqs = uv_obj.Nfreqs
shell0 = utils.mparray((Nskies, Npix, Nfreqs), dtype=float)
dnu = np.diff(freqs)[0]/1e6
om = 4*np.pi/float(Npix)
Zs = 1420e6/freqs - 1
dV0 = comoving_voxel_volume(Zs[Nfreqs/2], dnu, om)
print("Making skies")
sys.stdout.flush()
for fi in range(Nfreqs):
    dV = comoving_voxel_volume(Zs[fi], dnu, om) 
    s = sig  * np.sqrt(dV0/dV)
    shell0[:,:,fi] = np.random.normal(0.0, s, (Nskies, Npix))


# ---------------------------
# Beam^2 integral
# ---------------------------
za, az = obs.calc_azza(Nside, obs.pointing_centers[0])
beam_sq_int = np.sum(obs.beam.beam_val(az, za)**2)
beam_sq_int = np.ones(Nfreqs) * beam_sq_int * om 

# ---------------------------
# Run simulation
# ---------------------------

print("Running simulation")
sys.stdout.flush()
visibs, time_array, baseline_inds = obs.make_visibilities(shell0, Nprocs=Nprocs)

uv_obj.time_array = time_array
uv_obj.set_lsts_from_time_array()
uv_obj.baseline_array = bl_array[baseline_inds]
uv_obj.ant_1_array, uv_obj.ant_2_array = uv_obj.baseline_to_antnums(uv_obj.baseline_array)

uv_obj.spw_array = np.array([0])
uv_obj.Npols = 1
uv_obj.polarization_array=np.array([1])
uv_obj.Nspws = 1
uv_obj.set_uvws_from_antenna_positions()
uv_obj.channel_width = np.diff(freqs)[0]
uv_obj.integration_time = np.ones(uv_obj.Nblts) * np.diff(time_arr)[0] * 24 * 3600.  # Seconds
uv_obj.history = 'eorsky'
uv_obj.set_drift()
uv_obj.telescope_name = 'eorsky'
uv_obj.instrument = 'simulator'
uv_obj.object_name = 'zenith'
uv_obj.vis_units = 'Jy'

if sjob_id is None:
    sjob_id = ''

if beam_type == 'gaussian':
    fwhm = beam_attr['sigma'] * 2.355
    uv_obj.extra_keywords = {'bsq_int': beam_sq_int[0], 'skysig': sky_sigma, 'bm_fwhm': fwhm, 'nside': Nside, 'slurm_id': sjob_id}
else:
    uv_obj.extra_keywords = {'bsq_int': beam_sq_int[0], 'skysig': sky_sigma, 'nside': Nside, 'slurm_id': sjob_id}


for sky_i in range(Nskies):
    if Nskies > 1:
        filing_params['outfile_suffix'] = '{}sky_uv'.format(sky_i)
    else:
        filing_params['outfile_suffix'] = 'uv'
    filing_params['outfile_prefix'] = \
             'eorsky_{:.2f}hours_Nside{}'.format(Ntimes/(3600./11.0), Nside)

    data_arr = visibs[:,sky_i,:]  # (Nblts, Nskies, Nfreqs)
    data_arr = data_arr[:,np.newaxis,:,np.newaxis]  # (Nblts, Nspws, Nfreqs, Npol)
    uv_obj.data_array = data_arr

    uv_obj.flag_array = np.zeros(uv_obj.data_array.shape).astype(bool)
    uv_obj.nsample_array = np.ones(uv_obj.data_array.shape).astype(float)

    uv_obj.check()
    pyuvsim.simsetup.write_uvfits(uv_obj, filing_params, miriad=True)
