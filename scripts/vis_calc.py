
"""
Calculate visibilities for:
    > Gaussian beam
    > Single baseline
    > Sky from file or generated on the fly

and save to MIRIAD file
"""

import numpy as np
from eorsky import visibility
import pylab as pl
from scipy.stats import binned_statistic
import os, sys
from pyuvsim.simsetup import check_file_exists_and_increment
from pyuvdata import UVData
from pyuvdata import utils as uvutils

ofilename = 'eorsky_gauss_sim.uv'

try:
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
except KeyError:
    task_id = None

# Observatory
latitude  = -30.7215277777
longitude =  21.4283055554
altitude = 1073.
fov = 10  #Deg
ant1_enu = np.array([0, 0, 0])
ant2_enu = np.array([0.0, 14.6, 0])
bl = visibility.baseline(ant1_enu, ant2_enu)

# Time

t0 = 2451545.0      #Start at J2000 epoch
#Ntimes = 7854  # 24 hours in 11 sec chunks
Nfreqs = 100
Ntimes = 2000
#Ntimes = 500
time_arr = np.linspace(t0, t0 + Ntimes/float(3600. * 24 / 11.), Ntimes)

# Frequency
freqs  = np.linspace(1e8, 1.5e8, Nfreqs)  #Hz
Nfreqs = freqs.size

# Shells
Nside = 64
Npix = 12*Nside**2
sig = 3.0
Nskies = 1
shell0 = np.random.normal(0.0, sig, (Nskies, Npix, Nfreqs))

#Make observatories
visibs = []
#fwhms = [2.5, 5.0, 10.0, 20.0, 25.0, 30.0]
#fwhms = [35, 40, 45.0, 50.0, 55.0, 60.0]
fwhms = [35.0]
sigmas = [ f/2.355 for f in fwhms]
obs = visibility.observatory(latitude, longitude, array=[bl], freqs=freqs)
obs.set_fov(fov)
obs.set_pointings(time_arr)
sigs_used = []
if task_id >= len(sigmas):
    sys.exit()
for i,s in enumerate(sigmas):
    if task_id is not None:
        if not i == task_id:
            continue
    print s
    sigs_used.append(s)
    obs.set_beam('gaussian', sigma=s)
    visibs.append(obs.make_visibilities(shell0))

import IPython; IPython.embed()
import sys; sys.exit()

uv = UVData()
data_arr = visibs[0][:,0,:]  # (Nblts, Nskies, Nfreqs)
data_arr = data_arr[:,np.newaxis,:,np.newaxis]  # (Nblts, Nspws, Nfreqs, Npol)

# Convert from K str to Jy:

def jy2Tstr(f):#, bm):
    '''Return [mK sr] / [Jy] vs. frequency (in Hz)
        f = frequencies (Hz!)
    '''
    c_cmps = 2.99792458e10   # cm/s
    k_boltz = 1.380658e-16   # erg/K
    lam = c_cmps / f   #cm
    bm = 1.0 # steradian
    return 1e-23 * lam**2 / (2 * k_boltz * bm) * 1e3

conv_factor = jy2Tstr(freqs)

data_arr[:,0,:,0] /= conv_factor

# Get beam_sq_int

za, az = obs.calc_azza(Nside, obs.pointing_centers[0])
beam_sq_int = np.sum(obs.beam.beam_val(az, za)**2)
beam_sq_int = np.ones(Nfreqs) * beam_sq_int

uv.Nbls = 1
uv.Ntimes = Ntimes
uv.spw_array = [0]
uv.Nfreqs = Nfreqs
uv.freq_array = freqs[np.newaxis, :]
uv.Nblts = uv.Ntimes * uv.Nbls
uv.data_array = data_arr
uv.ant_1_array = np.zeros(uv.Nblts, dtype=int)
uv.ant_2_array = np.ones(uv.Nblts, dtype=int)
uv.baseline_array = uv.antnums_to_baseline(uv.ant_1_array, uv.ant_2_array)
uv.time_array = time_arr
uv.Npols = 1
uv.polarization_array=np.array([-5])
uv.Nants_telescope = 2
uv.Nants_data = 2
uv.antenna_positions = uvutils.ECEF_from_ENU(np.stack([ant1_enu, ant2_enu]), latitude, longitude, altitude)
uv.flag_array = np.zeros_like(uv.data_array).astype(bool)
uv.nsample_array = np.ones_like(uv.data_array).astype(float)
uv.Nspws = 1
uv.antenna_numbers = np.array([0,1])
uv.antenna_names = ['ant0', 'ant1']
#uv.channel_width = np.ones(uv.Nblts) * np.diff(freqs)[0]
uv.channel_width = np.diff(freqs)[0]
uv.integration_time = np.ones(uv.Nblts) * np.diff(time_arr)[0] * 24 * 3600.  # Seconds
uv.uvw_array = np.tile(ant1_enu - ant2_enu, uv.Nblts).reshape(uv.Nblts, 3)
uv.history = 'Eorsky simulated'
uv.set_drift()
uv.telescope_name = 'Eorsky gaussian'
uv.instrument = 'simulator'
uv.object_name = 'zenith'
uv.vis_units = 'k str'
uv.telescope_location_lat_lon_alt_degrees = (latitude, longitude, altitude)
uv.set_lsts_from_time_array()
uv.extra_keywords = {'bsq_int': beam_sq_int[0], 'skysig': sig, 'bm_fwhm' : fwhms[0], 'nside': Nside}


uv.check()

uv.write_miriad(ofilename, clobber=True)
