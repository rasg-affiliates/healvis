#!/bin/env python

# SBATCH -J healvis
# SBATCH -t 12:00:00
# SBATCH -n 5
# SBATCH --mem=30G


"""
Calculate visibilities for:
    > Gaussian beam
    > Single baseline
    > Sky from file

and save to MIRIAD file

Requires eorsky
"""

import numpy as np
from healvis import observatory
import os
from pyuvdata import UVData
from pyuvdata import utils as uvutils
from eorsky import eorsky
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-w",
    "--beam_width",
    dest="fwhm",
    help="Primary gaussian beam fwhm, in degrees",
    default=50,
    type=float,
)
parser.add_argument(
    "--fov", dest="fov", help="Field of view, in degrees", default=100, type=float
)
parser.add_argument(
    "-t",
    "--Ntimes",
    dest="Ntimes",
    help="Number of 11sec integration times, default is 24 hours' worth",
    default=7854,
    type=int,
)
parser.add_argument(
    "-b",
    "--baseline_length",
    dest="bllen",
    help="Baseline length in meters",
    default=14.6,
    type=float,
)
parser.add_argument(
    "shell",
    help="Path to hdf5 shell file",
    default="pk_z7.5_euler-calsky_2Gpc_512.hdf5",
)

args = parser.parse_args()

# Observatory
latitude = -30.7215277777
longitude = 21.4283055554
altitude = 1073.0
fov = args.fov  # Deg
ant1_enu = np.array([0, 0, 0])
ant2_enu = np.array([0.0, args.bllen, 0])
bl = observatory.Baseline(ant1_enu, ant2_enu)

# Time

t0 = 2451545.0  # Start at J2000 epoch
Ntimes = args.Ntimes
time_arr = np.linspace(t0, t0 + Ntimes / float(3600.0 * 24 / 11.0), Ntimes)

# Shells

ek = eorsky()
ek.read_hdf5(args.shell)
shell0 = ek.shell[np.newaxis, ...]
Nside = ek.Nside
freqs = ek.freqs
Nfreqs = ek.Nfreq
Nskies = 1
Npix = 12 * Nside ** 2

# Make observatories
visibs = []
fwhm = args.fwhm
sigma = fwhm / 2.355
obs = observatory.Observatory(latitude, longitude, array=[bl], freqs=freqs)
obs.set_fov(fov)
obs.set_pointings(time_arr)
obs.set_beam("gaussian", sigma=sigma)
visibs.append(obs.make_visibilities(shell0))
# Visibilities are in Jy

# Get beam_sq_int
za, az = obs.calc_azza(Nside, obs.pointing_centers[0])
beam_sq_int = np.sum(obs.beam.beam_val(az, za) ** 1)
dnu = np.diff(freqs)[0] / 1e6
om = 4 * np.pi / float(Npix)
Zs = 1420e6 / freqs - 1
beam_sq_int = np.ones(Nfreqs) * beam_sq_int * om


uv = UVData()
uv.Nbls = 1
uv.Ntimes = Ntimes
uv.spw_array = [0]
uv.Nfreqs = Nfreqs
uv.freq_array = freqs[np.newaxis, :]
uv.Nblts = uv.Ntimes * uv.Nbls
uv.ant_1_array = np.zeros(uv.Nblts, dtype=int)
uv.ant_2_array = np.ones(uv.Nblts, dtype=int)
uv.baseline_array = uv.antnums_to_baseline(uv.ant_1_array, uv.ant_2_array)
uv.time_array = time_arr
uv.Npols = 1
uv.polarization_array = np.array([1])
uv.Nants_telescope = 2
uv.Nants_data = 2
uv.antenna_positions = uvutils.ECEF_from_ENU(
    np.stack([ant1_enu, ant2_enu]), latitude, longitude, altitude
)
uv.Nspws = 1
uv.antenna_numbers = np.array([0, 1])
uv.antenna_names = ["ant0", "ant1"]
uv.channel_width = np.diff(freqs)[0]
uv.integration_time = np.ones(uv.Nblts) * np.diff(time_arr)[0] * 24 * 3600.0  # Seconds
uv.uvw_array = np.tile(ant1_enu - ant2_enu, uv.Nblts).reshape(uv.Nblts, 3)
uv.history = "Eorsky simulated"
uv.set_drift()
uv.telescope_name = "Eorsky gaussian"
uv.instrument = "simulator"
uv.object_name = "zenith"
uv.vis_units = "k str"
uv.telescope_location_lat_lon_alt_degrees = (latitude, longitude, altitude)
uv.set_lsts_from_time_array()
uv.extra_keywords = {
    "bsq_int": beam_sq_int[0],
    "filename": os.path.basename(args.shell),
    "bm_fwhm": fwhm,
    "nside": Nside,
}

for sky_i in range(Nskies):
    if Nskies > 1:
        ofilename = "healvis_gauss{}d_{:.2f}hours_{}m_{}nside_{}fov_{}sky_uv".format(
            args.fwhm, args.Ntimes / (3600.0 / 11.0), args.bllen, Nside, args.fov, sky_i
        )
    else:
        ofilename = "healvis_gauss{}d_{:.2f}hours_{}m_{}nside_{}fov_uv".format(
            args.fwhm, args.Ntimes / (3600.0 / 11.0), args.bllen, Nside, args.fov
        )
    print("ofilename: ", ofilename)
    data_arr = visibs[0][:, sky_i, :]  # (Nblts, Nskies, Nfreqs)
    data_arr = data_arr[:, np.newaxis, :, np.newaxis]  # (Nblts, Nspws, Nfreqs, Npol)
    uv.data_array = data_arr

    uv.flag_array = np.zeros(uv.data_array.shape).astype(bool)
    uv.nsample_array = np.ones(uv.data_array.shape).astype(float)

    uv.check()
    uv.write_miriad(ofilename, clobber=True)
