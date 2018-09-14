#!/bin/env python

#SBATCH -J eorsky
#SBATCH --mem=10G
#SBATCH -n 8
#SBATCH -t 8:00:00
#SBATCH --array=0-12

"""
    Compare, for a single baseline, visibilities:
        >> Covariance matrices of visibs with different time steps and same beam width
        >> Covariance matrices of visibs with different beam widths and same time steps
        >> Covariance matrices of visibs from different resolution shells
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
from eorsky import visibility
import pylab as pl
from scipy.stats import binned_statistic
import os, sys
from pyuvsim.simsetup import check_file_exists_and_increment

try:
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
except KeyError:
    task_id = None

def bin_covariance(cov, time_arr, Nbins=300):
	Ntimes = time_arr.size
#	inds = np.arange(Ntimes)
	inds = time_arr
	xind, yind = np.meshgrid(inds, inds)
	diff_inds = xind-yind
	means, bins, binnums = binned_statistic(diff_inds.flatten(), np.real(cov).flatten(), bins=Nbins)
	lag_bins = ((bins[1:] + bins[:-1])/2.)
	#pl.plot(bins[1:], means)
	return means, lag_bins


# Observatory
latitude  = -30.7215277777
longitude =  21.4283055554
fov = 180  #Deg
ant1_enu = np.array([0, 0, 0])
ant2_enu = np.array([0.0, 14.6, 0])
bl = visibility.baseline(ant1_enu, ant2_enu)

# Time
#Ntimes = 500
#t0 = 2451545.0      #Start at J2000 epoch
#dt_min_short = 30.0/60.
#dt_min_long = 10.0
#dt_days_short = dt_min_short * 1/60. * 1/24. 
#dt_days_long = dt_min_long * 1/60. * 1/24.
#time_arr_short = np.arange(Ntimes) * dt_days_short + t0
#time_arr_long = np.arange(Ntimes) * dt_days_long + t0
#
#time_arr = time_arr_short

t0 = 2451545.0      #Start at J2000 epoch
#Ntimes = 7854  # 24 hours in 11 sec chunks
Nfreqs = 100
Ntimes = 3000
#Ntimes = 500
time_arr = np.linspace(t0, t0 + Ntimes/float(3600. * 24 / 11.), Ntimes)

# Frequency
freqs  = np.linspace(1e8, 1.5e8, Nfreqs)  #Hz
freqs  = np.ones(Nfreqs)*1.5e8
Nfreqs = freqs.size

# Shells
Nside = 128
Npix = 12*Nside**2
sig = 3.0
shell0 = np.random.normal(0.0, sig, (Npix, Nfreqs))

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

Nbins = 2500
covs = []
for i, vis in enumerate(visibs):
    covar = np.corrcoef(np.abs(vis)**2)
    cov, lags = bin_covariance(covar, time_arr * 60 * 24, Nbins=Nbins)
    covs.append(cov)
#    pl.plot(lags, cov, label= str(fwhms[i])+"deg")
covar = np.corrcoef(np.abs(np.random.normal(0.0, np.var(vis), vis.shape))**2)
covr, lags = bin_covariance(covar, time_arr * 60*24, Nbins=Nbins)
#pl.plot(lags, covr, label= 'Random vis')

#covar = np.cov(visibs)
#print np.diff(time_arr)[0]*60*24

fmt = '{:02d}'
fname = check_file_exists_and_increment('covariances_onefreq_'+fmt.format(task_id)+'.npz', fmt=fmt)
np.savez(fname, covs=covs, sigmas=sigs_used, lags=lags, covar_rand = covr)

#pl.xlabel("Lag (minutes)")
#pl.ylabel("Binned Correlation Coefficient")
#pl.legend()
#pl.savefig('covariance_comparison.png')
# with pdf.PdfPages("covariance_comparison.pdf") as pdffile:
#     pdffile.savefig()
#pl.imshow(np.real(covar))
#pl.show()

#import IPython; IPython.embed()
#pl.imshow(np.real(covar))
#pl.show()
