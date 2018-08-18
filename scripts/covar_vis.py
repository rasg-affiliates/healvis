#!/bin/env python

"""
    Compare, for a single baseline, visibilities:
        >> Covariance matrices of visibs with different time steps and same beam width
        >> Covariance matrices of visibs with different beam widths and same time steps
        >> Covariance matrices of visibs from different resolution shells
"""

import numpy as np
from eorsky import visibility
import pylab as pl
from scipy.stats import binned_statistic

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
fov = 20  #Deg
ant1_enu = np.array([0, 0, 0])
ant2_enu = np.array([0.0, 14.6, 0])
bl = visibility.baseline(ant1_enu, ant2_enu)

# Time
Ntimes = 1000
t0 = 2451545.0      #Start at J2000 epoch
dt_min_short = 30.0/60.
dt_min_long = 10.0
dt_days_short = dt_min_short * 1/60. * 1/24. 
dt_days_long = dt_min_long * 1/60. * 1/24.
time_arr_short = np.arange(Ntimes) * dt_days_short + t0
time_arr_long = np.arange(Ntimes) * dt_days_long + t0

time_arr = time_arr_short

# Frequency
freqs  = np.linspace(1e8, 1.5e8, 100)  #Hz
Nfreqs = freqs.size

# Shells
Nside = 128
Npix = 12*Nside**2
sig = 3.0
shell0 = np.random.normal(0.0, sig, (Npix, Nfreqs))

#Make observatories
obs = visibility.observatory(latitude, longitude, array=[bl], freqs=freqs)
obs.set_fov(fov)
obs.set_beam('gaussian', sigma=2.0)

obs.set_pointings(time_arr)
visibs0 = obs.make_visibilities(shell0)
obs.set_beam('gaussian', sigma=1.0)
visibs1 = obs.make_visibilities(shell0)
obs.set_beam('gaussian', sigma=0.5)
visibs2 = obs.make_visibilities(shell0)
#import IPython; IPython.embed()

covar = np.corrcoef(visibs0)
cov0, lags = bin_covariance(covar, time_arr * 60*24, Nbins=50)
covar = np.corrcoef(visibs1)
cov1, lags = bin_covariance(covar, time_arr * 60*24, Nbins=50)
covar = np.corrcoef(visibs2)
cov2, lags = bin_covariance(covar, time_arr * 60*24, Nbins=50)

#covar = np.cov(visibs)
#print np.diff(time_arr)[0]*60*24

#cov, lags = bin_covariance(covar, time_arr * 60*24, Nbins=50)
pl.plot(lags, cov0,label= '2deg')
pl.plot(lags, cov1,label= '1deg')
pl.plot(lags, cov2,label= '0.5deg')
pl.legend()
pl.show()
#pl.imshow(np.real(covar))
#pl.show()
