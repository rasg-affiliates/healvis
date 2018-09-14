#!/bin/env python

"""
    Compare, for a single baseline, visibilities:
        >> Covariance matrices of visibs with different time steps and same beam width
        >> Covariance matrices of visibs with different beam widths and same time steps
        >> Covariance matrices of visibs from different resolution shells
"""

import numpy as np
import  matplotlib.backends.backend_pdf as pdf
from eorsky import visibility
import pylab as pl
from scipy.stats import binned_statistic
from mpi4py import MPI

comm = MPI.COMM_WORLD
Npus = comm.Get_size()
rank = comm.Get_rank()
if rank == 0: print Npus
def bin_covariance(cov, time_arr, Nbins=300):
    Ntimes = time_arr.size
#    inds = np.arange(Ntimes)
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
fov = 15  #Deg
ant1_enu = np.array([0, 0, 0])
ant2_enu = np.array([0.0, 300.6, 0])
bl = visibility.baseline(ant1_enu, ant2_enu)

# Time
#Ntimes = 500
t0 = 2451545.0      #Start at J2000 epoch
#dt_min_short = 30.0/60.
#dt_min_long = 10.0
#dt_days_short = dt_min_short * 1/60. * 1/24. 
#dt_days_long = dt_min_long * 1/60. * 1/24.
#time_arr_short = np.arange(Ntimes) * dt_days_short + t0
#time_arr_long = np.arange(Ntimes) * dt_days_long + t0

#Ntimes = 7854  # 24 hours in 11 sec chunks
#Ntimes = 1000
#Ntimes = 500
#time_arr = np.linspace(t0, t0 + Ntimes/float(3600. * 24 / 11.), Ntimes)

Ntimes=1000
dt_days_long = 0.5/60. * 1/60. * 1/24.   #For testing, 30 sec
time_arr = np.arange(Ntimes) * dt_days_long + t0

stride = int(np.ceil(Ntimes / float(Npus)))
if (rank+1)*stride >=Ntimes:
    time_inds = np.arange(rank * stride, Ntimes)
else:
    time_inds = np.arange(rank * stride, (rank + 1) * stride)
local_time_arr = time_arr[time_inds]
if Npus==1:
    local_time_arr = time_arr
# Frequency
freqs  = np.linspace(1e8, 1.5e8, 100)  #Hz
Nfreqs = freqs.size

# Shells
Nside = 128
Npix = 12*Nside**2
sig = 3.0
shell0 = np.random.normal(0.0, sig, (Npix, Nfreqs))

#Make observatories
visibs = []
#fwhms = [5.0, 20.0, 40.0]
fwhms = [0.5, 5.0, 10.0]
sigmas = [ f/2.355 for f in fwhms]
obs = visibility.observatory(latitude, longitude, array=[bl], freqs=freqs)
obs.set_fov(fov)
obs.set_pointings(local_time_arr)

for s in sigmas:
    if rank == 0 : print(s)
    obs.set_beam('gaussian', sigma=s)
    visibs.append(obs.make_visibilities(shell0))
if rank == 0: vis_all = []
for vis in visibs:
    vis_root = comm.gather(vis, root=0)
    if rank == 0 : vis_all.append(vis_root)
if rank == 0:

    Nbins = 300
    covs = []
    for i, vis in enumerate(vis_all):
        if Npus > 0: vis = np.concatenate(vis)
        covar = np.corrcoef(vis)
        cov, lags = bin_covariance(covar, time_arr * 60 * 24, Nbins=Nbins)
        covs.append(cov)
        pl.plot(lags, cov, label= str(fwhms[i])+"deg")
    covr, lags = bin_covariance(covar, time_arr * 60*24, Nbins=Nbins)
    
    #covar = np.cov(visibs)
    #print np.diff(time_arr)[0]*60*24
    
    pl.plot(lags, covr, label= 'Random vis')
    pl.xlabel("Lag (minutes)")
    pl.ylabel("Binned Correlation Coefficient")
    pl.legend()
#    with pdf.PdfPages("covariance_comparison.pdf") as pdffile:
#        pdffile.savefig()
    #pl.imshow(np.real(covar))
    pl.show()
else:
    del visibs
    del vis_root
    del shell0
