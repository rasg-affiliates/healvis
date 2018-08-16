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


# Observatory
latitude  = -30.7215277777
longitude =  21.4283055554
fov = 20  #Deg
ant1_enu = np.array([0, 0, 0])
ant2_enu = np.array([0.0, 14.6, 0])
bl = visibility.baseline(ant1_enu, ant2_enu)

# Time
Ntimes = 100
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


obs = visibility.observatory(latitude, longitude, array=[bl], freqs=freqs)
obs.set_fov(fov)
obs.set_beam('gaussian', sigma=2.0)

obs.set_pointings(time_arr)
visibs = obs.make_visibilities(shell0)

cov = np.cov(visibs)
#import IPython; IPython.embed()
print np.diff(time_arr)[0]*60*24
pl.imshow(np.real(cov))
pl.show()
