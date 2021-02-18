import numpy as np
from healvis import observatory
import pylab as pl


latitude = -30.7215277777
longitude = 21.4283055554
fov = 2  # Deg
ant1_enu = np.array([0, 0, 0])
ant2_enu = np.array([0.0, 14.6, 0])
bl = observatory.Baseline(ant1_enu, ant2_enu)

# Time
Ntimes = 100
t0 = 2451545.0  # Start at J2000 epoch
dt_min_short = 30.0 / 60.0
dt_min_long = 10.0
dt_days_short = dt_min_short * 1 / 60.0 * 1 / 24.0
dt_days_long = dt_min_long * 1 / 60.0 * 1 / 24.0
time_arr_short = np.arange(Ntimes) * dt_days_short + t0
time_arr_long = np.arange(Ntimes) * dt_days_long + t0

time_arr = time_arr_short
freqs = [1e8]
obs = observatory.Observatory(latitude, longitude, array=[bl], freqs=freqs)
obs.set_fov(fov)

obs.set_pointings(time_arr)

Nside = 128

pixels = obs.get_observed_region(Nside)
