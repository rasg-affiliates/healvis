# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

import numpy as np
import healvis as hv
import healpy as hp
from pyuvsim import AnalyticBeam
    
def test_external_beam():
    """
    None of this should fail.
    """

    obs_latitude = -30.7215277777
    obs_longitude = 21.4283055554
    obs_height = 1073.0
    
    # Create baselines
    
    # Random antenna locations
    number = 3
    x = np.random.random(number)*40
    y = np.random.random(number)*40
    z = np.random.random(number)*1
    ants = {}
    for i in range(number):
        ants[i] = ( x[i], y[i], z[i] )
    
    baselines = []
    for i in range(len(ants)):
        for j in range(i+1, len(ants)):
            bl = hv.observatory.Baseline(ants[i], ants[j], i, j)
            baselines.append(bl)
    
    # Set frequency axis
    freqs = np.linspace(100e6, 150e6, 2, endpoint=False)
    Nfreqs = len(freqs)
    
    # Set times
    times = np.linspace(2458000.1, 2458000.6, 2)
    Ntimes = len(times)
    
    # Create the observatory
    fov = 360  # Deg
    obs = hv.observatory.Observatory(obs_latitude, obs_longitude, obs_height, array=baselines, freqs=freqs)
    obs.set_pointings(times)
    obs.set_fov(fov)
    
    # Create external beam
    one_beam = AnalyticBeam("gaussian", sigma=0.2)
    beam = [ one_beam for i in range(len(ants)) ]
    
    # set beam
    obs.set_beam(beam)
    
    # create a noise-like sky
    np.random.seed(0)
    Nside = 64
    Npix = hp.nside2npix(Nside)
    eor = hv.sky_model.construct_skymodel("flat_spec", freqs=freqs, Nside=Nside, ref_chan=0, sigma=1e-3)
    
    # Compute Visibilities for eor
    eor_vis, times, bls = obs.make_visibilities(eor)
