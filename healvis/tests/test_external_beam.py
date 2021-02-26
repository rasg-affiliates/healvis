# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

import numpy as np
from healvis.observatory import Baseline, Observatory
from healvis.sky_model import construct_skymodel

class ExternalBeam:
    """
    A dummy beam object for testing.

    It returns an array of the right size from interp().
    Demonstrates that any beam with an interp() method
    can be used.
    """

    def interp(self, az_array, za_array, freq_array):
        """
        Evaluate the primary beam at given az, za locations (in radians).
        Parameters
        ----------
        az_array : array_like
            Azimuth values in radians (same length as za_array). The azimuth
            here has the UVBeam convention: North of East(East=0, North=pi/2)

        za_array : array_like
            Zenith angle values in radians (same length as az_array).

        freq_array : array_like
            Frequency values to evaluate at.

        Returns
        -------
        interp_data : array_like
            Array of beam values, shape (Naxes_vec, Nspws, Nfeeds or Npols,
            Nfreqs or freq_array.size if freq_array is passed,
            Npixels/(Naxis1, Naxis2) or az_array.size if az/za_arrays are passed)

        interp_basis_vector : array_like
            Array of interpolated basis vectors (or self.basis_vector_array
            if az/za_arrays are not passed), shape: (Naxes_vec, Ncomponents_vec,
            Npixels/(Naxis1, Naxis2) or az_array.size if az/za_arrays are passed)
        """

        # Empty data array
        interp_data = np.zeros((2, 1, 2, freq_array.size, za_array.size))
    
        values = np.empty((freq_array.size, za_array.size))
        values[:] = np.cos(za_array)

        interp_data[1, 0, 0, :, :] = values
        interp_data[0, 0, 1, :, :] = values
        interp_basis_vector = None
        return interp_data, interp_basis_vector


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
            bl = Baseline(ants[i], ants[j], i, j)
            baselines.append(bl)
    
    # Set frequency axis
    freqs = np.linspace(100e6, 150e6, 2, endpoint=False)
    Nfreqs = len(freqs)
    
    # Set times
    times = np.linspace(2458000.1, 2458000.6, 2)
    Ntimes = len(times)
    
    # Create the observatory
    fov = 360  # Deg
    obs = Observatory(obs_latitude, obs_longitude, obs_height, array=baselines, freqs=freqs)
    obs.set_pointings(times)
    obs.set_fov(fov)
    
    # Create external beam
    one_beam = ExternalBeam()
    beam = [ one_beam for i in range(len(ants)) ]
    
    # set beam
    obs.set_beam(beam)
    
    # create a noise-like sky
    np.random.seed(0)
    Nside = 64
    eor = construct_skymodel("flat_spec", freqs=freqs, Nside=Nside, ref_chan=0, sigma=1e-3)
    
    # Compute Visibilities for eor
    eor_vis, times, bls = obs.make_visibilities(eor)

