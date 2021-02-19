# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

from __future__ import absolute_import, division, print_function

import numpy as np
import pylab as pl
import healpy as hp

from healvis.sky_model import SkyModel


# -----------------------
# Makes a healpix map with the following properties:
#   - Nonzero pixels are equally-spaced.
#   - Values are constant at each latitude
#   - The value of each nonzero pixel is an integer indexing the latitude.
# -----------------------

nside0 = 12
nside1 = 128
freqs = np.array([100e6])

dpix_spacing = (nside1 / nside0) * np.degrees(hp.nside2resol(nside1))

ra, dec = hp.pix2ang(nside0, np.arange(12 * nside0 ** 2), lonlat=True)
udecs = np.unique(dec)
map0 = np.zeros(12 * nside0 ** 2)
for di, d in enumerate(udecs):
    map0[dec == d] = di

map1 = np.zeros(12 * nside1 ** 2)

inds0 = hp.ang2pix(nside1, ra, dec, lonlat=True)

map1[inds0] = map0
map1 = map1[:, np.newaxis]

sky = SkyModel(freqs=freqs, Nside=nside1, data=map1)

sky.write_hdf5("healvis/data/imaging_test_map.hdf5", clobber=True)
