# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

from __future__ import absolute_import, division, print_function

import numpy as np

layout_csv_name = 'imaging_layout.csv'
Nants = 80
freq = 100e6
c = 3e8
Nside = 128

res = np.sqrt(4 * np.pi / (12 * Nside**2))
lam = c / freq
maxbl = 1.22 * (lam / res)      # Resolution ~ 1.22 * lambda/maxbl.

core_width = maxbl/2.
sigma = core_width/2.355

# Want to underresolve pixels (treating them as point sources)
minbl = 5  # m
E, N = np.random.normal(0, sigma, (2, Nants))
U = np.zeros(Nants)

enu = np.vstack((E, N, U)).T
antnames = ["ant{}".format(i) for i in range(Nants)]
antnums = range(Nants)
col_width = max([len(name) for name in antnames])
header = ("{:" + str(col_width) + "} {:8} {:8} {:10} {:10} {:10}\n").format("Name", "Number", "BeamID", "E", "N", "U")
beam_ids = np.zeros(Nants).astype(int)

with open(layout_csv_name, 'w') as lfile:
    lfile.write(header + '\n')
    for i in range(Nants):
        e, n, u = enu[i]
        beam_id = beam_ids[i]
        name = antnames[i]
        num = antnums[i]
        line = ("{:" + str(col_width) + "} {:8d} {:8d} {:10.4f} {:10.4f} {:10.4f}\n").format(name, num, beam_id, e, n, u)
        lfile.write(line)

print("Layout file written: " + layout_csv_name)
