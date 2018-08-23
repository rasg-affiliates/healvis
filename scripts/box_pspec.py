## Make a box with a specified power spectrum.


###
# Amp = sqrt( Pk * volume / 2.)
###

import numpy as np
from eorsky import box_dft_pspec
import sys

pkfile = sys.argv[1]
## Columns of file are k, P(k)
## Interpolate these to the 3D grid

L = 300 #Mpc
N = 256

M = N/2 # k-space dimension

dx = L/float(N)

kbins = np.fft.fftfreq(N, d=dx)
kbins = kbins[:M]  #Only positive k's

dat = np.loadtxt(pkfile)
k0, pk0 = dat[:,0], dat[:,1]

pk1d = np.interp(kbins, k0, pk0)

#box = np.random.normal(0.0, 1.0, (N,N,N))
#_box = np.fft.fftn(box)
#_box *= 2.0

kx, ky, kz = np.meshgrid(kbins, kbins, kbins)
kx, ky, kz = [k.flatten() for k in (kx, ky, kz)]

kmags = np.sqrt(kx**2 + ky**2 + kz**2)
kbins = np.unique(kmags)
kbins = np.linspace(min(kbins), max(kbins), M)

kmags = kmags.reshape(M,M,M)

#Get indices of kbin values for the kmags array.
pk_3d = np.zeros_like(kmags)
kbin3d = np.zeros_like(kmags)
counts = np.zeros_like(kmags)
for ki, km in enumerate(kbins[:-1]):
    inds = np.where((kmags >= kbins[ki])&(kmags < kbins[ki+1]))
    np.add.at(pk_3d, inds, pk1d[ki])
    np.add.at(kbin3d, inds, km)
    np.add.at(counts, inds, 1)

nz = np.where(counts != 0)
pk_3d[nz] /= counts[nz]

#pk_3d = kmags.reshape(M,M,M)/np.max(kmags)

#pk_3d = np.ones((M,M,M)) * 3.0

_box = (np.random.normal(0.0, 1.0, (N,N,N))
        + (1j)*np.random.normal(0.0, 1.0, (N,N,N))) / np.sqrt(2.)

print np.var(_box)

dV = (L/float(N))**3
amp = np.sqrt(pk_3d*float(L**3))
_box[:M,:M,:M] *= amp
_box[M:,M:,M:] = (-1)*((_box[M-1::-1,M-1::-1,M-1::-1])).conj()  #Reversal is done before slicing.
#_box[0,0,0] = 0   #DC mode = 0

## Box is now real-valued
box = np.fft.ifftn(_box)

knew, pknew = box_dft_pspec(box, L, cosmo=True)

import pylab as pl
fig = pl.figure()
ax = fig.add_subplot(111)

ax.plot(k0, pk0, label='Input pspec')
ax.plot(knew, pknew, label='Output pspec')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
pl.show()



##This works, but I'm not sure I understand the scaling... there's an extra factor of 2 in the amplitude


