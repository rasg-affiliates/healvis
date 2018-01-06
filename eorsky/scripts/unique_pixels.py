#!/bin/env python

## Calculate the number of unique pixels per redshift selected from a cartesian cube by CosmoTile
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from astropy import units
from astropy.constants import c
import healpy as hp
from numpy import pi
import sys


#Cosmological parameters

pick = sys.argv[1]
print pick, sys.argv
paper_freqs = np.linspace(100,200,203)
paper_freqs_full = np.linspace(100,200,1024)
mwa_freqs = np.linspace(167.075,197.715,384)


if pick=='PAPER_comp': freqs = paper_freqs
if pick=='PAPER_full': freqs = paper_freqs_full
if pick=='MWA': freqs = mwa_freqs

###Choose one
#freqs = mwa_freqs
#freqs = paper_freqs_full
#freqs = paper_freqs

#Zmin, Zmax = 6.93, 7.01

Zs = 1420/freqs - 1
Zs = Zs[::-1]   #Reverse

#imin = np.argmin(np.abs(Zs - Zmin))
#imax = np.argmin(np.abs(Zs - Zmax))
#Zs = Zs[imin:imax]
#nfreq = imax-imin
nfreq = freqs.size

df = np.diff(freqs)[0]
r_mpc = cosmo.comoving_distance(Zs).to("Mpc").value

### Healpix params
nside=512
npix = hp.nside2npix(nside)
pixsize = hp.nside2pixarea(nside) * (4*pi)/(2*pi)**2   # Approximately go from square radians to steradians

### The input cube parameters
L = 300 # Mpc, side length
N = 256 # Pixels per side
dx = L/float(N)
npix_cart = N**3  #Total pixels

### Measure rms error of pixels location.
#ninds = np.zeros(nfreq)
#for i in range(nfreq):
#	r = r_mpc[i]
#	X,Y,Z = hp.pix2vec(nside, np.arange(npix))
#	XYZ = np.array([X,Y,Z])
#	XYZmod = np.mod(r*XYZ,L)
#	delx,dely,delz = (XYZmod)%dx
#	rms = np.sqrt(np.mean( (delx**2 + dely**2 + delz**2)))
#	print i, rms
#	#l,m,n = (xyz/dx).astype(int)
#	#inds = l + m*(N+1) + n*(N+1)**2
#	#ninds[i] = np.unique(inds).shape[0]/float(npix)
#
## Find the number of unique indices selected from the Cartesian cube by the healpix map at each redshift:
ninds = np.zeros(nfreq)
#total_sq_err = []
all_inds = []
X,Y,Z = hp.pix2vec(nside, np.arange(npix))
XYZ = np.array([X,Y,Z])
for i in range(nfreq):
    print i
    r = r_mpc[i]
    xyz = (r*XYZ)%L
    #delx,dely,delz = (xyz)%dx
    #total_sq_err += (delx**2 + dely**2 + delz**2).tolist()
    l,m,n = np.round(xyz/dx).astype(int)
    inds = l + m*(N+1) + n*(N+1)**2
    #all_inds += inds.tolist()
    ninds[i] = np.unique(inds).shape[0]/float(npix)
    #ninds[i] = np.unique(inds).shape[0]/float(N**3)
#total_unique = np.unique(all_inds)
#print 'Total fraction of unique indices from box:', total_unique.size/float(N**3)
print 'Fraction of total indices to input indices:', npix*nfreq/float(N**3)
#print 'Rms pixel location error:', np.sqrt(np.mean(total_sq_err))

np.savez(pick+".npz",ninds=ninds,Zs=Zs)

## Estimate the overlap in pixel selection between neighboring healpix shells vs r.
### There doesn't seem to be any overlap (ie., picking the same cart pixel for two neighboring maps) per healpix pixel.
### There is, of course, some overlap in overall selection. Each map uses about 40% of the cube. It shouldn't matter, though, since
### 	the cube changes with Z.
#for i in range(nfreq-1):
#	r1, r2 = r_mpc[i], r_mpc[i+1]
#	X,Y,Z = hp.pix2vec(nside, np.arange(npix))
#	XYZ = np.array([X,Y,Z])
#	l1,m1,n1 = (np.floor(((r1*XYZ)%L)/dx)).astype(int)
#	l2,m2,n2 = (np.floor(((r2*XYZ)%L)/dx)).astype(int)
#	inds1 = np.unique(l1 + m1*(N+1) + n1*(N+1)**2)
#	inds2 = np.unique(l2 + m2*(N+1) + n2*(N+1)**2)
#	overlap = list(set(inds1.tolist()) & set(inds2.tolist()))
#	print i,len(overlap)
#	
