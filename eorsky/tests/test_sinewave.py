from eorsky import  r_pspec_1D
import numpy as np
import pylab
from astropy.cosmology import WMAP9

freqs = np.linspace(182.95-30.72/2.,182.95+30.72/2., 384)
Zs = 1420./freqs - 1.
r_mpc = WMAP9.comoving_distance(Zs).to("Mpc").value

L = (r_mpc[-1] - r_mpc[0])/2.   #Just take half the total length.
Nx = 10; Ny = 10
period = 5   # Mpc 

cube = np.sin(2*np.pi * r_mpc/period)
cube = np.repeat(cube,Nx*Ny).reshape(384,Nx,Ny).swapaxes(0,2)

(kx,ky,kz),pk = r_pspec_1D(cube,L,return_3d=True)
pk1d = np.mean(pk,axis=(0,1))
pylab.plot(kz,pk1d)
pylab.xlabel(r'kz Mpc$^{-1}$')
pylab.ylabel("power")
pylab.show()
