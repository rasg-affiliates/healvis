from eorsky import eorsky, r_pspec_sphere, r_pspec_1D
import healpy as hp
import numpy as np
import pylab

ek = eorsky()
L = 300
N = 256
NSIDE=256

ek.set_freqs('mwa')

ek.nside = NSIDE

ek.make_gaussian_cube(N,L,var=1.0)
ek.update()

print ek.Nfreq, ek.Npix

k0,pk0 = r_pspec_1D(ek.rect_cube,L)   #The pspec of the input gaussian cube
ek.slice()
ek.update()

#k1, pk1,err1 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100,pyramid=True)
k1, pk1,err1 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100,lomb_scargle=True)
#k2, pk2,err2 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100)

pylab.plot(k0,pk0,marker='.')
pylab.plot(k1,pk1,marker='.')
pylab.show()
