from eorsky import eorsky, r_pspec_sphere, r_pspec_1D
import healpy as hp
import numpy as np
import pylab

ek = eorsky()
L = 300
N = 256
NSIDE=256
nsect = 100

ek.set_freqs('mwa')

ek.nside = NSIDE

ek.make_gaussian_cube(N,L,var=1.0)

print ek.Nfreq, ek.Npix

k0,pk0 = r_pspec_1D(ek.rect_cube,L)   #The pspec of the input gaussian cube
ek.slice()

#k1, pk1,err1 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=nsect,Nkbins=75,pyramid=True)
#k1, pk1,err1 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100,lomb_scargle=True)
k1, pk1,err1 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15,r_mpc=ek.r_mpc, freqs=ek.freqs,N_sections=nsect,Nkbins=100)

pylab.plot(k0,pk0,marker='.')
pylab.plot(k1,pk1,marker='.')
pylab.xlabel(r'k Mpc$^{-1}$')
pylab.ylabel(r'P(k)  mK$^2$ Mpc$^{-3}$')
pylab.show()
pylab.savefig("/users/alanman/rectloop_eorsky_dft_test.png")