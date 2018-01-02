from eorsky import eorsky, r_pspec_sphere, r_pspec_1D
import healpy as hp
import numpy as np
import pylab

ek = eorsky()
L = 500
N = 256
NSIDE=2048

#freqs=np.linspace(100,200,1024)
#freqs = freqs[430:633]
freqs=np.linspace(100,200,203)
freqs = freqs[85:115]
##Select 20 MHz
ek.nside = NSIDE
#freqs
ek.freqs = freqs

ek.make_gaussian_cube(N,L,var=1.0)
ek.update()
k0,pk0 = r_pspec_1D(ek.rect_cube,L)   #The pspec of the input gaussian cube

ek.slice()

#20\deg radius selection

k1, pk1,err1 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100,pyramid=True)
#k1, pk1,err1 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100,lomb_scargle=True)
#k2, pk2,err2 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100)

pylab.plot(k0,pk0,marker='.')
pylab.plot(k1,pk1,marker='.')
pylab.show()
