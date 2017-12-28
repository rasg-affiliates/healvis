from eorsky import eorsky, r_pspec_sphere
import numpy as np
import pylab

ek = eorsky()
NSIDE=512
#freqs=np.linspace(100,200,1024)
#freqs = freqs[430:633]
freqs=np.linspace(100,200,203)
freqs = freqs[85:115]
##Select 20 MHz

ek.make_gaussian_shell(NSIDE,freqs,var=1.0)

#20\deg radius selection

k0, pk0,err0 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100,pyramid=True)
k1, pk1,err1 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100,lomb_scargle=True)
k2, pk2,err2 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100)

pylab.errorbar(k0,pk0,err0,marker='.')
pylab.errorbar(k1,pk1,err1,marker='.')
pylab.errorbar(k2,pk2,err2,marker='.')
pylab.show()
