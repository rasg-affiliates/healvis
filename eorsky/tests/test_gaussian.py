#Test estimators of gaussian cube

from eorsky import eorsky, shell_pspec
import numpy as np
import pylab

ek = eorsky()
ek.set_freqs("mwa")
#ek.select_range(z=[6.10,6.20])
NSIDE=64
ek.nside=NSIDE
ek.make_gaussian_shell(NSIDE,ek.freqs,var=3.0)

#20\deg radius selection
#k0, pk0,err0 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100,pyramid=True)
#k1, pk1,err1 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100,lomb_scargle=True)
k2, pk2,err2 = shell_pspec(ek,radius=20)

#ek.make_gaussian_cube

#pylab.errorbar(k0,pk0,err0,marker='.')
#pylab.errorbar(k1,pk1,err1,marker='.')
pylab.errorbar(k2,pk2,err2,marker='.')
pylab.show()
