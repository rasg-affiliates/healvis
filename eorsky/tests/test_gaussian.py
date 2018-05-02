#Test estimators of gaussian cube

from eorsky import eorsky, shell_pspec
import numpy as np
import pylab
from astropy.cosmology import WMAP9

ek = eorsky()
ek.set_freqs("mwa")
ek.select_range(chan=[0,30]) #z=[6.10,6.20])
NSIDE=512
ek.nside=NSIDE
ek.make_gaussian_shell(NSIDE,ek.freqs,var=2.0)
zavg = np.mean(ek.Z)

L=300.
trans_dist = WMAP9.comoving_transverse_distance(zavg).to("Mpc").value
select_radius = (L/trans_dist)*(180./np.pi)



#20\deg radius selection
#k0, pk0,err0 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100,pyramid=True)
#k1, pk1,err1 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100,lomb_scargle=True)
k2, pk2,err2 = shell_pspec(ek,radius=select_radius,N_sections=1,cosmo=True)

#ek.make_gaussian_cube

#pylab.errorbar(k0,pk0,err0,marker='.')
#pylab.errorbar(k1,pk1,err1,marker='.')
pylab.errorbar(k2,pk2,err2,marker='.')
pylab.show()
