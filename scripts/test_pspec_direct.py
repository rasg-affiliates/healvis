# Make a cube with a specified power spectrum.
# Slice it, and compare with input.

from eorsky import eorsky, shell_pspec, r_pspec_1D
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9

ch0, ch1 = 0,130
ek = eorsky()
ek.set_freqs("mwa")
ek.select_range(chan=[0,130]) #z=[6.10,6.20])

Nsec=1
NSIDE=512
L=300.
N=256
cosmo=True

ek.nside=NSIDE
ek.make_gaussian_cube(N,L,var=2.0)

r_mpc = np.linspace(0,L,N)    # Test the DFT mode.
k0, pk0 = r_pspec_1D(ek.rect_cube,L, r_mpc = r_mpc, cosmo=cosmo)


## Now check the sliced shell

ek.slice(cosmo=cosmo)
zavg = np.mean(ek.Z)
trans_dist = WMAP9.comoving_distance(zavg).to("Mpc").value
select_radius = (L/trans_dist)*(180./np.pi) * 2
print 'Select radius = {}'.format(select_radius)

k1, pk1,err1 = shell_pspec(ek,radius=select_radius,N_sections=Nsec,cosmo=cosmo)

plt.plot(k0,pk0, label='Cube')
plt.plot(k1,pk1, label='Shell')
plt.xscale('log'); plt.yscale('log')
plt.legend()
plt.show()

