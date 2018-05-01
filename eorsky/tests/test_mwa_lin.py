from eorsky import eorsky, shell_pspec
from astropy.cosmology import WMAP9
import matplotlib.pyplot as plt
import numpy as np
import os

ek = eorsky()
ch0,ch1 = 0,50

ek.read_hdf5("/users/alanman/data/alanman/BubbleCube/TiledHpxCubes/mwa_lin_light_cone_surfaces.hdf5",chan_range=[ch0,ch1])
#ek.read_hdf5("/users/alanman/data/alanman/BubbleCube/TiledHpxCubes/tiled_gaussian_nside512_light_cone_surfaces.hdf5",chan_range=[ch0,ch1])
#ek.read_bin("delta_T_v3_no_halos_z008.40_nf0.595490_useTs0_zetaX-1.0e+00_alphaX-1.0_TvirminX-1.0e+00_aveTb012.19_Pop-1_256_300Mpc")

L = 300
N = 256
NSIDE=512

ek.nside = NSIDE


freqs = np.linspace(100,200,384)
ek.freqs = freqs[ch0:ch1]

Zs = 1420./freqs - 1
zmin = Zs[ch1]
zmax = Zs[ch0]
zavg = (zmax+zmin)/2.

path="/gpfs/data/jpober/21cmSimulation/21cmFAST/Output_files/Deldel_T_power_spec/"
files = os.listdir(path)
files = [ f for f in files if f.startswith('ps_no_halos') ]
Z = [ f.split("_")[3] for f in files ]
Z = np.array([ float(t[1:]) for t in Z])
ind_n = np.argmin(np.abs(Z - zmin))
ind_x = np.argmin(np.abs(Z - zmax))
ind_a = np.argmin(np.abs(Z - zavg))
ek.read_pspec_text(os.path.join(path, files[ind_a]))

#20\deg radius selection
#r_pspec_sphere(ek.hpx_shell, ek.nside, 20, freqs=ek.freqs,N_sections=10)#,lomb_scargle=True)

## Select radius based on transverse comoving size of box.
trans_dist = WMAP9.comoving_transverse_distance(zavg).to("Mpc").value
select_radius = (L/trans_dist)*(180./np.pi)
print 'Z_avg ', zavg
print 'Radius (deg)={}, trans_dist (Mpc)={}'.format(select_radius,trans_dist)

k2, pk2,err2 = shell_pspec(ek,radius=select_radius,N_sections=1,cosmo=True)

pk2 *= k2**3/(2*np.pi**2)
pk2 *= 1e6    # K^2 -> mK^2

plt.plot(k2,pk2,marker='.', label='Data')
plt.plot(ek.ref_ks,ek.ref_pk,label='Ref')
plt.legend()
plt.xscale('log'); plt.yscale('log')
plt.show()
