from eorsky import eorsky, shell_pspec
import numpy as np

ek = eorsky()
ch0,ch1 = 0,30

#ek.read_hdf5("/users/alanman/data/alanman/BubbleCube/TiledHpxCubes/mwa_lin_light_cone_surfaces.hdf5",chan_range=[ch0,ch1])
ek.read_hdf5("/users/alanman/data/alanman/BubbleCube/TiledHpxCubes/tiled_gaussian_nside512_light_cone_surfaces.hdf5",chan_range=[ch0,ch1])
#ek.read_bin("delta_T_v3_no_halos_z008.40_nf0.595490_useTs0_zetaX-1.0e+00_alphaX-1.0_TvirminX-1.0e+00_aveTb012.19_Pop-1_256_300Mpc")

#L = 300
#N = 256
#NSIDE=512
#
#ek.nside = NSIDE
#
#
#freqs = np.linspace(100,200,384)
#ek.freqs = freqs[ch0:ch1]
#
#Zs = 1420./freqs - 1
#zmin = Zs[ch1]
#zmax = Zs[ch0]
#zavg = (zmax+zmin)/2.
#
#path="/gpfs/data/jpober/21cmSimulation/21cmFAST/Output_files/Deldel_T_power_spec/"
#files = os.listdir(path)
#files = [ f for f in files if f.startswith('ps_no_halos') ]
#Z = [ f.split("_")[3] for f in files ]
#Z = np.array([ float(t[1:]) for t in Z])
#ind_n = np.argmin(np.abs(Z - zmin))
#ind_x = np.argmin(np.abs(Z - zmax))
#ind_a = np.argmin(np.abs(Z - zavg))
#ek.read_pspec(path + files[ind_a])

#20\deg radius selection
#r_pspec_sphere(ek.hpx_shell, ek.nside, 20, freqs=ek.freqs,N_sections=10)#,lomb_scargle=True)


k2, pk2,err2 = shell_pspec(ek,radius=20,cosmo=False)
pylab.errorbar(k2,pk2,err2,marker='.')
pylab.show()
