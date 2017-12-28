from eorsky import eorsky, r_pspec_sphere
import numpy as np
import pylab

ek = eorsky()
ek.read_hdf5("/users/alanman/data/alanman/BubbleCube/TiledHpxCubes/paper_comp_lin_light_cone_surfaces.hdf5",chan_range=[85,115])
ek.read_pspec("/gpfs/data/jpober/21cmSimulation/21cmFAST/Output_files/Deldel_T_power_spec/ps_no_halos_z008.30_nf0.574576_useTs0_zetaX-1.0e+00_alphaX-1.0_TvirminX-1.0e+00_aveTb011.56_Pop-1_256_300Mpc_v3")

k0, pk0, err0 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100,pyramid=True)
k1, pk1, err1 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100,lomb_scargle=True)
k2, pk2, err2 = r_pspec_sphere(ek.hpx_shell, ek.nside, 15, freqs=ek.freqs,N_sections=5,Nkbins=100)

pylab.plot(k0,pk0,marker='.')
pylab.plot(k1,pk1,marker='.')
pylab.plot(k2,pk2,marker='.')
pylab.show()
