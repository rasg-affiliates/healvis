from eorsky import eorsky, r_pspec_sphere
import numpy as np

ek = eorsky()
ch0,ch1 = 161,238

#ek.read_hdf5("/users/alanman/data/alanman/BubbleCube/TiledHpxCubes/mwa_lin_light_cone_surfaces.hdf5",chan_range=[ch0,ch1])
ek.read_hdf5("../data/tiled_gaussian_nside512_light_cone_surfaces.hdf5",chan_range=[ch0,ch1])


#20\deg radius selection
kbins, pk, err = r_pspec_sphere(ek.hpx_shell, ek.nside, 20, freqs=ek.freqs,N_sections=10,lomb_scargle=True)

