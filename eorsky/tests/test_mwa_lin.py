from eorsky import eorsky, r_pspec_sphere
import numpy as np

ek = eorsky()
#ek.read_hdf5("/users/alanman/data/alanman/BubbleCube/TiledHpxCubes/mwa_lin_light_cone_surfaces.hdf5",chan_range=[161,238])
ek.read_hdf5("/users/alanman/data/alanman/BubbleCube/TiledHpxCubes/mwa_points.hdf5",chan_range=[161,238])

#20\deg radius selection
r_pspec_sphere(ek.hpx_shell, ek.nside, 20, freqs=ek.freqs,N_sections=10)#,lomb_scargle=True)
