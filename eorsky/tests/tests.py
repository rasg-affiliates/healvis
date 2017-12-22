from eorsky import eorsky, r_pspec_sphere

ek = eorsky()
ek.read_hdf5("../data/gaussian_cube.hdf5")

r_pspec_sphere(ek.hpx_shell, ek.nside, 40, freqs=ek.freqs,N_sections=5)
