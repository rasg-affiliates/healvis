from eorsky import eorsky, r_pspec_sphere
import numpy as np

ek = eorsky()
ek.read_hdf5("../data/gaussian_cube.hdf5")


## Replace the shell with gaussian noise, and attemp to recover the power spectrum
#sh = ek.hpx_shell.shape
#ek.hpx_shell = np.random.normal(0.0,1.0,size=sh)
#print "nside: ",ek.nside
#r_pspec_sphere(ek.hpx_shell, ek.nside, 10, freqs=ek.freqs,N_sections=100)

## Try a lomb-scargle periodogram in the radial direction
sh = ek.hpx_shell.shape
ek.hpx_shell = np.random.normal(0.0,1.0,size=sh)
print "nside: ",ek.nside
r_pspec_sphere(ek.hpx_shell, ek.nside, 10, freqs=ek.freqs,N_sections=1,lomb_scargle=True)
