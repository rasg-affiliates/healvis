from eorsky import tlmk_transform
import numpy as np, healpy as hp
from astropy.cosmology import WMAP9
from scipy.special import spherical_jn as jl

freqs = np.linspace(167.075, 177.075, 129)
Zs = 1420./freqs - 1.
r_mpc = WMAP9.comoving_distance(Zs).to("Mpc").value
nside = 256
Nchan = freqs.size
Npix = hp.nside2npix(nside)

##Gaussian shell
#shell = np.random.normal(0.0,1.0,(Npix,Nchan))
##Sine wave shell with period of 5 Mpc (sine wave with radius).
lmax = 3*nside-1
Nlm  = hp.sphtfunc.Alm.getsize(lmax)
ls, ms = hp.sphtfunc.Alm.getlm(lmax)

###

dr = np.abs(r_mpc[1]-r_mpc[0])
per=5.
wave = np.sin(2*np.pi* r_mpc/per)#* np.exp( - (r_mpc - np.mean(r_mpc))**2 / ( 2 * Nchan*dr)) 

#kz = 2*np.pi/per
#wave = jl(100,kz*r_mpc)   #Check for orthogonality 

alm_shell = np.tile(wave,Nlm).reshape((Nlm,Nchan))
#alm_shell[np.where(ls!=10)] = 0  # Confirm that the transformation is in the right l
alm2map = lambda x: hp.alm2map(np.ascontiguousarray(x).astype(np.complex128),nside,sigma=0.0,fwhm=0.0)
shell = np.apply_along_axis(alm2map,0,alm_shell)

tlmk, wlk, basis  = tlmk_transform(shell,nside,r_mpc)
kz, ls, ms = basis

#test = tlmk[np.where(ls==0)]
