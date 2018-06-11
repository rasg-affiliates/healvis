from scipy.special import spherical_jn as jl
import numpy as np
from astropy.cosmology import WMAP9
import matplotlib.pyplot as p
import matplotlib.colors as mcolors
import matplotlib.cm as cm


freqs = np.linspace(167.075, 177.075, 129)
Zs = 1420./freqs - 1.
r_mpc = WMAP9.comoving_distance(Zs).to("Mpc").value
Nchan = freqs.size

r_mpc = np.linspace(min(r_mpc),max(r_mpc),Nchan)
dr = r_mpc[1] - r_mpc[0]

kz = np.fft.fftfreq(Nchan,d=dr)*2*np.pi
kz = kz[kz>0]

lmin=0
lmax=700
dl = 25
ki=0

normalize = mcolors.Normalize(vmin=lmin, vmax=lmax)
colormap = cm.viridis
for l in range(lmin,lmax,dl):
    check=np.any((kz[ki]*r_mpc)>l)
    print l, check
    if check: p.plot(r_mpc,jl(l,kz[ki]*r_mpc)**2,color=colormap(normalize(lmax)))
    else: p.plot(r_mpc,jl(l,kz[ki]*r_mpc)**2,color=colormap(normalize(lmin)))
    #p.plot(r_mpc,jl(l,kz[ki]*r_mpc)**2,color=colormap(normalize(l)))
p.yscale('log')
p.ylim(1e-30,1e-4)
p.xlabel(r'r [Mpc]')
p.ylabel(r'W$_{l}$(k)')
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(np.arange(lmax-lmin))
cbar = p.colorbar(scalarmappaple,label=r'$l$')
p.show()
