#Compare calculation of 21cmFAST cube pspec to its calculated pspec.
#TODO Slice the cube and do the r_pspec_sphere calculation on it. Compare to calculation on the initial cube.

from eorsky import r_pspec_1D, eorsky, r_pspec_sphere
import pylab as p
import numpy as np
pi = np.pi

ref_file = "../data/pspec_21cmFAST_z008.00.dat"
dat_file = "../data/21cmFAST_z008.00_256_300Mpc"
L = 300.   #Mpc
N = 256    #Pixels

dat = np.fromfile(dat_file,dtype=np.float32).reshape((N,N,N))
dat -= np.mean(dat)
ks, pk = r_pspec_1D(dat,L, cosmo=True)
pk *= (ks**3/2*np.pi**2)


Nz = N
dz = L/float(N)
kfact = 2*np.pi
kz = np.fft.fftfreq(Nz,d=dz)*kfact   #Mpc^-1

ref_spectrum = np.genfromtxt(ref_file)
ref_ks = ref_spectrum[:,0]
ref_pk = ref_spectrum[:,1]

#p.plot(ks, pk,marker='.',label='Meas')
#p.plot(ref_ks,ref_pk,marker='.',label="Ref")
#p.yscale('log'); p.xscale('log')
#p.show()

ek = eorsky()
ek.rect_cube = dat
ek.L = L
ek.N = N

#freqs = np.linspace(100,200,1024)[481:633]
#freqs = np.linspace(100,200,204)[95:125]
freqs = np.linspace(182.95-30.72/2.,182.95+30.72/2., 384)    ## Actual MWA channel structure... I've been doing it wrong.
freqs = freqs[100:200]
print "Bandwidth: ", freqs[-1] - freqs[0]
ek.freqs = freqs
ek.nside=512

ek.update()
ek.slice()

s_ks, s_pk, errs = r_pspec_sphere(ek.hpx_shell, ek.nside, 20, freqs=ek.freqs)
s_pk *= s_ks**3/(2*np.pi**2)

p.plot(ks, pk,marker='.',label='Meas')
p.plot(ref_ks,ref_pk,marker='.',label="Ref")
p.plot(s_ks,s_pk,marker='.',label="spherical")
p.yscale('log'); p.xscale('log')
p.legend()
p.show()
