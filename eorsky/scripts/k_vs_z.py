from astropy.cosmology import WMAP9 as cosmo
import healpy as hp, numpy as np
import matplotlib.pyplot as p
import matplotlib.colors as mcolors
import matplotlib.cm as cm

#Plot k_max of 500 Mpc box against kmax vs. z for different NSIDE or channel_widths

kperp = False
kpar = True
both = False
n_curve = 4

freqs = np.linspace(100,200,1024)
Zs = 1420./freqs - 1
pixsize = np.zeros((n_curve,1024))

box_L = 300.    #Mpc
box_N = 256.
dx = box_L/box_N

D_mpc = cosmo.angular_diameter_distance(Zs).to("Mpc").value
if kperp or both:
 for i in range(n_curve):
    nside=2**(i+8)
    pixsize[i,:]=hp.nside2resol(nside) * D_mpc
    p.plot(Zs,2*np.pi/pixsize[i,:],label='NSIDE='+str(nside),color='k')
 if not both:
     p.axhspan(2*np.pi/(np.sqrt(3)*box_L),2*np.pi/dx,alpha=0.5,color='r')
#     p.plot(Zs,2*np.pi/(300/256.) * np.ones_like(D_mpc),lw=3.0, label="Box L=300, N=256")
#     p.legend()
     p.xlabel("Z")
     p.ylabel(r'k$_\perp$ Mpc$^{-1}$')
     p.show()


#df = np.array([40, 50, 60, 70, 80,90])/1e6   #GHz
df = np.arange(60,120,10)/1e6  #GHz
fmax = np.max(df)

#normalize = mcolors.Normalize(vmin=df.min(), vmax=df.max())
normalize = mcolors.Normalize(vmin=df.min()*1e6, vmax=df.max()*1e6)
colormap = cm.viridis

##From capo
def dL_df(z, omega_m=cosmo.Om0):
    '''[Mpc]/GHz, from Furlanetto et al. (2006)'''
    return (1.7 / 0.1) * ((1+z) / 10.)**.5 * (omega_m/0.15)**-0.5 * 1e3 * 0.7
if kpar or both:
 Lfull = np.array([dL_df(z)*0.01 for z in Zs]).astype(np.float)
 print Lfull
 for i in range(len(df)):
    Ls = np.array([dL_df(z)*df[i] for z in Zs])
    col = cm.jet(df[i]/fmax)
    p.plot(Zs, 2*np.pi/Ls,color=colormap(normalize(df[i]*1e6)))#, label='df='+str(df[i]*1e6)+'kHz') 
 Ls = np.array([dL_df(z)*80e-6 for z in Zs]).astype(np.float)
 p.plot(Zs, 2*np.pi/Ls,label='MWA',linestyle="-",color='k')#, label='df='+str(df[i]*1e6)+'kHz')
 Ls = np.array([dL_df(z)*98e-6 for z in Zs]).astype(np.float)
 p.plot(Zs, 2*np.pi/Ls,label='HERA (full)',linestyle="--",color='k')#, label='df='+str(df[i]*1e6)+'kHz')
 Ls = np.array([dL_df(z)*492e-6 for z in Zs]).astype(np.float)
 p.plot(Zs, 2*np.pi/Ls,label='PAPER (comp)',linestyle=":",color='k')#, label='df='+str(df[i]*1e6)+'kHz')
 p.plot(Zs, 2*np.pi/Lfull, lw=4.0, label="BW 10MHz",color='k')
 #p.plot(Zs,2*np.pi/dx * np.ones_like(D_mpc),lw=3.0, label="Box")
 p.axhspan(2*np.pi/(np.sqrt(3.)*box_L),2*np.pi/dx,alpha=0.25,color='r')
 p.legend()
 scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
 scalarmappaple.set_array(np.arange(df.size))
 cbar = p.colorbar(scalarmappaple,label="Channel width (kHz)")
 cbar.ax.set_yticklabels(map(str,df*1e6))
 p.xlabel("Z")
 p.ylabel(r'k$_\parallel$ Mpc$^{-1}$')
 p.show()



