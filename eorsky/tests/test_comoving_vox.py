## Plot comoving voxel volume vs. different parameters.

import numpy as np
from eorsky import comoving_voxel_volume, common_freqs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

## TODO Plot volume vs (z,omega) for the mwa channel width. Logspaced pixel areas.
###     Plot volume vs (z, dnu) for different Nsides (multiple plots).

z_v_o=False
z_v_ch=True

Nfreq = 200
Nomega = 100
Nch = 300

dnu_mwa = 0.080   #MHz
dnu_all = np.linspace(0.030,0.090,Nch)

Nsides = [256,512,1024,2048]

Omax, Omin = 4*np.pi/(12.*256**2), 4*np.pi/(12.*2048**2)
omegas = np.logspace(np.log10(Omin),np.log10(Omax),Nomega)

freqs = common_freqs('mwa')

Z_mwa = 1420./freqs - 1.

z_list = np.linspace(min(Z_mwa),max(Z_mwa),Nfreq)

ref_vol = (300/256.)**3   #Reference, for 21cmFAST cube


if z_v_o:

    voxels = comoving_voxel_volume(z_list,dnu_mwa, omegas)
    
    ext = [z_list[0],z_list[-1],Omin,Omax]
    
    plt.imshow(voxels/ref_vol,extent=ext,origin='lower')
    for nside in Nsides:
        Om = 4*np.pi/(12.*float(nside)**2)
        plt.axhline(Om,label=str(nside))
        plt.text(0,np.log10(Om)+1.0,str(nside),rotation=0,color='red')
    plt.colorbar()
    plt.yscale('log')
    plt.legend()
    plt.show()

if z_v_ch:
    ##Plot vol vs (z,channel_width)

    fig,axes = plt.subplots(nrows=1,ncols=len(Nsides),sharey='all')
   

    ### Why are the y ranges reversed from what they should be?
    for ni in range(len(Nsides)):
        ax = axes[ni]
        nside= Nsides[ni]
        Om = 4*np.pi/(12*float(nside)**2)
        voxels = comoving_voxel_volume(z_list,dnu_all,Om)
        ext = [z_list[0],z_list[-1],dnu_all[0],dnu_all[-1]]
        im = ax.imshow(voxels/ref_vol,extent=ext,aspect='auto')
        if ni>0: ax.yaxis.set_visible(False)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="20%", pad=0.05)
        cbar = plt.colorbar(im,cax=cax)
#        ax.set_yscale('log')
        print ext
        ax.set_title("Nside="+str(nside))
    plt.show()
    
