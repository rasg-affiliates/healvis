# -*- coding: utf-8 -*-
from eorsky import pspec_funcs, comoving_voxel_volume, comoving_radial_length, comoving_transverse_length
from astropy.cosmology import WMAP9
import nose.tools as nt
import numpy as np
import pylab as pl
import matplotlib.colors
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def compare_averages_shell_pspec_dft():
    """
    Take a gaussian shell and confirm its power spectrum using shell_project_pspec.
    """

    select_radius = 5. #degrees

    Nside=256
    Npix = 12 * Nside**2
    Omega = 4*np.pi/float(Npix)

    Nfreq = 100
    freqs = np.linspace(167.0, 177.0, Nfreq)
    dnu = np.diff(freqs)[0]
    Z = 1420/freqs - 1.

    sig = 2.0
    mu = 0.0
    shell = np.random.normal(mu, sig, (Npix, Nfreq))

    dV = comoving_voxel_volume(Z[Nfreq/2], dnu, Omega)
    variances = []
    means = []
    pks = []

    gs = gridspec.GridSpec(2, 3)
    fig = pl.figure()

    ax0 = pl.subplot(gs[0, 0:2])
    ax1 = pl.subplot(gs[1, 0])
    ax3 = pl.subplot(gs[1, 1])
    ax2 = pl.subplot(gs[:, 2])

    steps = range(10,110,10)
    vmin,vmax = min(steps),max(steps)
    normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.viridis

    for n in steps:
        Nkbins = 100
        kbins, pk = pspec_funcs.shell_project_pspec(shell, Nside, select_radius, freqs=freqs, Nkbins=Nkbins, N_sections=n, cosmo=True, method='dft', error=False)
        variances.append(np.var(pk[0:Nkbins-5]))
        means.append(np.mean(pk[0:Nkbins-5]))
        pks.append(pk)
        ax0.plot(kbins, pk, label=str(n), color=colormap(normalize(n)))

    ax0.axhline(y=dV*sig**2, color='k', lw=2.0)
#    ax0.legend()
    scalarmappable = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappable.set_array(steps)
    fig.colorbar(scalarmappable,label=r'Number of snapshots', ax=ax0)
    ax0.set_ylabel(r"P(k) [mK$^2$ Mpc$^{3}]$")
    ax0.set_xlabel(r"k [Mpc$^{-1}]$")
    ax1.plot(steps, np.array(variances), label="Variance")
    ax1.set_ylabel(r"Variance(P(k)) [mK$^4$ Mpc$^{6}]$")
    ax1.set_xlabel(u"Number of 5° snapshots")
    ax3.plot(steps, means, label="Mean")
    ax3.set_ylabel(r"Mean(P(k)) [mK$^2$ Mpc$^{3}]$")
    ax3.set_xlabel(u"Number of 5° snapshots")
    ax1.legend()
    ax3.legend()
    im = ax2.imshow(np.array(pks)[:,0:Nkbins-5], aspect='auto')#, norm=mcolors.LogNorm())
    fig.colorbar(im, ax=ax2)
    print('Fractional deviation: ', np.mean(np.abs(pk - dV*sig**2)))
    pl.show()

def compare_selection_radii_shell_pspec_dft():
    N_sections=10

    Nside=256
    Npix = 12 * Nside**2
    Omega = 4*np.pi/float(Npix)

    Nfreq = 100
    freqs = np.linspace(167.0, 177.0, Nfreq)
    dnu = np.diff(freqs)[0]
    Z = 1420/freqs - 1.

    sig = 2.0
    mu = 0.0
    shell = np.random.normal(mu, sig, (Npix, Nfreq))

    dV = comoving_voxel_volume(Z[Nfreq/2], dnu, Omega)
    variances = []
    pks = []
    means = []

    gs = gridspec.GridSpec(2, 3)
    fig = pl.figure()

    ax0 = pl.subplot(gs[0, 0:2])
    ax1 = pl.subplot(gs[1, 0])
    ax3 = pl.subplot(gs[1, 1])
    ax2 = pl.subplot(gs[:, 2])

    steps = np.linspace(2,20,20)
    vmin,vmax = min(steps),max(steps)
    normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.viridis

    for s in steps:
        Nkbins = 100
        kbins, pk = pspec_funcs.shell_project_pspec(shell, Nside, s, freqs=freqs, Nkbins=Nkbins, N_sections=N_sections, cosmo=True, method='dft', error=False)
        variances.append(np.var(pk[0:Nkbins-5]))
        pks.append(pk)
        means.append(np.mean(pk[0:Nkbins-5]))
        ax0.plot(kbins, pk, label=u'{:0.2f}°'.format(s), color=colormap(normalize(s)))

    ax0.axhline(y=dV*sig**2)
#    ax0.legend(ncol=2,loc=3)
    scalarmappable = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappable.set_array(steps)
    fig.colorbar(scalarmappable,label=r'Selection radius', ax=ax0)
    ax0.set_ylabel(r"P(k) [mK$^2$ Mpc$^{3}]$")
    ax0.set_xlabel(r"k [Mpc$^{-1}]$")
    ax1.plot(steps, np.array(variances), label="Variance")
    ax1.set_ylabel(r"Variance(P(k)) [mK$^4$ Mpc$^{6}]$")
    ax1.set_xlabel(u"Selection radius (degrees)")
    ax3.plot(steps, means, label="Mean")
    ax3.set_ylabel(r"Mean(P(k)) [mK$^2$ Mpc$^{3}]$")
    ax3.set_xlabel(u"Selection radius (degrees)")
    ax1.legend()
    ax3.legend()
    ax1.xaxis.set_major_formatter(FormatStrFormatter(u'%0.2f°'))
    ax3.xaxis.set_major_formatter(FormatStrFormatter(u'%0.2f°'))
    im = ax2.imshow(np.array(pks)[:,0:Nkbins-5], aspect='auto', norm=mcolors.LogNorm())
    fig.colorbar(im, ax=ax2)
    pl.show()



if __name__ == '__main__':
    #compare_averages_shell_pspec_dft()
    compare_selection_radii_shell_pspec_dft()
