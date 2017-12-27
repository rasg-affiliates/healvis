## Different power spectrum estimators for sphere or Cartesian cubes.

import scipy.stats, numpy as np, healpy as hp
from astropy.cosmology import WMAP9 as cosmo
from astropy.stats import LombScargle

def ls_pspec_1D(cube, L, r_mpc, Nkbins=100):
    """ Estimate the 1D power spectrum for square regions with non-uniform distances using Lomb-Scargle periodogram in the radial direction."""
    ## TODO Enable a set of k_|| bins to be passed in
    Nx,Ny,Nz = cube.shape
    try:
        Lx, Ly, Lz = L
    except TypeError:
        Lx = L; Ly = L; Lz = L   # Assume L is a single side length, same for all
    dx, dy, dz = Lx/float(Nx), Ly/float(Ny), Lz/float(Nz)
    kxvals = np.fft.fftfreq(Nx,d=dx)   #Mpc^-1
    kyvals = np.fft.fftfreq(Ny,d=dy)   #Mpc^-1
    assert len(r_mpc) == Nz
    kzvals, powtest = LombScargle(r_mpc,cube[0,0,:]).autopower(nyquist_factor=1,normalization="psd")

    _cube = np.zeros((Nx,Ny,kzvals.size))
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            kzvals, power = LombScargle(r_mpc,cube[i,j,:]).autopower(nyquist_factor=1,normalization="psd")
            _cube[i,j] = np.sqrt(power)

    _d = np.fft.fft2(_cube,axes=(0,1))
    kxvals = kxvals[kxvals>0]
    kyvals = kyvals[kyvals>0]
    Nx = np.sum(kxvals>0)
    Ny = np.sum(kyvals>0)
    _d = _d[1:Nx+1,1:Ny+1,:]     #Remove the nonpositive k-terms from the 2D FFT
    ps3d = np.abs(_d*_d.conj())/float(Nx*Ny)
    ps1d = ps3d.ravel()
    kx, ky, kz = np.meshgrid(kxvals,kyvals,kzvals)
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    means, bins, binnums = scipy.stats.binned_statistic(k.ravel(),ps1d,statistic='mean',bins=Nkbins)
    pk = means
    kbins  = (bins[1:] + bins[:-1])/2.
    return pk, kbins

def r_pspec_1D(cube, L, Nkbins=100):
    """ Estimate the 1D power spectrum for a rectilinear cube """
    Nx,Ny,Nz = cube.shape
    try:
        Lx, Ly, Lz = L
    except TypeError:
        Lx = L; Ly = L; Lz = L   # Assume L is a single side length, same for all
    dx, dy, dz = Lx/float(Nx), Ly/float(Ny), Lz/float(Nz)
    _d = np.fft.fftn(cube)
    kxvals = np.fft.fftfreq(Nx,d=dx)   #Mpc^-1
    kyvals = np.fft.fftfreq(Ny,d=dy)   #Mpc^-1
    kzvals = np.fft.fftfreq(Nz,d=dz)   #Mpc^-1
    kxvals = kxvals[kxvals>0]
    kyvals = kyvals[kyvals>0]
    kzvals = kzvals[kzvals>0]
    Nx = np.sum(kxvals>0)
    Ny = np.sum(kyvals>0)
    Nz = np.sum(kzvals>0)
    _d = _d[1:Nx+1,1:Ny+1,1:Nz+1]     #Remove the nonpositive k-terms from the 2D FFT
    ps3d = np.abs(_d*_d.conj())/float(Nx*Ny*Nz)
    ps1d = ps3d.ravel()
    kx, ky, kz = np.meshgrid(kxvals,kyvals,kzvals)
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    means, bins, binnums = scipy.stats.binned_statistic(k.ravel(),ps1d,statistic='mean',bins=Nkbins)
    pk = means
    kbins  = (bins[1:] + bins[:-1])/2.
    return pk, kbins

def r_pspec_sphere(shell, nside, radius, dims=None, hpx_inds = None, N_sections=None, freqs = None, dist=None, lomb_scargle=False, Nkbins=None):
    """ Estimate the power spectrum of a healpix shell by projecting regions to Cartesian space.
            Shell = (Npix, Nfreq)  Healpix shell, or section of one.
            radius = (analogous to beam width)
            N_sections = Compute the power spectrum for this many regions (like snapshots in observing)
            dims (3) = Box dimensions in Mpc. If not available, check for freqs and use cosmology module
            lomb_scargle = Use the Lomb-Scargle Periodogram to estimate radial FT.
     """ 
    ## For now, pick a random point within the (assumed full) spherical shell, and get the pixels within distance radius around it.
    if hpx_inds is None: 
        Npix = hp.nside2npix(nside)
        hpx_inds = np.arange(Npix)
    else:
        Npix = hpx_inds.size
    if N_sections is None: N_sections=1

    radius *= np.pi/180.   #Deg2Rad

    if freqs is None and lomb_scargle: raise ValueError(" Frequency data must be provided to use the Lomb-Scargle method ")

    ## Loop over sections, choosing a different center each time.
    if dims is None:
        if freqs is None:
            if dist is None: raise ValueError("Angular diameter distance or frequencies must be defined")
            Lx = 2*radius*dist  #Dist = angular diameter distance in Mpc
            Ly = Lx; Lz = Lx
        else:
            Zs = 1420./freqs - 1.
            r_mpc = cosmo.comoving_distance(Zs).to("Mpc").value
            Lz = np.max(r_mpc) - np.min(r_mpc)
            dist = cosmo.angular_diameter_distance(np.mean(Zs)).value
            Lx = 2*radius*dist
            Ly = Lx
        dims = [Lx,Ly,Lz]
    if lomb_scargle:
        try:
            r_mpc
        except:
            Zs = 1420./freqs - 1.
            r_mpc = cosmo.comoving_distance(Zs).to("Mpc").value
            

    ## Average power spectrum estimates from different parts of the sky.

    if Nkbins is None: Nkbins=100
    pk = np.zeros(Nkbins)
    for s in range(N_sections):

        print "Section ",s
        cent = hp.pix2vec(nside,np.random.choice(hpx_inds))
        inds = hp.query_disc(nside,cent,radius)
        vecs = hp.pixelfunc.pix2vec(nside, inds)
	
        print "Pixels in selection: ",inds.shape
    
        dt, dp = hp.rotator.vec2dir(cent,lonlat=True)
    
        #Estimate X extent by the number of pixels in the selection. 2*radius/pixsize
        Xextent = int(2*radius/hp.nside2resol(nside))
        mwp = hp.projector.CartesianProj(xsize=Xextent, rot=(dt,dp,0))
    
        i,j   = mwp.xy2ij(mwp.vec2xy(vecs[0],vecs[1],vecs[2]))
        imin, imax = min(i), max(i)
        jmin, jmax = min(j), max(j)
        fun = lambda x,y,z: hp.pixelfunc.vec2pix(nside,x,y,z,nest=False)
    
        ## Construct a projected cube:
    
        cube = np.zeros(((imax-imin),(jmax-jmin),shell.shape[1]))
    
        for i in range(shell.shape[1]):
            cube[:,:,i] = mwp.projmap(shell[:,i], fun)[imin:imax, jmin:jmax]
 
        if lomb_scargle:
            pki, kbins = ls_pspec_1D(cube,dims,r_mpc,Nkbins=Nkbins)
        else:
            pki, kbins = r_pspec_1D(cube,dims,Nkbins=Nkbins)
        pk += pki

    pk /= N_sections
    import IPython; IPython.embed()
    import sys; sys.exit()
    return kbins, pk

    
