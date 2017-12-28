## Different power spectrum estimators for sphere or Cartesian cubes.

import scipy.stats, numpy as np, healpy as hp
from astropy.cosmology import WMAP9 as cosmo
from astropy.stats import LombScargle


def bin_1d( pk, kcube, Nkbins=100, sigma=False):
    """
        kcube= 3-tuple containing the k values (kx, ky, kz)
        pk   = power cube
    """
    kx, ky, kz = kcube
    Nx, Ny, Nz = pk.shape
    if Nx == kx.size and Ny == ky.size and Nz == kz.size:
         kx, ky, kz = np.meshgrid(kx, ky, kz)
    k1d = np.sqrt(kx**2 + ky**2 + kz**2).flatten()
    p1d = pk.flatten()
    means, bins, binnums = scipy.stats.binned_statistic(k1d,p1d,statistic='mean',bins=Nkbins)
    kbins  = (bins[1:] + bins[:-1])/2.
    if sigma:
        errs, bins, binnums = scipy.stats.binned_statistic(k1d,p1d,statistic=np.var,bins=Nkbins)
        return means, kbins, np.sqrt(errs)
    else:
        return means, kbins

    
def pyramid_pspec(cube, radius, r_mpc, Zs, kz = None, Nkbins=100,sigma=False):
    """
        Estimate the power spectrum in a "pyramid" appoximation:
            radius = in radians
            For each r_mpc, calculate the width of the projected region from the angular diameter distance.
            In the radial direction, calculate discrete fourier transform. In tangential directions, do FFT.
    """
    Nx, Ny, Nz = cube.shape
    D_mpc = cosmo.angular_diameter_distance(Zs).to("Mpc").value
 
    if kz is None:
        dz = np.abs(r_mpc[-1] - r_mpc[0])/float(Nz)   #Mean spacing
        kz = np.fft.fftfreq(Nz,d=dz)
    else:
        assert kz.size == r_mpc.size
    
    ## DFT in radial direction
    M = np.exp(np.pi * 2 * (-1j) * np.outer(kz,r_mpc) )
    _c = np.apply_along_axis(lambda x: np.dot(M,x),2, cube)
    
    ## 2D FFT in the transverse directions
    _c = np.fft.fft2(_c,axes=(0,1))
    
    ##kx and ky
    L = D_mpc * radius   # Transverse lengths in Mpc
    kx = np.array([np.fft.fftfreq(Nx,d=l/Nx) for l in L])
    ky = np.array([np.fft.fftfreq(Ny,d=l/Ny) for l in L])
    kx = np.moveaxis(np.tile(kx[:,:,np.newaxis],Ny),[1,2],[0,1])
    ky = np.moveaxis(np.tile(ky[:,:,np.newaxis],Nx),[1,2],[1,0])
    kz = np.tile(kz[np.newaxis,:],Nx*Ny).reshape(Nx,Ny,Nz)
    
    pk3d = np.abs(_c)**2/(Nx*Ny*Nz)
    
    results = bin_1d(pk3d,(kx,ky,kz),Nkbins=Nkbins,sigma=sigma)
    return results


def ls_pspec_1D(cube, L, r_mpc, Nkbins=100, sigma=False):
    """ Estimate the 1D power spectrum for square regions with non-uniform distances using Lomb-Scargle periodogram in the radial direction."""
    ## TODO Enable a set of kz bins to be passed in
    Nx,Ny,Nz = cube.shape
    try:
        Lx, Ly, Lz = L
    except TypeError:
        Lx = L; Ly = L; Lz = L   # Assume L is a single side length, same for all
    dx, dy, dz = Lx/float(Nx), Ly/float(Ny), Lz/float(Nz)
    kx = np.fft.fftfreq(Nx,d=dx)   #Mpc^-1
    ky = np.fft.fftfreq(Ny,d=dy)   #Mpc^-1
    assert len(r_mpc) == Nz
    kz, powtest = LombScargle(r_mpc,cube[0,0,:]).autopower(nyquist_factor=1,normalization="psd")

    _cube = np.zeros((Nx,Ny,kz.size))
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            kz, power = LombScargle(r_mpc,cube[i,j,:]).autopower(nyquist_factor=1,normalization="psd")
            _cube[i,j] = np.sqrt(power)

    _d = np.fft.fft2(_cube,axes=(0,1))
    kx = kx[kx>0]
    ky = ky[ky>0]
    Nx = np.sum(kx>0)
    Ny = np.sum(ky>0)
    _d = _d[1:Nx+1,1:Ny+1,:]     #Remove the nonpositive k-terms from the 2D FFT
    pk3d = np.abs(_d)**2/(Nx*Ny)
    results = bin_1d(pk3d,(kx,ky,kz),Nkbins=Nkbins, sigma=sigma)

    return results

def r_pspec_1D(cube, L, Nkbins=100,sigma=False):
    """ Estimate the 1D power spectrum for a rectilinear cube """
    Nx,Ny,Nz = cube.shape
    try:
        Lx, Ly, Lz = L
    except TypeError:
        Lx = L; Ly = L; Lz = L   # Assume L is a single side length, same for all
    dx, dy, dz = Lx/float(Nx), Ly/float(Ny), Lz/float(Nz)
    _d = np.fft.fftn(cube)
    kx = np.fft.fftfreq(Nx,d=dx)   #Mpc^-1
    ky = np.fft.fftfreq(Ny,d=dy)   #Mpc^-1
    kz = np.fft.fftfreq(Nz,d=dz)   #Mpc^-1

    pk3d = np.abs(_d)**2/(Nx*Ny*Nz)
    results = bin_1d(pk3d,(kx,ky,kz),Nkbins=Nkbins,sigma=sigma)

    return results

def r_pspec_sphere(shell, nside, radius, dims=None, hpx_inds = None, N_sections=None, freqs = None, dist=None, pyramid=False, lomb_scargle=False, Nkbins=None):
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
    if lomb_scargle or pyramid:
        try:
            r_mpc
            Zs
        except:
            Zs = 1420./freqs - 1.
            r_mpc = cosmo.comoving_distance(Zs).to("Mpc").value

    ## Average power spectrum estimates from different parts of the sky.

    if Nkbins is None: Nkbins=100
    pk = np.zeros(Nkbins)
    errs = np.zeros(Nkbins)
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
            pki, kbins,errsi = ls_pspec_1D(cube,dims,r_mpc,Nkbins=Nkbins,sigma=True)
        elif pyramid:
            pki, kbins,errsi = pyramid_pspec(cube, radius, r_mpc, Zs, Nkbins=Nkbins,sigma=True)

        else:
            pki, kbins,errsi = r_pspec_1D(cube,dims,Nkbins=Nkbins,sigma=True)
        pk += pki
        errs += errsi

    pk /= N_sections
    errs /= N_sections
    return kbins, pk, errs

    
