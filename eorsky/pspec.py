## Different power spectrum estimators for sphere or Cartesian cubes.

import scipy.stats, numpy as np, healpy as hp

def r_pspec_1D(cube, L, Nkbins=100):
    """ Estimate the 1D power spectrum for a rectilinear cube """
    ## TODO --> Don't assume all sides have the same length
    Nx,Ny,Nz = cube.shape   #Assume equal side lengths
    try:
        Lx, Ly, Lz = L
    except TypeError:
        Lx = L; Ly = L; Lz = L   # Assume L is a single side length, same for all
    dx, dy, dz = Lx/float(Nx), Ly/float(Ny), Lz/float(Nz)
    _d = np.fft.fftn(cube)
    ps3d = np.abs(_d*_d.conj())
    ps1d = ps3d.ravel()
    kxvals = np.fft.fftfreq(Nx,d=dx)   #Mpc^-1
    kyvals = np.fft.fftfreq(Ny,d=dy)   #Mpc^-1
    kzvals = np.fft.fftfreq(Nz,d=dz)   #Mpc^-1
    kx, ky, kz = np.meshgrid(kxvals,kyvals,kzvals)
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    means, bins, binnums = scipy.stats.binned_statistic(k.ravel(),ps1d,statistic='mean',bins=Nkbins)
    pk = means/(Nx*Ny*Nz)
    kbins  = (bins[1:] + bins[:-1])/2.
    return pk, kbins

def r_pspec_sphere(shell, nside, radius, dims=None, hpx_inds = None, N_sections=None, freqs = None, dist=None):
    """ Estimate the power spectrum of a healpix shell by projecting regions to Cartesian space.
            Shell = (Npix, Nfreq)  Healpix shell, or section of one.
            radius = (analogous to beam width)
            N_sections = Compute the power spectrum for this many regions (like snapshots in observing)
            dims (3) = Box dimensions in Mpc. If not available, check for freqs and use cosmology module
     """ 
    ## For now, pick a random point within the (assumed full) spherical shell, and get the pixels within distance radius around it.
    if hpx_inds is None: 
        Npix = hp.nside2npix(nside)
        hpx_inds = np.arange(Npix)
    else:
        Npix = hpx_inds.size
    if N_sections is None: N_sections=1

    radius *= np.pi/180.   #Deg2Rad

    ## Loop over sections, choosing a different center each time.
    ## Need to know box dimensions (in Mpc) before starting.
    ##      Lx = Ly = 2*radius/angular_diameter_distance
    ##      Lz = Lx
    if dims is None:
        if freqs is None:
            if dist is None: raise ValueError("Angular diameter distance or frequencies must be defined")
            Lx = 2*radius/dist  #Dist = angular diameter distance in Mpc
            Ly = Lx; Lz = Lx
        else:
            from astropy.cosmology import WMAP9 as cosmo
            Zs = 1420./freqs - 1.
            r_mpc = cosmo.comoving_distance(Zs).to("Mpc").value
            Lz = np.max(r_mpc) - np.min(r_mpc)
            dist = cosmo.angular_diameter_distance(np.mean(Zs)).value
            Lx = 2*radius/dist
            Ly = Lx
        dims = [Lx,Ly,Lz]

    ## Average power spectrum estimates from different parts of the sky.

    Nkbins = 100
    pk = np.zeros(Nkbins)
    for _ in range(N_sections):

        print _ 
        cent = hp.pix2vec(nside,np.random.choice(hpx_inds))
        inds = hp.query_disc(nside,cent,radius)
        vecs = hp.pixelfunc.pix2vec(nside, inds)
    
        dt, dp = hp.rotator.vec2dir(cent,lonlat=True)
    
        #Estimate X extent by the number of pixels in the selection. 2*radius/pixsize
        Xextent = int(2*radius/hp.nside2resol(nside))
        mwp = hp.projector.CartesianProj(xsize=Xextent, rot=(dt,dp,0))
    
        i,j   = mwp.xy2ij(mwp.vec2xy(vecs[0],vecs[1], vecs[2]))
        imin, imax = min(i), max(i)
        jmin, jmax = min(j), max(j)
    
        fun = lambda x,y,z: hp.pixelfunc.vec2pix(nside,x,y,z,nest=False)
    
        ## Construct a projected cube:
    
        cube = np.zeros(((imax-imin),(jmax-jmin),shell.shape[1]))
    
        for i in range(shell.shape[1]):
            cube[:,:,i] = mwp.projmap(shell[:,i], fun)[imin:imax, jmin:jmax]
    
        pki, kbins = r_pspec_1D(cube,dims,Nkbins=Nkbins)
        pk += pki

    pk /= N_sections

    import IPython; IPython.embed()

    
