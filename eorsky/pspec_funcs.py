## Different power spectrum estimators for sphere or Cartesian cubes.

import scipy.stats, numpy as np, healpy as hp, time
from astropy.cosmology import WMAP9
from astropy.stats import LombScargle
from scipy.special import spherical_jn as jl   ## Requires scipy version 0.18.0 at least


def bin_1d( pk, kcube, Nkbins=100, error=False):
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

    if error:
        errs, bins, binnums = scipy.stats.binned_statistic(k1d,p1d,statistic=np.var,bins=Nkbins)
        return kbins, means, np.sqrt(errs)/2.
    else:
        return kbins, means

def pk_sphere(slk,wlk):
    """
        Bin a 2d power spectrum (l,k basis) to 1D.
    """
    pk = np.sum(wlk**2*slk,axis=0)
    norm = np.sum(wlk**2,axis=0)
    print 'Norm: ',norm
    return pk/norm

def slk_estimator(tlmk,wlk,ls,ms):
    """
        Given the results of a T(r) -> Tlm(k) transformation, calculate the 2d power spectrum S_l(k)
        (Bin in m, then divide by weights)
    """
    plmk = abs(tlmk)**2
    Nl,Nk = wlk.shape
    Slk = np.zeros((Nl,Nk),dtype=np.complex128)
    lmat = np.unique(ls)
    lmat = np.repeat(lmat,Nk).reshape((Nl,Nk))
    #Bin in m
    for ki in range(Nk):
        np.add.at(Slk[:,ki],ls,plmk[:,ki])
    Slk /= (2*lmat+1)              ## Check --- do I actually get m from -l to l from healpy?
#    check = np.sum(np.abs(np.imag(Slk/wlk)))
#    if check: print 'Slk is not purely real.
    Slk /= wlk
    Slk = np.ma.array(Slk,mask=(wlk==0))
    return Slk

def tlmk_transform(shell, nside, r_mpc, kz=None,lmax=None):
    """
        Return the Tlmk and weights of the transformed shell for power spectrum estimators.
    """
    if lmax is None:  lmax = 3*nside - 1; ret_flag =True
    nlm  = hp.sphtfunc.Alm.getsize(lmax)
    ls, ms = hp.sphtfunc.Alm.getlm(lmax)

    map2alm = lambda x: hp.map2alm(x,lmax=lmax)
    alms = np.apply_along_axis(map2alm,0,shell)
    Nchan = r_mpc.size

    if kz is None:
        ret_flag = True
        dr = np.min(-np.diff(r_mpc))
        print 'dr: ', dr
        kz = np.fft.fftfreq(Nchan,d=dr)*2*np.pi   #Default to using the radial momenta from the minimum spacing.
    kz = kz[np.where(kz>0)]                       #Ignore negative k's since these are meaningless as isotropic k.
                                                  #Ignore k=0.
    Nk = kz.size

    t0 =time.time()
    bess_mat = np.zeros((lmax+1,Nk,Nchan),dtype=np.complex128)
    wlk = np.zeros((lmax+1,Nk))
    for l in range(lmax+1):
        for ki in range(Nk):
            if np.any(kz[ki]*r_mpc < l): continue # These modes will already be very close to zero.
            bess_mat[l,ki,:] = r_mpc**2 * jl(l,kz[ki]*r_mpc)
            wlk[l,ki] = np.sum((jl(l,kz[ki]*r_mpc))**2)
    print "Bessel and weight matrix timing: ", time.time() - t0
    t0 = time.time()
    Tlmk = np.zeros((nlm, Nk),dtype=np.complex64)
    for i,l in enumerate(ls):
       Tlmk[i,:] = np.dot(bess_mat[l],alms[l,:])
    print "Tlmk timing: ", time.time() - t0

    if ret_flag:
        return Tlmk, wlk, (kz, ls, ms)
    else:
        return Tlmk, wlk

def box_pyramid_pspec(cube, radius, r_mpc, Zs, kz = None, Nkbins=100,error=False,cosmo=False):
    """
        Estimate the power spectrum in a "pyramid" appoximation:
            radius = in radians
            For each r_mpc, calculate the width of the projected region from the angular diameter distance.
            In the radial direction, calculate discrete fourier transform. In tangential directions, do FFT.
            TODO -- Reconsider the design here... I'm taking the change in L_x/L_y into account, but not the variation of L_z with angle?
    """
    Nx, Ny, Nz = cube.shape
    D_mpc = WMAP9.comoving_transverse_distance(Zs).to("Mpc").value
 
    if kz is None:
        dz = np.abs(r_mpc[-1] - r_mpc[0])/float(Nz)   #Mean spacing
        kz = np.fft.fftfreq(Nz,d=dz)*2*np.pi
    else:
        assert kz.size == r_mpc.size
    
    ## DFT in radial direction
    M = np.exp(np.pi * 2 * (-1j) * np.outer(kz,r_mpc) )
    _c = np.apply_along_axis(lambda x: np.dot(M,x),2, cube)
    
    ## 2D FFT in the transverse directions
    _c = np.fft.fft2(_c,axes=(0,1))
    
    ##kx and ky
    L = D_mpc * radius   # Transverse lengths in Mpc
    kx = np.array([np.fft.fftfreq(Nx,d=l/Nx) for l in L])*2*np.pi
    ky = np.array([np.fft.fftfreq(Ny,d=l/Ny) for l in L])*2*np.pi
    kx = np.moveaxis(np.tile(kx[:,:,np.newaxis],Ny),[1,2],[0,1])
    ky = np.moveaxis(np.tile(ky[:,:,np.newaxis],Nx),[1,2],[1,0])
    kz = np.tile(kz[np.newaxis,:],Nx*Ny).reshape(Nx,Ny,Nz)
    
    pk3d = np.abs(_c)**2/(Nx*Ny*Nz)
    
    results = bin_1d(pk3d,(kx,ky,kz),Nkbins=Nkbins,error=error)
    return results


def box_ls_pspec(cube, L, r_mpc, Nkbins=100, error=False, kz=None,cosmo=False):
    """ Estimate the 1D power spectrum for square regions with non-uniform distances using Lomb-Scargle periodogram in the radial direction."""
    Nx,Ny,Nz = cube.shape
    try:
        Lx, Ly, Lz = L
    except TypeError:
        Lx = L; Ly = L; Lz = L   # Assume L is a single side length, same for all
    dx, dy, dz = Lx/float(Nx), Ly/float(Ny), Lz/float(Nz)
    kx = np.fft.fftfreq(Nx,d=dx)*2*np.pi   #Mpc^-1
    ky = np.fft.fftfreq(Ny,d=dy)*2*np.pi   #Mpc^-1
    assert len(r_mpc) == Nz
    if kz is None:
        kz, powtest = LombScargle(r_mpc,cube[0,0,:]).autopower(nyquist_factor=1,normalization="psd")

    _cube = np.zeros((Nx,Ny,kz.size))
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            if kz is None:
                kz, power = LombScargle(r_mpc,cube[i,j,:]).autopower(nyquist_factor=1,normalization="psd")
            else:
                power = LombScargle(r_mpc, cube[i,j,:]).power(kz,normalization="psd")
            _cube[i,j] = np.sqrt(power)
    kz *= 2*np.pi
    _d = np.fft.fft2(_cube,axes=(0,1))
    kx = kx[kx>0]
    ky = ky[ky>0]
    Nx = np.sum(kx>0)
    Ny = np.sum(ky>0)
    _d = _d[1:Nx+1,1:Ny+1,:]     #Remove the nonpositive k-terms from the 2D FFT
    pk3d = np.abs(_d)**2/(Nx*Ny)
    results = bin_1d(pk3d,(kx,ky,kz),Nkbins=Nkbins, error=error)

    return results

def box_dft_pspec(cube, L, r_mpc=None, Nkbins=100,error=False,cosmo=False,return_3d=False):
    """ Estimate the 1D power spectrum for a rectilinear cube
            cosmo=Use cosmological normalization convention
    """
    print 'cube shape: ',cube.shape
    Nx,Ny,Nz = cube.shape
    try:
        Lx, Ly, Lz = L
    except TypeError:
        Lx = L; Ly = L
        if r_mpc is not None: Lz = r_mpc[-1] - r_mpc[0]  
        else: Lz = L          # Assume L is a single side length, same for all
    dx, dy, dz = Lx/float(Nx), Ly/float(Ny), Lz/float(Nz)
    dV = dx * dy * dz
    if isinstance(Lx, int): Lx = float(Lx)
    if isinstance(Ly, int): Ly = float(Ly)
    if isinstance(Lz, int): Lz = float(Lz)
    if cosmo:
        kfact=2*np.pi
        print Lx
        dV = dx*dy*dz
        print 'dV: ', dV
        pfact = 1/(Lx*Ly*Lz)# * (1./float(Nkbins))
    else:
       kfact=1
       dV = 1
       pfact = 1/float(Nx*Ny*Nz)
    kx = np.fft.fftfreq(Nx,d=dx)*kfact   #Mpc^-1
    ky = np.fft.fftfreq(Ny,d=dy)*kfact   #Mpc^-1
    kz = np.fft.fftfreq(Nz,d=dz)*kfact   #Mpc^-1

    if Nkbins == 'auto':
        mkx, mky, mkz = np.meshgrid(kx,ky,kz)
        ks = np.sqrt(mkx**2 + mky**2 + mkz**2).flatten()
        print np.around(ks, decimals=2)
        Nkbins = np.unique(np.around(ks,decimals=2)).size/2.
        #Nkbins=int(float(Nx*Ny*Nz)**(1/3.))
        print 'auto Nkbins=', Nkbins

    if not r_mpc is None:
        print "Nonuniform r"
        dz = np.abs(r_mpc[-1] - r_mpc[0])/float(Nz)   #Mean spacing
        kz = np.fft.fftfreq(Nz,d=dz)*kfact
    
        ## DFT in radial direction
        M = np.exp((-1j) * np.outer(kz,r_mpc) )
        _c = np.apply_along_axis(lambda x: np.dot(M,x),2, cube)
        _d = np.fft.fft2(_c,axes=(0,1))*dV   ## Multiply by the voxel size dV
    else:
        print 'Uniform r'
        _d = np.fft.fftn(cube)*dV

    pk3d = np.abs(_d)**2 * pfact
    if return_3d: return (kx,ky,kz), pk3d
    results = bin_1d(pk3d,(kx,ky,kz),Nkbins=Nkbins,error=error)

#    print 'Vol= {}, Lx={}, Ly={}, Lz={}, pfact= {:.5e},dV= {:.5e}'.format(Lx*Ly*Lz,Lx,Ly,Lz,pfact,dV)
    return results

def shell_pspec(shell, **kwargs):
    """
        r_pspec_sphere wrapper 
    """
    from eorsky import eorsky
    if not isinstance(shell, eorsky):
        raise TypeError("This function is for estimating power spectra of eorsky shells.")
    nside= shell.Nside
    hpx_shell = shell.shell
    r_mpc = shell.r_mpc
    freqs = shell.freqs

    ## Defaults:
    Nsec = 20 if "N_sections" not in kwargs else kwargs['N_sections']
    radius = 5 if "radius" not in kwargs else kwargs['radius']
    Nkbins = 'auto' if "Nkbins" not in kwargs else kwargs['Nkbins']
    method = 'fft' if 'method' not in kwargs else kwargs['method']
    cosmo = True if 'cosmo' not in kwargs else kwargs['cosmo']

    print "Settings: N_sections={}, Radius={:.2f} degree, Nkbins = {}, method = {}, cosmo={},Nside={}".format(Nsec,radius,Nkbins,method,cosmo,nside)

    return shell_project_pspec(hpx_shell,nside,radius,freqs=freqs,N_sections=Nsec,Nkbins=Nkbins,r_mpc = r_mpc,method=method,cosmo=cosmo)


def orthoslant_project(shell, center, radius, degrees=False):
    """
        Take a conical section of a sphere and bin into a cube for power spectrum estimation.
    """

    Npix, Nfreq = shell.shape
    Nside = hp.npix2nside(Npix)
    if degrees:
        radius *= np.pi/180.

    # Define xy grid by the pixel area and selection radius.


    radpix = hp.nside2resol(Nside)
    extent = 2*np.floor(radius/radpix).astype(int)

    orthogrid = np.zeros((extent,extent, Nfreq))

    # Get vectors, rotate so center is overhead, project vectors to get xy bins

    hpx_inds = hp.query_disc(Nside, center, 2*np.sqrt(2)*radius)   #TODO This can be replaced with a "query_polygon"
 
    vecs = hp.pix2vec(Nside, hpx_inds)

    dp, dt = hp.rotator.vec2dir(center, lonlat=True)
    rot = hp.rotator.Rotator(rot=(dp, dt, 0))
    vx, vy, vz = hp.rotator.rotateVector(rot.mat, vecs)

    ## Project onto the yz plane, having rotated so the x axis is at zenith
    xinds = np.floor(vy/radpix).astype(int) + extent/2
    yinds = np.floor(vz/radpix).astype(int) + extent/2    #Center on the grid

    boxinds = np.where((xinds < extent)&(xinds>-1)&(yinds < extent)&(yinds>-1))

    xinds = xinds[boxinds]
    yinds = yinds[boxinds]   #Radial selection was bigger than the grid
    hpx_inds = hpx_inds[boxinds]
    weights = np.ones((orthogrid.shape[0], orthogrid.shape[1]))
    #np.add.at(weights, (xinds,yinds), 1.0)
    #nz = np.where(weights > 0)
    #weights[nz] = 1/weights[nz]
    for fi in np.arange(Nfreq):
        np.add.at(orthogrid[:,:,fi],(xinds, yinds), shell[hpx_inds,fi])
        orthogrid[:,:,fi] *= weights

    return orthogrid



def shell_project_pspec(shell, nside, radius, dims=None,r_mpc=None, hpx_inds = None, N_sections=None,
                     freqs = None, kz=None, method='fft', Nkbins='auto',cosmo=False, error=False):
    """ Estimate the power spectrum of a healpix shell by projecting regions to Cartesian space.
            Shell = (Npix, Nfreq)  Healpix shell, or section of one.
            radius = (analogous to beam width)
            N_sections = Compute the power spectrum for this many regions (like snapshots in observing)
            dims (3) = Box dimensions in Mpc. If not available, check for freqs and use cosmology module
            method = Choose estimator method (lomb-scargle, dft, pyramid)
     """ 
    ## For now, pick a random point within the (assumed full) spherical shell, and get the pixels within distance radius around it.
    if hpx_inds is None: 
        Npix = hp.nside2npix(nside)
        hpx_inds = np.arange(Npix)
    else:
        Npix = hpx_inds.size
    if N_sections is None: N_sections=1

    radius *= np.pi/180.   #Deg2Rad

    if freqs is None and method=='lomb_scargle': raise ValueError(" Frequency data must be provided to use the Lomb-Scargle method ")

    if dims is None:
        if freqs is None:
            if dist is None: raise ValueError("Comoving distance or frequencies must be defined")
            Lx = 2*radius*dist  #Dist = comoving distance in Mpc
            Ly = Lx; Lz = Lx
        else:
            Zs = 1420./freqs - 1.
            rcomov = WMAP9.comoving_distance(Zs).to("Mpc").value
            Lz = np.max(rcomov) - np.min(rcomov)
            dist = WMAP9.comoving_transverse_distance(np.mean(Zs)).value    #Identical to comoving if Omega_k = 0
            Lx = 2*radius*dist
            Ly = Lx
        dims = [Lx,Ly,Lz]
    if method=='lomb_scargle' or method=='pyramid':
        try:
            r_mpc
            Zs
        except:
            Zs = 1420./freqs - 1.
            r_mpc = WMAP9.comoving_distance(Zs).to("Mpc").value

    print 'Dimensions: ',dims

    pk = None
    kbins = None
    errs = None

    ## Loop over sections, choosing a different center each time.
    ## Average power spectrum estimates from different parts of the sky.

    for s in range(N_sections):

        cent = hp.pix2vec(nside,np.random.choice(hpx_inds))
        inds = hp.query_disc(nside,cent,radius)

        print "Section, pixels: ",s, inds.shape
    
        cube = orthoslant_project(shell, cent, radius)

        #if s==0:
        #    np.savez('projected_cube',cube=cube,dims=dims)
 
        if method=='lomb_scargle':
            if kz is not None:
                results_i = box_ls_pspec(cube,dims,r_mpc,Nkbins=Nkbins,error=error,kz=kz,cosmo=cosmo)
            else:
                results_i = box_ls_pspec(cube,dims,r_mpc,Nkbins=Nkbins,error=error,cosmo=cosmo)
        elif method=='pyramid':
            results_i = box_pyramid_pspec(cube, radius, r_mpc, Zs, Nkbins=Nkbins,error=error,cosmo=cosmo)

        else:
            results_i = box_dft_pspec(cube,dims,Nkbins=Nkbins,r_mpc=r_mpc,error=error,cosmo=cosmo)
        if pk is None:
            pk = results_i[1]
            kbins = results_i[0]
            Nkbins = pk.size
            if error:
                errs = results_i[2]
        else:
            pk += results_i[1]
            kbins += results_i[0]
            if error:
                errs += results_i[2]
        print s, np.sum(np.isnan(results_i[1]))
    pk /= N_sections
    kbins /= N_sections

    if error:
        errs /= N_sections
        return kbins, pk, errs
    else:
        return kbins, pk
