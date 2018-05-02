## Different power spectrum estimators for sphere or Cartesian cubes.

import scipy.stats, numpy as np, healpy as hp, time
from astropy.cosmology import WMAP9
from astropy.stats import LombScargle
from scipy.special import spherical_jn as jl   ## Requires scipy version 0.18.0 at least


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

def pyramid_pspec(cube, radius, r_mpc, Zs, kz = None, Nkbins=100,sigma=False,cosmo=False):
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
    
    results = bin_1d(pk3d,(kx,ky,kz),Nkbins=Nkbins,sigma=sigma)
    return results


def ls_pspec_1D(cube, L, r_mpc, Nkbins=100, sigma=False, kz=None,cosmo=False):
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
    results = bin_1d(pk3d,(kx,ky,kz),Nkbins=Nkbins, sigma=sigma)

    return results

def r_pspec_1D(cube, L, r_mpc=None, Nkbins=100,sigma=False,cosmo=False,return_3d=False):
    """ Estimate the 1D power spectrum for a rectilinear cube
            cosmo=Use cosmological normalization convention
    """
    print 'cube shape: ',cube.shape
    Nx,Ny,Nz = cube.shape
    if Nkbins == 'auto':
        #Nkbins=int(float(Nx*Ny*Nz)**(1/3.))
        Nkbins=min([Nx,Ny,Nz])
        print 'auto Nkbins=', Nkbins
    try:
        Lx, Ly, Lz = L
    except TypeError:
        Lx = L; Ly = L
        if r_mpc is not None: Lz = r_mpc[-1] - r_mpc[0]  
        else: Lz = L          # Assume L is a single side length, same for all
    dx, dy, dz = Lx/float(Nx), Ly/float(Ny), Lz/float(Nz)
    dV = dx * dy * dz
    if cosmo:
        kfact=2*np.pi
        dV = dx*dy*dz
        pfact = 1/(Lx*Ly*Lz) * (1./float(Nkbins))
    else:
       kfact=1
       dV = 1
       pfact = 1/float(Nx*Ny*Nz)
    kx = np.fft.fftfreq(Nx,d=dx)*kfact   #Mpc^-1
    ky = np.fft.fftfreq(Ny,d=dy)*kfact   #Mpc^-1
    kz = np.fft.fftfreq(Nz,d=dz)*kfact   #Mpc^-1

    if not r_mpc is None:
        print "Nonuniform z"
        dz = np.abs(r_mpc[-1] - r_mpc[0])/float(Nz)   #Mean spacing
        kz = np.fft.fftfreq(Nz,d=dz)*kfact
    
        ## DFT in radial direction
        M = np.exp((-1j) * np.outer(kz,r_mpc) )
        _c = np.apply_along_axis(lambda x: np.dot(M,x),2, cube)
        _d = np.fft.fft2(_c,axes=(0,1))*dV   ## Multiply by the voxel size dV
    else:
        print 'Uniform z'
        _d = np.fft.fftn(cube)*dV
    pk3d = np.abs(_d)**2 * pfact
    if return_3d: return (kx,ky,kz), pk3d
    results = bin_1d(pk3d,(kx,ky,kz),Nkbins=Nkbins,sigma=sigma)

    print 'Vol= {}, Lx={}, Ly={}, Lz={}, pfact= {:.5e},dV= {:.5e}'.format(Lx*Ly*Lz,Lx,Ly,Lz,pfact,dV)
    return results

def shell_pspec(shell, **kwargs):
    """
        r_pspec_sphere wrapper 
    """
    from eorsky import eorsky
    if not isinstance(shell, eorsky):
        raise TypeError("This function is for estimating power spectra of eorsky shells.")
    nside= shell.nside
    hpx_shell = shell.hpx_shell
    r_mpc = shell.r_mpc
    freqs = shell.freqs

    ## Defaults:
    Nsec = 20 if "N_sections" not in kwargs else kwargs['N_sections']
    radius = 10 if "radius" not in kwargs else kwargs['radius']
    Nkbins = 'auto' if "N_kbins" not in kwargs else kwargs['N_kbins']
    method = 'fft' if 'method' not in kwargs else kwargs['method']
    cosmo = True if 'cosmo' not in kwargs else kwargs['cosmo']

    print "Settings: N_sections={}, Radius={:.2f} degree, Nkbins = {}, method = {}, cosmo={},Nside={}".format(Nsec,radius,Nkbins,method,cosmo,nside)

    return r_pspec_sphere(hpx_shell,nside,radius,freqs=freqs,N_sections=Nsec,Nkbins=Nkbins,r_mpc = r_mpc,method=method,cosmo=cosmo)
    

def r_pspec_sphere(shell, nside, radius, dims=None,r_mpc=None, hpx_inds = None, N_sections=None,
                     freqs = None, kz=None, dist=None, method='fft', Nkbins='auto',cosmo=False):
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
    errs = None

    ## Loop over sections, choosing a different center each time.
    ## Average power spectrum estimates from different parts of the sky.
    Xextent = int(2*np.pi/hp.nside2resol(nside))    ## The projector does the whole map at once. This roughly preserves pixel area.
    Yextent = int(np.pi/hp.nside2resol(nside))    ## The projector does the whole map at once. This roughly preserves pixel area.
    ctp = hp.projector.CartesianProj(xsize=Xextent,ysize=Yextent)
    ctp.set_flip("astro")
    fullcube = np.zeros((Yextent,Xextent,shell.shape[1]))
    fun = lambda x,y,z: hp.pixelfunc.vec2pix(nside,x,y,z,nest=False)
    for i in range(shell.shape[1]):
        fullcube[:,:,i] = ctp.projmap(shell[:,i],fun)
    for s in range(N_sections):

        cent = hp.pix2vec(nside,np.random.choice(hpx_inds))
        inds = hp.query_disc(nside,cent,radius)
        vecs = hp.pixelfunc.pix2vec(nside, inds)
	
        print "Section, pixels: ",s, inds.shape
    
    #    dt, dp = hp.rotator.vec2dir(cent,lonlat=True)
    
        #Estimate X extent by the number of pixels in the selection. 2*radius/pixsize
#        Xextent = int(2*radius/hp.nside2resol(nside))

        i,j   = ctp.xy2ij(ctp.vec2xy(vecs[0],vecs[1],vecs[2]))
        imin, imax = min(i), max(i)
        jmin, jmax = min(j), max(j)
 
        ## Construct a projected cube:
    
        cube = np.zeros((imax-imin,jmax-jmin,shell.shape[1]))
        for fi in range(shell.shape[1]):
            cube[:,:,fi] = np.roll(fullcube[:,:,fi],(-imin,-jmin),(0,1))[:imax-imin,:jmax-jmin]

        if s==0:
            np.savez('projected_cube',cube=cube,dims=dims)
 
        if method=='lomb_scargle':
            if kz is not None:
                kbins, pki, errsi = ls_pspec_1D(cube,dims,r_mpc,Nkbins=Nkbins,sigma=True,kz=kz,cosmo=cosmo)
            else:
                kbins, pki, errsi = ls_pspec_1D(cube,dims,r_mpc,Nkbins=Nkbins,sigma=True,cosmo=cosmo)
        elif method=='pyramid':
            kbins,pki, errsi = pyramid_pspec(cube, radius, r_mpc, Zs, Nkbins=Nkbins,sigma=True,cosmo=cosmo)

        else:
            kbins, pki, errsi = r_pspec_1D(cube,dims,Nkbins=Nkbins,r_mpc=r_mpc,sigma=True,cosmo=cosmo)
        if pk is None:
            pk = pki
            errs = errsi
            Nkbins = pk.size
        else:
            pk += pki
            errs += errsi

    pk /= N_sections
    errs /= N_sections
    return kbins, pk, errs

    
