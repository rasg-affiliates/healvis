import numpy as np
from astropy.cosmology import WMAP9 as cosmo
import scipy.stats
from astropy import units
from astropy.constants import c
from scipy.io import readsav
import healpy as hp
import h5py, sys
import warnings

pi =  np.pi
freq_21 = 1420.

class eorsky(object):

    N = 256   # Npixels per side of box, default 256
    L = 500   # box side length in Mpc, default 500
    shell = None

    box = None
    boxes = None
    box_file_list = None
    box_z_list = None
    

    Nside = 256
    Npix = None
    pixsize = None    # Pixel size in steradians
    Nfreq = None
    freqs = None    # MHz
    Z = None
    updated = []      #Track which attributes are changed
    hpx_inds = None
    r_mpc = None         # Shell radii in Mpc
    ref_pk = None          ## Reference 1D power spectrum
    ref_ks = None

    est_pk = None   # Estimated 1D power spectrum
    est_ks = None

    def __init__(self, **kwargs):
        for key, value in kwargs.iteritems():
            print key, value
            if hasattr(self, key): setattr(self, key, value)
        self.update()

    def __setattr__(self,name,value):
        if self.updated is None: self.updated = []
        self.updated.append(name)
        super(eorsky, self).__setattr__(name, value)

    def make_gaussian_cube(self, N, L, mean=0.0, var=1.0):
        """ Make a gaussian cube with the given structure """
        self.N = N
        self.L = L
        self.box = np.random.normal(mean,var,((N,N,N)))
        self.update()

    def make_gaussian_shell(self, Nside, freqs, mu=0.0, sigma=1.0):
        """ Make a gaussian shell with the given structure """
        Npix = hp.nside2npix(Nside)
        try:
            Nfreq = freqs.size
        except:
            freqs = np.array(freqs); Nfreq = freqs.size
        self.Nfreq = Nfreq; self.Npix = Npix
        self.Nside = Nside
        self.freqs = freqs
        self.shell = np.random.normal(mu,sigma,((Npix,Nfreq)))
        self.hpx_inds = np.arange(Npix)
        self.update()


    def update(self):
        """ Make Z, freq, healpix params, and rectilinear params consistent. 
            Assume that whatever parameter was just changed has priority over others.
            If two parameters from the same group change at the same time, confirm that they're consistent.
        """

        hpx_params = ['Nside','Npix','shell','hpx_inds']
        z_params   = ['Z', 'freqs','Nfreq','r_mpc']
        r_params   = ['L','N','box']

        ud = np.unique(self.updated)
        for p in ud:
            if p == 'boxes':
                if self.box == None: self.box = self.boxes[0]
            if p == 'freqs':
                self.Z = 1420./self.freqs - 1.
                self.r_mpc = cosmo.comoving_distance(self.Z).to("Mpc").value
                self.Nfreq = self.freqs.size
            if p == 'Nside':
                if self.Npix is None: self.Npix = hp.nside2npix(self.Nside)
                if self.hpx_inds is None: self.hpx_inds = np.arange(self.Npix)
            if p == 'box':
                self.N = self.box.shape[0]
            if p == 'Nside':
                if 'hpx_inds' not in ud:
                    if 'Npix' not in ud: self.Npix = hp.nside2npix(self.Nside)
                    self.hpx_inds = np.arange(self.Npix)
            if p == 'hpx_inds':
                self.Npix = self.hpx_inds.size
        self.updated = []

    def read_pspec_text(self,filename):
        """ Read in a text file of power spectrum values """
        ref_spectrum = np.genfromtxt(filename)
        self.ref_ks = ref_spectrum[:,0]
        self.ref_pk = ref_spectrum[:,1]


    def read_pspec_eppsilon(self,filename):
        """ Read from an eppsilon idlsave file. Ensure it ends in 1dkpower.idsave """
        assert filename.endswith("1dkpower.idlsave")
        f = readsav(filename)
        self.ref_ks = (f['k_edges'][1:] + f['k_edges'][:-1])/2.
        self.ref_pk = f['power']


    def read_hdf5(self, filename, chan_range=None):
        print 'Reading: ', filename
        infile = h5py.File(filename)
        skymodel = 'spectral_info' in infile.keys()
        transpose = False
        if skymodel:
            freqs_key = 'spectral_info/freq'
            shell_key = 'spectral_info/spectrum'
        else:
            freqs_key = 'specinfo/freqs'
            shell_key = 'skyinfo/surfaces'
            transpose = True
        if chan_range is None:
            chan_range = [0, infile[freqs_key].shape[0]]
        c0,c1 = chan_range
        self.freqs = infile[freqs_key][c0:c1]/1e6  # Hz -> MHz
        if not transpose:
            self.shell = infile[shell_key][:,c0:c1]
        else:
            self.shell = infile[shell_key][c0:c1,:].T
        self.Npix, self.Nfreq = self.shell.shape
        self.hpx_inds = np.arange(self.Npix)
        self.Nside = hp.npix2nside(self.Npix)
        infile.close()
        self.update()

    def read_bin(self, filename, L=None, dtype=float):
        print 'Reading: ', filename
        self.box = np.fromfile(filename,dtype=dtype)
        if L is not None: self.L = L
        self.N = self.box.shape[0]

    def write_hdf5(self, filename):
        with h5py.File(filename, 'w') as fileobj:
            freqs_Hz = self.freqs * 1e6
            hdr_grp = fileobj.create_group('header')
            hdr_grp['units'] = "K"
            hdr_grp['is_healpix'] = 1
            spec_group = fileobj.create_group('spectral_info')
            freq_dset = spec_group.create_dataset('freq', data=freqs_Hz, compression='gzip', compression_opts=9)
            freq_dset.attrs['units'] = 'Hz'
            spectrum_dset = spec_group.create_dataset('spectrum', data=self.shell, compression='gzip', compression_opts=9)

    def cube_inds(self,which=0):
        """ Group healpix indices and radii by which box repetition they're in. """
        N, L = self.N, np.float(self.L)
        r_mpc = self.r_mpc
        Vx, Vy, Vz = hp.pix2vec(self.Nside, self.hpx_inds)
        l = np.floor(np.outer(Vx, r_mpc)/L).astype(int)
        m = np.floor(np.outer(Vy, r_mpc)/L).astype(int)
        n = np.floor(np.outer(Vz, r_mpc)/L).astype(int)
        offset = (-1)*np.min([l,m,n]) + 1
        l += offset
        m += offset
        n += offset
        box_inds = l + m * offset + n * offset**2
        _, inv = np.unique(box_inds, return_inverse=True)
        inv = inv.reshape((self.Npix,self.Nfreq))
        print "Choosing cube ", which," of ", len(_)
        return np.where(inv == which)

    def set_freqs(self, instr):
        """ Set frequencies to those of a known instrument. """
#        if instr.lower() == 'mwa': freqs = np.linspace(182.95-30.72/2.,182.95+30.72/2., 384)
#        if instr.lower() == 'paper': freqs = np.linspace(100,200,203)
#        if instr.lower() == 'hera': freqs = np.linspace(100,200,1024)
#        self.freqs = freqs
        self.freqs = common_freqs(instr)
        self.update()

    def select_range(self, **kwargs):
        assert self.freqs is not None
        if 'z' in kwargs:
            z0,z1 = kwargs['z']
            imin, imax = np.argmin(np.abs(self.Z - z1)), np.argmin(np.abs(self.Z - z0))
            self.freqs = self.freqs[imin:imax]
        if 'f' in kwargs:
            f0,f1 = kwargs['f']
            imin, imax = np.argmin(np.abs(self.freqs - f0)), np.argmin(np.abs(self.freqs - f1))
            self.freqs = self.freqs[imin:imax]
        if 'chan' in kwargs:
            imin,imax = int(kwargs['chan'][0]),int(kwargs['chan'][1])
            self.freqs = self.freqs[imin:imax]
            self.update()
        if not self.shell is None:
            self.shell = self.shell[:,imin:imax]
        self.update()
            

    def slice(self, **kwargs):
        """ Take a box and slice it into a healpix shell """

        assert self.Nside is not None
        assert self.N is not None
        assert self.L is not None
        if self.Npix is None: self.Npix = hp.nside2npix(self.Nside)
        if self.hpx_inds is None: self.hpx_inds = np.arange(self.Npix)
        if 'cosmo' in kwargs: cosmo=kwargs['cosmo']

        shell = np.zeros((self.Npix,self.Nfreq))

        Vx, Vy, Vz = hp.pix2vec(self.Nside, self.hpx_inds)
        vecs = np.vstack([Vx,Vy,Vz])    #(3,npix)
        L = float(self.L)
        dx = L / float(self.N)

        cube = self.box
        print "Slicing"
        for i in range(self.Nfreq):
            print 'channel ',i
            r = self.r_mpc[i]
            XYZmod = (vecs * r) % L
            l,m,n = (XYZmod / dx).astype(int)
            ## If using cosmological conventions, scale by relative voxel volumes.
            if cosmo:
                hpx_omega = 4*np.pi/float(self.Npix)
                scale = comoving_voxel_volume(self.Z[i], dnu, hpx_omega)/ dx**3
                print scale
            shell[:,i] += cube[l,m,n]*scale
        self.shell = shell
        self.update()

    def unslice(self,**kwargs):
        params = ["N","L","Nside","hpx_inds"]
        for key, value in kwargs.iteritems():
            if hasattr(self, key) and key in params: setattr(self, key, value)

        if 'freq_sel' in kwargs:
            f_min, f_max = kwargs['freq_sel']    #Select a range of frequencies. Passed in as a list [f_min, f_max]
            freq_arr = self.freqs
            i = np.argmin(np.abs(freq_arr - f_min))
            f = np.argmin(np.abs(freq_arr - f_max))
            freq_arr = freq_arr[i:f]
            freq_inds = np.arange(i,f,1)
        elif 'freq_ind_range' in kwargs:
            i,f = kwargs['freq_inds']
            freq_inds = np.arange(i,f,1)
            freq_arr = self.freqs[freq_inds]
        elif 'freq_inds' in kwargs:
            freq_inds = kwargs['freq_inds']
            freq_arr = self.freqs[freq_inds]
        else:
            freq_arr = self.freqs
            freq_inds = np.arange(self.Nfreq)
 
        N,L = self.N, np.float(self.L)
        dx = L/float(N)
        cube = np.zeros((N,N,N))
        Vx, Vy, Vz = hp.pix2vec(self.Nside, self.hpx_inds)
        vecs = np.vstack([Vx,Vy,Vz]).T    #(npix, 3)

        cube_sel = -1
        if 'cube_sel' in kwargs:
            cube_sel = int(kwargs['cube_sel'])        #Select a particular cube repetition
            box = self.cube_inds(cube_sel)
        shell = self.shell
        ones_shell = np.ones_like(shell)
        weights = np.zeros_like(cube)
        for i in freq_inds:
            print i
            r = self.r_mpc[i]
            inds = np.floor((r*vecs)%(L)/dx).astype(int)   # (npix, 3)
            if 'cube_sel' in kwargs:
                finder = np.where(box[1] == i)
                hpi = box[0][finder]
                if len(hpi) == 0 : continue
                l,m,n = inds[hpi,:].T
                cart_inds = (l,m,n)
                np.add.at(cube,cart_inds,shell[hpi,i])
                np.add.at(weights,cart_inds,ones_shell[hpi,i])
            else:
                cart_inds = (inds[:,0],inds[:,1],inds[:,2])
                np.add.at(cube,cart_inds,shell[:,i])
                np.add.at(weights,cart_inds,ones_shell[:,i])
        zeros = np.where(weights == 0.0)
        weights = 1/weights
        weights[zeros] = 0.0
        self.box = cube*weights


def common_freqs(instr):
    """ Set frequencies to those of a known instrument. """
    if instr.lower() == 'mwa': freqs = np.linspace(167075000.0,197715000.0, 384)
    if instr.lower() == 'paper': freqs = np.linspace(100,200,203)
    if instr.lower() == 'hera': freqs = np.linspace(100,200,1024)
    return freqs


def comoving_voxel_volume(z,dnu,omega):
    """
        From z, dnu, and pixel size (omega), get voxel volume.
        dnu = MHz
        Omega = sr
    """
    if isinstance(z,np.ndarray):
        if isinstance(omega,np.ndarray):
            z, omega = np.meshgrid(z,omega)
        elif isinstance(dnu,np.ndarray):
            z, dnu = np.meshgrid(z,dnu)
    elif isinstance(dnu, np.ndarray) and isinstance(omega, np.ndarray):
        dnu, omega = np.meshgrid(dnu, omega)
    nu0 = 1420./(z + 1) - dnu/2.
    nu1 = nu0 + dnu
    dz = 1420.*(1/nu0 - 1/nu1)
    vol = cosmo.differential_comoving_volume(z).value*dz*omega
    return vol


def comoving_radial_length(z,dnu):
    """
        What is the comoving radial length for a given dnu?
    """
    nu0 = 1420./(z + 1) - dnu/2.
    nu1 = nu0 + dnu
    z0, z1 = 1420./nu0 - 1, 1420./nu1 - 1
    L = abs(np.diff(cosmo.comoving_distance([z0,z1]).value)[0])
    return L


def comoving_transverse_length(z,Omega):
    """
        What is the comoving transverse length for a given pixel area? (in Mpc)
    """
    r = cosmo.comoving_distance(z).value
    return np.sqrt(r**2*Omega)
