## Class to handle EoR simulation data
##	> Read in hdf5 files and binary files for cubes and shells
##	> Convert between rectilinear cubes and tiled healpix shells.
##	> Estimate 1D power spectrum from rectilinear cube.
##	> (eventually) Perform spherical Bessel transform to estimate pspec from healpix shell directly


import numpy as np
from astropy.cosmology import WMAP9 as cosmo
import scipy.stats
from astropy import units
from astropy.constants import c
import healpy as hp
from numpy import pi
import h5py, sys


freq_21 = 1420.

class eorsky(object):

    N = 256   # Npixels per side of rect_cube
    L = 500   # rect_cube side length in Mpc
    hpx_shell = None
    rect_cube = None
    nside = 256
    Npix = None
    pixsize = None    # Pixel size in steradians
    Nfreq = None
    freqs = None    # MHz
    Z = None
    updated = []      #Track which attributes are changed
    hpx_inds = None
    r_mpc = None         # Shell radii in Mpc
    pk = None       ## 1D power spectrum
    kbins = None

    def __init__(self, **kwargs):
        for key, value in kwargs.iteritems():
            if hasattr(self, key): setattr(self, key, value)
        self.update()

    def __setattr__(self,name,value):
        if self.updated is None: self.updated = []
        self.updated.append(name)
        super(eorsky, self).__setattr__(name, value)

    def update(self):
        """ Make Z, freq, healpix params, and rectilinear params consistent.  """
        if 'freqs' in self.updated:
            if 'Z' in self.updated:
                assert np.all(self.Z == freq_21/self.freqs - 1.)
            else:
                self.Z = 1420./self.freqs - 1.
                self.r_mpc = cosmo.comoving_distance(self.Z).to("Mpc").value
            if "Nfreq" in self.updated:
                assert self.Nfreq == self.freqs.size
            else:
                self.Nfreq = self.freqs.size
        if 'Npix' in self.updated:
            assert self.hpx_inds.size == self.Npix
            if 'nside' in self.updated:
                assert self.nside == hp.npix2nside(self.Npix)
            else:
                self.nside = hp.npix2nside(self.Npix)
        if self.Npix is None: self.Npix = hp.nside2npix(self.nside)
        
        if "nside" in self.updated:
            self.pixsize = hp.nside2pixarea(self.nside) * (4*pi)/(2*pi)**2   # Approximately go from square radians to steradians
        
        self.updated = []
 
    def read_hdf5(self,filename):
        infile = h5py.File(filename)
        self.freqs = infile['spectral_info/freq'][()]/1e6  # Hz -> MHz
        self.hpx_shell = infile['spectral_info/spectrum'][()]
        self.Npix, self.Nfreq = infile['spectral_info/spectrum'].shape
        self.hpx_inds = np.arange(self.Npix)
        infile.close()
        self.update()

    def write_hdf5(self, filename):
        with h5py.File(filename, 'w') as fileobj:
            freqs_Hz = self.freqs * 1e6
            hdr_grp = fileobj.create_group('header')
            hdr_grp['units'] = "K"
            hdr_grp['is_healpix'] = 1
            spec_group = fileobj.create_group('spectral_info')
            freq_dset = spec_group.create_dataset('freq', data=freqs_Hz, compression='gzip', compression_opts=9)
            freq_dset.attrs['units'] = 'Hz'
            spectrum_dset = spec_group.create_dataset('spectrum', data=self.hpx_shell, compression='gzip', compression_opts=9)

    def unslice(self,**kwargs):
        params = ["N","L","nside","hpx_inds"]
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
        Vx, Vy, Vz = hp.pix2vec(self.nside, self.hpx_inds)
        vecs = np.vstack([Vx,Vy,Vz]).T    #(npix, 3)

        cube_sel = -1
        if 'cube_sel' in kwargs:
            cube_sel = int(kwargs['cube_sel'])        #Select a particular cube repetition
                                                      #Choose the cube_sel'th nonzero cube
            Ncubes = (2*np.floor(np.max(self.r_mpc)/L))**3   #Right?
            print "Selecting cube ", cube_sel," of ",Ncubes

        for i in freq_inds:
            print i
            r = self.r_mpc[i]
            inds = np.floor((r*vecs)%(L)/dx).astype(int)   # (npix, 3)
            inds = (inds[:,0],inds[:,1],inds[:,2])
            if cube_sel>0:
                cube_inds = np.floor(((r*vecs)/L)).astype(int)
                import IPython;
                IPython.embed()
                sys.exit()
            np.add.at(cube,inds,self.hpx_shell[:,i])
        self.rect_cube = cube

    def pspec_1D(self,Nkbins=100):
        """ Estimate the 1D power spectrum for a rectilinear cube """
        L, N = self.L, self.N
        dx = L/float(N)
        assert self.rect_cube is not None
        _d = np.fft.fftn(self.rect_cube)
        ps3d = np.abs(_d*_d.conj())
        ps1d = ps3d.ravel()
        kvals = np.fft.fftfreq(N,d=dx)   #Mpc^-1
        kx, ky, kz = np.meshgrid(kvals,kvals,kvals)
        k = np.sqrt(kx**2 + ky**2 + kz**2)
        means, bins, binnums = scipy.stats.binned_statistic(k.ravel(),ps1d,statistic='mean',bins=Nkbins)
        self.pk = means/N**3
        self.kbins  = (bins[1:] + bins[:-1])/2.


if __name__ == '__main__':
  #  freqs = np.linspace(100,200,203)
  #  Nfreq = 203
  #  e = eorsky(freqs=freqs,Nfreq=Nfreq)
    e = eorsky()
    e.read_hdf5('/users/alanman/data/alanman/BubbleCube/TiledHpxCubes/gaussian_cube.hdf5')
    e.unslice(freq_inds=[10],cube_sel=2)

    import IPython; IPython.embed()
