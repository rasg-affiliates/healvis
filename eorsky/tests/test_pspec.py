### Tests for pspec.py

from eorsky import pspec_funcs, comoving_voxel_volume, comoving_radial_length, comoving_transverse_length
import nose.tools as nt
import numpy as np

def test_gaussian_box_fft():
    """
    Check the power spectrum of a gaussian box.
    """

    N = 256   #Vox per side
    L = 300.   #Mpc
    sig=2.0
    mu =0.0
    box = np.random.normal(mu, sig, (N,N,N))

    kbins, pk = pspec_funcs.box_dft_pspec(box, L, cosmo=True)

    dV = (L/float(N))**3

    tol = np.sqrt(np.var(pk))

    nt.assert_true(np.allclose(np.mean(pk), sig**2 * dV, atol=tol))

## TODO Download a 21cmFAST box and its calculated pspec an write a unittest to compare calculated vs. actual
## TODO Write a rotator test

def test_orthoslant_project():
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
    center = [np.sqrt(2)/2.,np.sqrt(2)/2.,0]
    #center = [1,0,0]

## For orientation tests, replace with a single point at center
#    import healpy
#    cind = healpy.vec2pix(Nside, *center)
#    shell *= 0.0
#    shell[cind] += 10

    orthogrid =  pspec_funcs.orthoslant_project(shell, center, 10, degrees=True)
    Nx, Ny, Nz = orthogrid.shape
    Lx = comoving_transverse_length(Z[Nfreq/2], Omega)*Nx
    Ly = Lx
    Lz = comoving_radial_length(Z[Nfreq/2], dnu)*Nz
    
    dV = comoving_voxel_volume(Z[Nfreq/2], dnu, Omega)
    print Lx*Ly*Lz/(Nx*Ny*Nz), dV

    kbins, pk = pspec_funcs.box_dft_pspec(orthogrid, [Lx, Ly, Lz], cosmo=True)
    print kbins, pk

    tol = np.sqrt(np.var(pk))

    nt.assert_true(np.allclose(np.mean(pk), sig**2 * dV, atol=tol))


def test_shell_pspec_dft():
    """
    Take a gaussian shell and confirm its power spectrum using shell_project_pspec.
    """

    select_radius = 10. #degrees
    N_sections = 5

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

    kbins, pk = pspec_funcs.shell_project_pspec(shell, Nside, select_radius, freqs=freqs, Nkbins=100, N_sections=N_sections, cosmo=True, method='dft')

    tol = np.sqrt(np.var(pk))

    nt.assert_true(np.allclose(np.mean(pk), sig**2 * dV, atol=tol))



#def shell_project_pspec(shell, nside, radius, dims=None,r_mpc=None, hpx_inds = None, N_sections=None,
#                             freqs = None, kz=None, dist=None, method='fft', Nkbins='auto',cosmo=False):



if __name__ == '__main__':
    test_shell_pspec_dft()

