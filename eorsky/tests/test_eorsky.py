
from eorsky import eorsky
from astropy.cosmology import WMAP9
import nose.tools as nt
import numpy as np

## Tests
# Selection test: Make a selection. Assert that the length is changed and verify update.
# common_freqs:
#       Store the actual frequency arrays for HERA/MWA in an npz file and verify that these are correct.
# Slice test?
#       >> Slice a cube of indices and see where they end up on the map.
#       >> Slice a gaussian box without voxel scaling and confirm preservation of 1 point stats.


def verify_update(sky_obj):
    """
        Check that the frequencies/redshifts/distances are consistent.
        If healpix, check that Npix/hpx_inds/Nside are consistent
    """

    if sky_obj.shell is not None:
        Nside = sky_obj.Nside
        hpx_inds = sky_obj.hpx_inds
        Npix = sky_obj.Npix

        if Npix == 12*Nside**2:
            nt.assert_true(all(hpx_inds == np.arange(Npix)))
        else:
            nt.assert_true(Npix == hpx_inds.size)
    if sky_obj.freqs is not None:
        freqs = sky_obj.freqs
        Nfreq = sky_obj.Nfreq
        Z = sky_obj.Z
        r_mpc = sky_obj.r_mpc

        Zcheck = 1420./freqs - 1.
        Rcheck = WMAP9.comoving_distance(Zcheck).to("Mpc").value
        nt.assert_true(all(Zcheck == Zcheck))
        nt.assert_true(all(Rcheck == Rcheck))
    print sky_obj.updated
    nt.assert_true(len(sky_obj.updated) == 0)   #The updated list should be cleared


def test_update():
    sky = eorsky()
    Nside=128
    Nfreq = 100
    freqs = np.linspace(167.0, 177.0, Nfreq)
    sky.make_gaussian_shell(Nside, freqs, mu=0.0, sigma=2.0)
    verify_update(sky)


def test_select():
    sky = eorsky()
    Nside=128
    Nfreq = 100
    freqs = np.linspace(167.0, 177.0, Nfreq)
    sky.make_gaussian_shell(Nside, freqs, mu=0.0, sigma=2.0)
    Nsel = 10
    z0 = sky.Z[Nsel]
    zlast = sky.Z[0]
    sky.select_range(z=(z0,zlast))
    nt.assert_equal(sky.freqs.size, Nsel)
    verify_update(sky)


