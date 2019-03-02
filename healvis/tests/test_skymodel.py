from healvis import skymodel
from astropy.cosmology import Planck15
import nose.tools as nt
import numpy as np


def verify_update(sky_obj):
    """
        Check that the frequencies/redshifts/distances are consistent.
        If healpix, check that Npix/indices/Nside are consistent
    """

    if sky_obj.data is not None:
        Nside = sky_obj.Nside
        indices = sky_obj.indices
        Npix = sky_obj.Npix

        if Npix == 12 * Nside**2:
            nt.assert_true(all(indices == np.arange(Npix)))
        else:
            nt.assert_true(Npix == indices.size)
    if sky_obj.freq_array is not None:
        freqs = sky_obj.freq_array
        Nfreq = sky_obj.Nfreqs
        Z = sky_obj.Z_array
        r_mpc = sky_obj.r_mpc

        Zcheck = 1420. / freqs - 1.
        Rcheck = Planck15.comoving_distance(Zcheck).to("Mpc").value
        nt.assert_true(all(Zcheck == Zcheck))
        nt.assert_true(all(Rcheck == Rcheck))
    nt.assert_true(len(sky_obj._updated) == 0)  # The _updated list should be cleared


def test_update():
    Nfreqs = 100
    freq_array = np.linspace(167.0, 177.0, Nfreqs)
    sky = skymodel(Nside=64, freq_array=freq_array)
    verify_update(sky)
    sky.ref_chan = 50
    sky.make_flat_spectrum_shell(sigma=2.0)
    verify_update(sky)


def test_flat_spectrum():
    Nfreqs = 100
    freq_array = np.linspace(167.0, 177.0, Nfreqs)
    sky = skymodel(Nside=64, freq_array=freq_array, ref_chan=50)
    sky.make_flat_spectrum_shell(sigma=2.0)
