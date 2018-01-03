## Test the effect of FHD's healpix interpolation on the power spectrum.

from eorsky import r_pspec_1D, eorsky
from scipy.io import readsav
import numpy as np
from astropy.cosmology import WMAP9 as cosmo

f = readsav("../data/bubble_model_stokes.sav")
stok = f['model_stokes_arr']
stoke = np.swapaxes(np.array([s for s in stok]),0,2)  # (512, 512, 384)

degpix=0.223812
N = 512
df = 80000.0/1e6  #MHz
Lx= N*degpix*(np.pi/180.)*cosmo.angular_diameter_distance(7.0).to("Mpc").value
Ly = Lx 
Lz = Lx  # For now

ks, pk = r_pspec_1D(stoke,Lx)
