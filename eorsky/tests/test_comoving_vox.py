## Plot comoving voxel volume vs. different parameters.

import numpy as np
from eorsky import comoving_voxel_volume

## TODO Plot volume vs (z,omega) for the mwa channel width. Logspaced pixel areas.
###     Plot volume vs (z, dnu) for different Nsides (multiple plots).

dnu_mwa = 0.080   #MHz
omegas = np.logspace(
