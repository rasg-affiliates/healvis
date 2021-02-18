#!/bin/env python

import subprocess
import numpy as np
import sys

Nbeams = 75
beam_min = 15.0
beam_max = 50.0

fwhm = np.linspace(beam_min, beam_max, Nbeams)

Nensemb = 49
for i in range(Nensemb):
    for fw in fwhm:
        print(fw)
        subprocess.call(
            [
                "sbatch",
                "vis_param_sim.py",
                "configs/obsparam_multibl.yaml",
                "-b",
                str(fw),
            ]
        )
