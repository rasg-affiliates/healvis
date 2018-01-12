# coding: utf-8

import numpy as np
from eorsky import slk_calc

f = np.load('saved_tlmk.npz')
tlmk = f['tlmk']
wlk = f['wlk']
kz, ls, ms = f['basis']
Slk_test = slk_calc(tlmk,wlk,ls,ms)
