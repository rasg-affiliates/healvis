import numpy as np
import multiprocessing as mp

class mparray(np.ndarray):

    """
    A multiprocessing RawArray accessible with numpy array slicing.
    """

    def __init__(self, *args, **kwargs):
        super(mparray, self).__init__(*args, **kwargs)
        size = np.prod(self.shape)
        ctype = np.sctype2char(self.dtype)
        arr = mp.RawArray(ctype, size)
        self.data = arr
        self.reshape(self.shape)
