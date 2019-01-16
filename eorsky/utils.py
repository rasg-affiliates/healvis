import numpy as np
import multiprocessing as mp

## TODO --- replace this. numpy no longer supports assignment to the data attribute:
##  https://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing

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
