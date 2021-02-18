# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

from __future__ import absolute_import, division, print_function

import numpy as np
import multiprocessing as mp
import time

from healvis import utils


def test_mparray():
    # This runs parallel processes, each of which receives
    # a shared mparray and a non-shared numpy ndarray.
    # Each process writes to each of these arrays.

    # The expected behavior is that the values written to the shared
    # array should be accessible to other processes. The ndarray is not shared,
    # so it is copied into each process and the written values are local.

    # This therefore confirms that the mparray is shared among processes

    Nprocs = 5

    def do_a_thing(arr, nsarr, ind):
        arr[ind] = ind * 2
        nsarr[ind] = ind * 3
        if ind == 1:
            time.sleep(1)
            assert np.all([arr[i] == 2 * i for i in range(Nprocs)])
            assert np.all([nsarr[i] == 100 for i in range(Nprocs)
                          if not i == 1])

    procs = []

    mpah = utils.mparray(Nprocs, dtype=int)
    mpah[()] = np.ones(Nprocs) * 100
    ah = np.ones(Nprocs) * 100

    for pid in range(Nprocs):
        p = mp.Process(target=do_a_thing, args=(mpah, ah, pid))
        p.start()
        procs.append(p)

    while np.any([p.is_alive() for p in procs]):
        continue


def assert_raises_message(exception_type, message, func, *args, **kwargs):
    """
    Check that the correct error message is raised.
    """
    nocatch = kwargs.pop('nocatch', False)
    if nocatch:
        func(*args, **kwargs)

    with pytest.raises(exception_type) as err:
        func(*args, **kwargs)

    try:
        assert message in str(err.value)
    except AssertionError as excp:
        print("{} not in {}".format(message, str(err.value)))
        raise excp
