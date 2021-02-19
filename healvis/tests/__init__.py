# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

import os
import pytest

from healvis.data import DATA_PATH

TESTDATA_PATH = os.path.join(DATA_PATH, "temporary_test_data/")


def assert_raises_message(exception_type, message, func, *args, **kwargs):
    """
    Check that the correct error message is raised.
    """
    nocatch = kwargs.pop("nocatch", False)
    if nocatch:
        func(*args, **kwargs)

    with pytest.raises(exception_type) as err:
        func(*args, **kwargs)

    try:
        assert message in str(err.value)
    except AssertionError as excp:
        print("{} not in {}".format(message, str(err.value)))
        raise excp
