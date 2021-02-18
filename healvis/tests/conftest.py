# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

"""Testing environment setup and teardown for pytest."""
import os
import shutil

import pytest
from astropy.time import Time
from astropy.utils import iers
from urllib.error import URLError, HTTPError

from healvis.data import DATA_PATH


@pytest.fixture(autouse=True, scope="session")
def setup_and_teardown_package():
    """Make data/test directory to put test output files in."""
    testdir = os.path.join(DATA_PATH, 'temporary_test_data/')
    if not os.path.exists(testdir):
        os.mkdir(testdir)

    # try to download the iers table. If it fails, turn off auto downloading for the tests
    # and turn it back on in teardown_package (done by extending auto_max_age)
    if iers.conf.auto_download:
        try:
            t1 = Time.now()
            t1.ut1
        except (URLError, HTTPError):
            try:
                iers.IERS.iers_table = iers.IERS_A.open(iers.IERS_A_URL_MIRROR)
                t1 = Time.now()
                t1.ut1
            except(URLError, HTTPError):
                iers.conf.auto_max_age = None
                iers.conf.auto_download = False
    # yield to allow tests to run
    yield

    iers.conf.auto_max_age = 30
    iers.conf.auto_download = True

    # clean up the test directory after
    if os.path.exists(testdir):
        shutil.rmtree(testdir)
