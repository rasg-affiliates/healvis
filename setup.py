# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from __future__ import absolute_import, division, print_function

from setuptools import setup
import glob
import os
import io
import numpy as np
import json

import sys
sys.path.append('healvis')
import version

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(os.path.join('healvis', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)

def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths

data_files = package_files('healvis', 'data')

setup_args = {
    'name': 'healvis',
    'author': 'Radio Astronomy Software Group',
    'url': 'https://github.com/RadioAstronomySoftwareGroup/healvis',
    'license': 'BSD',
    'description': 'a healpix-based radio interferometric visibility simulator',
    'package_dir': {'healvis': 'healvis'},
    'packages': ['healvis', 'healvis.tests'],
    'include_package_data': True,
    'package_data': {'healvis': data_files},
#    'scripts': glob.glob('scripts/*'),
    'version': version.version,
    'include_package_data': True,
    'setup_requires': ['numpy>=1.14', 'six>=1.10'],
    'install_requires': ['numpy>=1.14', 'six>=1.10', 'scipy', 'astropy>=2.0', 'numba', 'h5py', 'pyyaml', 'pyuvdata','astropy-healpix'],
    'keywords': 'radio astronomy interferometry'
}

if __name__ == '__main__':
    setup(**setup_args)
