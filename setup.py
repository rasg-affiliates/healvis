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

from healvis import version

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(os.path.join('healvis', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)

setup_args = {
    'name': 'healvis',
    'author': 'Radio Astronomy Software Group',
    'url': 'https://github.com/RadioAstronomySoftwareGroup/healvis',
    'license': 'BSD',
    'description': 'a healpix-based radio interferometric visibility simulator',
    'package_dir': {'healvis': 'healvis'},
    'packages': ['healvis', 'healvis.tests'],
#    'scripts': glob.glob('scripts/*'),
    'version': version.version,
    'include_package_data': True,
    'setup_requires': ['numpy>=1.14', 'six>=1.10'],
    'install_requires': ['numpy>=1.14', 'six>=1.10', 'scipy', 'astropy>=2.0'],
    'keywords': 'radio astronomy interferometry'
}

if __name__ == '__main__':
    setup(**setup_args)
