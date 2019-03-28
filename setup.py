# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from __future__ import absolute_import, division, print_function

from setuptools import setup
import os
import json

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

version_file = os.path.join('healvis', 'VERSION')
with open(version_file) as f:
    version = f.read().strip()

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
    'scripts': glob.glob('scripts/skymodel_vis_sim.py'),
    'version': version,
    'include_package_data': True,
    'setup_requires': ['numpy>=1.14', 'six>=1.10'],
    'install_requires': ['numpy>=1.14', 'six>=1.10', 'scipy', 'astropy', 'pyuvdata>=1.2.1', 'numba>=0.43.1'],
    'keywords': 'radio astronomy interferometry'
}

if __name__ == '__main__':
    setup(**setup_args)

    from healvis import version  # noqa
    data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
    with open(os.path.join('healvis', 'GIT_INFO'), 'w') as outfile:
        json.dump(data, outfile)
