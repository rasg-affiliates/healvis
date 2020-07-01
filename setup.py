# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from setuptools import setup
import os
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
    'version': version.version,
    'include_package_data': True,
    'setup_requires': ['numpy'],
    'install_requires': ['numpy', 'scipy', 'astropy', 'astropy_healpix', 'pyyaml'],
    'keywords': 'radio astronomy interferometry',
    'extras_require': {
        'gsm': "pygsm @ git+git://github.com/telegraphic/PyGSM.git",
        'all': ["pygsm @ git+git://github.com/telegraphic/PyGSM.git", 'scikit-learn']
    }
}

if __name__ == '__main__':
    setup(**setup_args)
