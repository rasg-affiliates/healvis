# healvis

![](https://github.com/aelanman/pyspherical/workflows/Tests/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/rasg-affiliates/healvis/branch/master/graph/badge.svg)](https://codecov.io/gh/rasg-affiliates/healvis)


Radio interferometric visibility simulator based on HEALpix maps.

**Note** This is a tool developed for specific research uses, and is not yet at the development standards of other RASG projects. Use at your own risk.

## Dependencies
Python dependencies for `healvis` include

* numpy
* astropy
* astropy-healpix
* scipy
* h5py
* pyyaml
* multiprocessing
* [pyuvdata](https://github.com/HERA-Team/pyuvdata/)

These will be installed automatically. Optional dependencies include

* [pygsm](https://github.com/telegraphic/PyGSM)
* [scikit-learn](https://scikit-learn.org/stable/)

The use of PyGSM within this package is subject to the GNU General Public License (GPL), due to its dependency on `healpy`.

## Installation
Clone this repository and run the installation script as
```pip install .```

To install optional dependencies, use ```pip install .[gsm]``` to install with PyGSM or ```pip install .[all]``` to install scikit-learn as well.

To install `healvis` for development, use ```pip install .[dev]```.

## Getting Started
To get started running `healvis`, see our [tutorial notebooks](https://github.com/RadioAstronomySoftwareGroup/healvis/tree/master/notebooks).

## Updates in this fork

- Allow a telescope height above ground to be specfied.
- Allow specifying a different beam for each antenna.
- Make sure beam is multiplied twice (not once) into cross-correlation calculations. If the beam is not a power beam. However, beam handling is inconsistent; see issue #1.
- Add the ability to seed the random number generator, which is used for generating the EoR shell.
- Ignore pixels below the horizon.
