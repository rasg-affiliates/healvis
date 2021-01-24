# healvis

[![Build Status](https://travis-ci.org/RadioAstronomySoftwareGroup/healvis.svg?branch=master)](https://travis-ci.org/RadioAstronomySoftwareGroup/healvis)

Radio interferometric visibility simulator based on HEALpix maps.

**Note** This is a tool developed for specific research uses, and is not yet at the development standards of other RASG projects. Use at your own risk.

## Dependencies
Python dependencies for `healvis` include

* numpy >= 1.14
* astropy >= 2.0
* scipy
* healpy >= 1.12.9
* h5py
* pyyaml
* numba
* multiprocessing
* [pyuvdata](https://github.com/HERA-Team/pyuvdata/)

Optional dependencies include

* [pygsm](https://github.com/telegraphic/PyGSM)
* [scikit-learn](https://scikit-learn.org/stable/)

## Installation
Clone this repo and run the installation script as
```python setup.py install```

## Getting Started
To get started running `healvis`, see our [tutorial notebooks](https://github.com/RadioAstronomySoftwareGroup/healvis/tree/master/notebooks).

## Updates in this fork

- Allow a telescope height above ground to be specfied.
- Allow specifying a different beam for each antenna.
- Make sure beam is multiplied twice (not once) into cross-correlation calculations. If the beam is not a power beam. However, beam handling is inconsistent; see issue #1.
- Add the ability to seed the random number generator, which is used for generating the EoR shell.
- Ignore pixels below the horizon.
