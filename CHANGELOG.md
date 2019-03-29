# Changelog

## [1.0.0] - 2019-3-29

### Added

- Documented convention that frequency array values refer to channel centers
- Test for consistency among setup parameters.
- Full support for all valid pyuvsim setup parameters.

### Changed

- Set integration_time on output UVData object to 0's, to reflect that it's not a meaningful parameter in simulation`

### Fixed

- Restored support for redundancy selection.
- Loading from hdf5 files into shared memory was not actually using the mparray.
- Single-frequency simulations can be run.

### Deprecated
