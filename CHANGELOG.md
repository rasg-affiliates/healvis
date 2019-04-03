# Changelog

## [v1.0.0] 4-3-2019

### Added

- Chromatic gaussian beam with power law width scaling with frequency.
- Frequency selections in obsparam to apply to loading skymodels
- Documented convention that frequency array values refer to channel centers
- Test for consistency among setup parameters.
- Full support for all valid pyuvsim setup parameters.

### Changed

- Set integration_time on output UVData object to 0's, to reflect that it's not a meaningful parameter in simulation`

### Fixed

- Bug fix in Observatory.beam_sq_int (missing square on Nside)
- Restored support for redundancy selection.
- Loading from hdf5 files into shared memory was not actually using the mparray.
- Single-frequency simulations can be run.

### Deprecated
