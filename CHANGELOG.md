# Changelog

## [v1.1.0] 7-22-2019

### Added

- 'anchor_ant' selection -- choose all baselines containing this antenna

### Fixed

- Azimuth angle calculation accounts for change in North pole position in ICRS.
- Baseline uvw convention --- ant2 - ant1 (previously 1 - 2)
- The azimuth convention did not match astropy and was mislabelled.
- Frequency interpolation is properly passed along.

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
