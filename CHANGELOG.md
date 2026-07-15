# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Effort-based versioning](https://jacobtomlinson.dev/effver/).

## [Unreleased]

### Changed

- `robust_phase_estimation`: replaced `epsilon` with `n_repetitions`, removed
  `sign_E0`, added an `rng` argument for deterministic sampling, and changed
  the `trotter_order` default from 2 to 1. The returned list now has length
  `n_repetitions` (no leading placeholder).
- `run_hadamard_test` / `rpe_get_hadamard_output`: replaced `seed` with an
  `rng` (`numpy.random.Generator`) argument.
- Renamed `rpe_distance` to `angular_distance`; it is now vectorized.
- `rpe_update_theta`: signature changed to `(phi_m, theta_ref, m)`, now
  returning a single angle.

### Removed

- `rpe_distance` (renamed to `angular_distance`).

## [1.1.0] - 2026-04-02

### Changed

- misc documentation improvement.
- `qiskit-aer` switched from test group to core dependency.
- `kahypar` switched from core to optional dependency.
- `cotengrust` added as optional dependency.

## [1.0.0] - 2026-03-18

### Added

- Initial public release of `qpe-toolbox`.
- `circuit` module: `quimb` circuit construction, parametrized circuits, QAOA, gate counting, serialization, and plotting.
- `hamiltonian` module: `Hamiltonian` class, spin models, `pyscf` interface for molecular chemistry, fermionic encodings via `openfermion`.
- `estimation` module: textbook and robust QPE variants, LCU walk operators, block encoding, QFT, Hadamard test.
- `tensor` module: MPS/MPO utilities beyond what `quimb` provides natively.
- Sphinx documentation deployed to GitHub Pages.
- Full test suite with ≥ 70 % branch coverage.
- CI/CD workflows for testing and PyPI publishing.

[Unreleased]: https://github.com/quobly-sw/qpe-toolbox/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/quobly-sw/qpe-toolbox/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/quobly-sw/qpe-toolbox/releases/tag/v1.0.0
