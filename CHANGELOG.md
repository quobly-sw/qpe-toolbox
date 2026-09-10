# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Effort-based versioning](https://jacobtomlinson.dev/effver/).

## [Unreleased]

### Changed

- `robust_phase_estimation`: replaced `epsilon` with `n_repetitions`, removed
  `sign_E0`, added an `rng` argument for deterministic sampling, added a `t0`
  argument setting the base evolution time, and changed the `trotter_order`
  default from 2 to 1. The estimated phase is now `E0 * t0`, so the
  energy is recovered as `theta / t0`. The returned list now has length
  `n_repetitions` (no leading placeholder).
- `run_hadamard_test` / `rpe_get_hadamard_output`: replaced `seed` with an
  `rng` (`numpy.random.Generator`) argument.
- `build_hadamard_test_circuit` / `run_hadamard_test`: renamed the `theta`
  argument to `phase_gate_angle`.
- `rpe_get_hadamard_output`: renamed the `m` argument to `evolution_time`. It
  now takes the evolution time itself instead of the exponent `m`; callers
  passing `m` positionally must pass `2**m` to keep the previous behaviour.
- Renamed `rpe_distance` to `angular_distance`; it is now vectorized.
- `rpe_update_theta`: signature changed to `(phi_m, theta_ref, m)`, now
  returning a single angle.
- `optuna` dependency moved from core dependency to recommended.
- `draw_layered_circuit` / `draw_layered_expval`: replaced the `list_names`
  argument with three keyword-only arguments `state_label`, `labels_1qubit` and
  `labels_2qubit`. A label list shorter than the circuit depth now raises
  `ValueError` instead of failing with an `IndexError` while drawing.

### Removed

- `rpe_distance` (renamed to `angular_distance`).
- `hyperoptimization` example and all QAOA-related code: `examples/hyperoptimization.py`,
  `src/qpe_toolbox/circuit/qaoa.py` and `tests/test_qaoa.py`. This removes
  `brute_force_maxcut`, `compute_qaoa_contraction_costs` and
  `study_optimization_time_costs` from `qpe_toolbox.circuit`.
- `two_qubit_rand_layer`: the `reverse` argument. It only changed the order in
  which overlapping gates were applied, and never the control direction its
  documentation claimed.

### Fixed

- `draw_layered_expval`: two-qubit layer labels were indexed in the opposite
  direction to single-qubit ones, so the two label lists disagreed on which
  layer they were naming.
- `draw_layered_circuit` / `draw_layered_expval`: unlabelled diagrams drew
  `['']` next to every qubit instead of nothing.

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
