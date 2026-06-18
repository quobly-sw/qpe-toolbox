# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Effort-based versioning](https://jacobtomlinson.dev/effver/).

## [Unreleased]

### Added

- Generic QPE circuit construction from arbitrary per-phase-qubit unitaries: `qpe_circuit` (with an optional `with_iqft` flag to stop after the controlled unitaries, i.e. the QPE first stage) and `qpe_gates` in the `estimation` module. Each power `U^(2^k)` is supplied as an independent gate list (possibly lazy), enabling non-squaring implementations such as Shor's algorithm.
- `exact_evolution_powers` and `trotter_evolution_powers` to build the controlled unitaries from a `Hamiltonian`.
- `qpe_gate_list` to build the QPE gate list without simulation, for resource analysis and serialization; gates are written to disk when its `savefile` argument is set.
- `trotter_evolution_gates` to build the gate sequence of a single Trotterized evolution `U(t)`.
- `add_gate_controls` in the `circuit` module.

### Changed

- **Breaking:** `qpe_sample`, `qpe_first_stage` and `qpe_energy` no longer accept the output-mode flags `run_simulation` / `write_gates`; use `qpe_gate_list` for the gate-tracking and serialization mode.
- **Breaking:** the Trotter discretization is now specified as an integer number of steps `n_trotter_steps` instead of a step size `dt` (computed internally as `dt = evolution_time / n_trotter_steps`). `qpe_sample` and `qpe_first_stage` replace `dt` with `n_trotter_steps`; `qpe_energy`, `robust_phase_estimation` and `rpe_get_hadamard_output` rename their `n_steps` argument to `n_trotter_steps`.
- **Breaking:** the Hadamard test (`build_hadamard_test_circuit`, `run_hadamard_test`) is now built on `qpe_circuit` and takes the unitary in the framework convention (argument `U_gate` renamed to `unitary`): either a single gate or an iterable of uncontrolled gates on data-register-local qubit indices.
- **Breaking:** `Hamiltonian.get_U_exact` and `Hamiltonian.get_trotter_step` now take the physical register as a keyword-only `phys_reg` argument (renamed from `data_reg`), defaulting to `range(n_qubits)`. `get_trotter_step` takes `trotter_order` before the keyword-only arguments (`get_trotter_step(dt, trotter_order, *, phys_reg=None)`), and `get_U_exact`'s `controls` is now optional (`get_U_exact(evolution_time, *, phys_reg=None, controls=None)`).

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
