# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Effort-based versioning](https://jacobtomlinson.dev/effver/).

## [Unreleased]

### Added

- Generic QPE circuit construction from arbitrary per-phase-qubit unitaries: `qpe_circuit`, `qpe_first_stage_circuit`, `qpe_gates` and `qpe_first_stage_gates` in the `estimation` module. Each power `U^(2^k)` is supplied as an independent gate list (possibly lazy), enabling non-squaring implementations such as Shor's algorithm.
- `exact_evolution_powers` and `trotter_evolution_powers` to build the controlled unitaries from a `Hamiltonian`.
- `qpe_gate_list` to build the QPE gate list without simulation, for resource analysis and serialization.
- `add_gate_controls` in the `circuit` module.

### Changed

- **Breaking:** `qpe_first_stage` and `qpe_sample` no longer accept `run_simulation`; use `qpe_gate_list` for the gate-tracking mode.

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
