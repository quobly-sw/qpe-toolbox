# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Effort-based versioning](https://jacobtomlinson.dev/effver/).

## [Unreleased]

### Added

- `transpile_mpo_to_circuit`: convenience wrapper chaining `init_cost_tn`,
  `find_transfer_structure`, `build_first_sweep` and `optimize_single_gate_update`
  to fit a brickwall circuit ansatz to a reference MPO in one call.
- `init_cost_tn` and `transpile_mpo_to_circuit` are now re-exported from
  `qpe_toolbox.circuit`.
- `state_preparation_mpo` is now re-exported from `qpe_toolbox.tensor`.
- `trotter_approx_as_MPO` is now re-exported from `qpe_toolbox.hamiltonian`.
- `optimize_single_gate_update` / `transpile_mpo_to_circuit`: keyword-only
  `optimize` argument selecting the contraction strategy of each gate's local
  environment.

### Changed

- `init_cost_tn`: replaced `seed` (int) with an `rng` (`numpy.random.Generator`)
  argument, consistent with the rest of the codebase.
- Trotter-Suzuki MPO functions (`trotter_approx_as_MPO`,
  `trotter1_approx_as_MPO`, `trotter2_approx_as_MPO`, `trotter4_approx_as_MPO`,
  `exp_Pauli_string_as_MPO`) moved from `qpe_toolbox.circuit.mpo_circuit_transpilation`
  to `qpe_toolbox.hamiltonian.trotterization`, and `rotation_gates` moved there from
  `qpe_toolbox.hamiltonian.hamiltonian`. `state_preparation_mpo` moved to
  `qpe_toolbox.tensor.mpomps_tools`. Importing the Trotter functions or
  `state_preparation_mpo` from `qpe_toolbox.circuit.mpo_circuit_transpilation` now
  raises `ImportError`; `rotation_gates` can still be imported from its old module.
- `trotter_approx_as_MPO`: renamed `order` to `trotter_order`, consistent with
  `Hamiltonian.get_trotter_step` and every QPE/RPE function in the codebase. `dt`
  is now positional, and `trotter_order`, `cutoff` and `max_bond` default to `1`,
  `1e-10` and `None`.
- `trotter1_approx_as_MPO` / `trotter2_approx_as_MPO` / `trotter4_approx_as_MPO`:
  take a `Hamiltonian` and a positional `dt` instead of `ham_terms, n_qubits`
  and a keyword-only `dt`. `cutoff` and `max_bond` default to `1e-10` and `None`.
- `exp_Pauli_string_as_MPO`: signature changed from `(ham_term, n_qubits, *, theta)`
  to `(term, dt, n_qubits)`, and it now builds `exp(-i * dt * coeff * P)`, matching
  `rotation_gates` and the Trotter functions: pass `dt = -theta` to recover the old
  `exp(i * theta * coeff * P)`. It also raises `ValueError` if a qubit is repeated
  or out of range.
- `state_preparation_mpo`: raises `ValueError` unless the MPS arrays are in `'lpr'`
  order (as returned by `DMRG2`), instead of silently building a wrong MPO.
- `optimize_single_gate_update`: now returns `(cost_tn, contracted_envs, overlap)`,
  where `overlap` is normalized by `2**n_qubits`.
- `mpo_circuit_transpilation` helpers: `update_dict_contr_envs` renamed to
  `update_contracted_envs`; it and `update_cost_tn` take a single gate tensor
  instead of a list. Arguments `dict_transf`, `dict_contr_envs` and `x` renamed to
  `transfer_structure`, `contracted_envs` and `site_index`.
- `tn_fit`: `tags`, `steps`, `tol` and `contract_optimize` are now keyword-only. A
  progress bar is always shown; set `TQDM_DISABLE=1` before importing `tqdm` to
  silence it. Raises `TypeError` on parametrized tensors, and `ValueError` if
  `tags` selects anything other than whole two-qubit gates (e.g. gates split by
  quimb's default `gate_contract`).
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
- `tn_fit`: the `progbar` argument.
- `PRC_loc_cost_tn`, inlined into the new `optimize_one_gate`.
- `two_qubit_rand_layer`: the `reverse` argument. It only changed the order in
  which overlapping gates were applied, and never the control direction its
  documentation claimed.

### Fixed

- `draw_layered_expval`: two-qubit layer labels were indexed in the opposite
  direction to single-qubit ones, so the two label lists disagreed on which
  layer they were naming.
- `draw_layered_circuit` / `draw_layered_expval`: unlabelled diagrams drew
  `['']` next to every qubit instead of nothing.
- `trotter4_approx_as_MPO`: the 4th-order Suzuki "triple-jump" composition applied
  its middle-layer factor twice instead of sandwiching it between two applications
  of the outer-layer factor, so `trotter_order=4` was less accurate than
  `trotter_order=1`. The error of a single step now scales as `dt**5`.
- `exp_Pauli_string_as_MPO`: Pauli letters were assigned in increasing qubit order,
  so terms listing their qubits unsorted put the letters on the wrong qubits.
- `exp_Pauli_string_as_MPO`: the MPO was compressed with a fixed `cutoff=1e-6`,
  which dropped Hamiltonian terms with `abs(dt * coeff)` below about `1e-3`. It is
  now only truncated at machine precision.
- `optimize_single_gate_update`: optimized gate tensors could end up with their
  input legs swapped, so reading their data as a gate matrix gave the wrong gate.

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
