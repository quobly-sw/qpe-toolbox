#!/usr/bin/env python3

import numpy as np
import quimb.tensor as qtn

from qpe_toolbox.circuit import init_cost_tn, transpile_mpo_to_circuit
from qpe_toolbox.circuit.mpo_circuit_transpilation import (
    build_first_sweep,
    find_transfer_structure,
    optimize_single_gate_update,
)


def test_transpile_mpo_to_circuit_converges_to_identity():
    # fitting a brickwall ansatz (initialized close to identity) against the
    # identity MPO should recover perfect overlap
    n_qubits = 4
    ref_mpo = qtn.MPO_identity(n_qubits)
    _, _, overlap = transpile_mpo_to_circuit(
        ref_mpo,
        1,
        1e-8,
        20,
        param_scaling=1e-1,
        closed=True,
        rng=np.random.default_rng(42),
    )
    assert np.isclose(overlap, 1.0, atol=1e-6)


def test_transpile_mpo_to_circuit_multi_sweep():
    # force several full LR+RL sweeps, exercising the rtol early-stopping
    # path and reusing the "RL" reversed-range site sequence multiple times
    n_qubits = 4
    ref_mpo = qtn.MPO_identity(n_qubits)
    _, _, overlap = transpile_mpo_to_circuit(
        ref_mpo,
        2,
        1e-10,
        100,
        param_scaling=1e-1,
        closed=True,
        rng=np.random.default_rng(1),
    )
    assert np.isclose(overlap, 1.0, atol=1e-6)


def test_optimized_gates_keep_index_order():
    # gate data must stay readable positionally as (out, out, in, in): the SVD
    # update must not permute the legs of the gate tensors
    n_qubits = 5
    cost_tn = init_cost_tn(
        qtn.MPO_identity(n_qubits), 2, closed=True, rng=np.random.default_rng(42)
    )
    gate_inds = {
        tag: cost_tn[tag].inds for tag in cost_tn.tags if tag.startswith("GATE_")
    }
    transfer_structure = find_transfer_structure(n_qubits, cost_tn)
    contracted_envs = build_first_sweep(n_qubits, cost_tn, transfer_structure)
    cost_tn, _, _ = optimize_single_gate_update(
        n_qubits, cost_tn, 1e-8, 5, transfer_structure, contracted_envs
    )
    for tag, inds in gate_inds.items():
        assert cost_tn[tag].inds == inds


if __name__ == "__main__":
    test_transpile_mpo_to_circuit_converges_to_identity()
    test_transpile_mpo_to_circuit_multi_sweep()
    test_optimized_gates_keep_index_order()
