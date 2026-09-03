#!/usr/bin/env python3

import numpy as np
import quimb.tensor as qtn

from qpe_toolbox.circuit import transpile_mpo_to_circuit


def test_transpile_mpo_to_circuit_converges_to_identity():
    # fitting a brickwall ansatz (initialized close to identity) against the
    # identity MPO should recover perfect overlap
    n_qubits = 4
    ref_mpo = qtn.MPO_identity(n_qubits)
    cost_tn, _ = transpile_mpo_to_circuit(
        ref_mpo,
        1,
        1e-8,
        20,
        param_scaling=1e-1,
        closed=True,
        rng=np.random.default_rng(42),
    )
    overlap = abs(cost_tn.contract(all, optimize="auto-hq")) / 2**n_qubits
    assert np.isclose(overlap, 1.0, atol=1e-6)


def test_transpile_mpo_to_circuit_multi_sweep():
    # force several full LR+RL sweeps, exercising the rtol early-stopping
    # path and reusing the "RL" reversed-range site sequence multiple times
    n_qubits = 4
    ref_mpo = qtn.MPO_identity(n_qubits)
    cost_tn, _ = transpile_mpo_to_circuit(
        ref_mpo,
        2,
        1e-10,
        10,
        param_scaling=1e-1,
        closed=True,
        rng=np.random.default_rng(1),
    )
    overlap = abs(cost_tn.contract(all, optimize="auto-hq")) / 2**n_qubits
    assert np.isclose(overlap, 1.0, atol=1e-6)


if __name__ == "__main__":
    test_transpile_mpo_to_circuit_converges_to_identity()
    test_transpile_mpo_to_circuit_multi_sweep()
