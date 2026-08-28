#!/usr/bin/env python3

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

from qpe_toolbox.circuit import (
    state_preparation_mpo,
    transpile_mpo_to_circuit,
    trotter_approx_as_MPO,
)
from qpe_toolbox.hamiltonian import heisenberg_hamiltonian

ham = heisenberg_hamiltonian(3)
ham_dense = ham.to_dense()


def _trotter_error(trotter_order, dt):
    exact = qu.expm(-1j * dt * ham_dense)
    approx = trotter_approx_as_MPO(
        ham, dt, trotter_order=trotter_order, cutoff=1e-12, max_bond=64
    ).to_dense()
    return np.linalg.norm(approx - exact)


def test_trotter_invalid_order_raises():
    with pytest.raises(ValueError):
        trotter_approx_as_MPO(ham, 0.1, trotter_order=3, cutoff=1e-12, max_bond=64)


def test_trotter_order_accuracy_ranking():
    # a higher-order product formula must approximate the exact evolution
    # more closely than a lower-order one, at fixed dt
    dt = 0.1
    err1 = _trotter_error(1, dt)
    err2 = _trotter_error(2, dt)
    err4 = _trotter_error(4, dt)
    assert err2 < err1
    assert err4 < err2


def test_trotter_order4_scaling():
    # regression test for the triple-jump composition bug: the global error
    # of the 4th-order product formula scales as dt**5
    dts = [0.2, 0.1, 0.05]
    errs = [_trotter_error(4, dt) for dt in dts]
    slope = np.log(errs[0] / errs[-1]) / np.log(dts[0] / dts[-1])
    assert 4.5 < slope < 5.5


def test_state_preparation_mpo():
    # state_preparation_mpo expects the boundary-tensor leg order DMRG2
    # produces (phys, bond) / (bond, phys) -- not MPS_rand_state's (bond, phys)
    dmrg = qtn.DMRG2(
        ham.to_mpo(), p0=qtn.MPS_rand_state(ham.n_qubits, bond_dim=2, seed=42)
    )
    dmrg.solve(max_sweeps=8, bond_dims=16, verbosity=0, cutoffs=1e-10)
    gs = dmrg.state
    gs_vec = gs.to_dense().reshape(-1)

    dense = state_preparation_mpo(state_mps=gs).to_dense()

    n = ham.n_qubits
    e0 = np.zeros(2**n, dtype=complex)
    e0[0] = 1.0
    assert np.allclose(dense @ e0, 2**n * gs_vec, atol=1e-8)

    e1 = np.zeros(2**n, dtype=complex)
    e1[1] = 1.0
    assert np.allclose(dense @ e1, 0, atol=1e-8)


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
    test_trotter_invalid_order_raises()
    test_trotter_order_accuracy_ranking()
    test_trotter_order4_scaling()
    test_state_preparation_mpo()
    test_transpile_mpo_to_circuit_converges_to_identity()
    test_transpile_mpo_to_circuit_multi_sweep()
