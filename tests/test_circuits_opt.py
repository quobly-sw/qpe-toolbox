#!/usr/bin/env python3

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

from qpe_toolbox.circuit import ansatz_circuit_su4, tn_fit
from qpe_toolbox.circuit.circuits_opt import svd_optimal_gate_update


def test_tn_fit():
    rng = np.random.default_rng(42)
    circ = ansatz_circuit_su4(2, 1, param_scaling=1.0, parametrize=False, rng=rng)
    tn1 = circ.psi
    tn2 = qtn.MPS_rand_state(2, 4, seed=42)
    tn_fit(tn1, tn2, tags="SU4SWAP", steps=1000, tol=1e-10)

    ovlp = abs((tn2.H & tn1).contract())
    assert abs(ovlp - 1.0) < 1e-8


def test_tn_fit_rejects_non_gate_tensors():
    rng = np.random.default_rng(42)
    target = qtn.MPS_rand_state(4, 2, seed=42)

    # default gate_contract splits a CNOT (rank 2) into two 3-leg tensors
    circ = qtn.Circuit(4)
    circ.apply_gate_raw(qu.controlled("not"), (0, 1), tags="FIT")
    with pytest.raises(ValueError, match="gate_contract=False"):
        tn_fit(circ.psi, target, tags="FIT")

    # parametrized gates cannot be updated in place
    circ = ansatz_circuit_su4(4, 1, rng=rng)
    with pytest.raises(TypeError, match="parametrize=False"):
        tn_fit(circ.psi, target)

    # tags=None also selects the initial state tensors
    circ = ansatz_circuit_su4(4, 1, parametrize=False, rng=rng)
    with pytest.raises(ValueError, match="whole two-qubit gates"):
        tn_fit(circ.psi, target, tags=None)


def _is_unitary(tensor, inds):
    matrix = tensor.transpose(*inds).data.reshape(4, 4)
    return np.allclose(matrix.conj().T @ matrix, np.eye(4))


def test_svd_optimal_gate_update_is_unitary():
    # an environment acting on |00> has rank 1: the update must stay unitary
    rng = np.random.default_rng(42)
    phi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    inds = ("o0", "o1", "i0", "i1")
    env = qtn.Tensor(np.outer(phi, [1, 0, 0, 0]).reshape(2, 2, 2, 2), inds=inds)
    new_gate, _ = svd_optimal_gate_update(env, ("o0", "o1"))
    assert _is_unitary(new_gate, inds)

    # first-layer gates of a fitted circuit act on |0...0>
    circ = ansatz_circuit_su4(4, 2, parametrize=False, rng=rng)
    psi = circ.psi
    tn_fit(psi, qtn.MPS_rand_state(4, 2, seed=42), tags="SU4SWAP", steps=5)
    for gate in psi.select_tensors("SU4SWAP"):
        assert _is_unitary(gate, gate.inds)


if __name__ == "__main__":
    test_tn_fit()
    test_tn_fit_rejects_non_gate_tensors()
    test_svd_optimal_gate_update_is_unitary()
