#!/usr/bin/env python3

import numpy as np
import pytest
import quimb.tensor as qtn

from qpe_toolbox.hamiltonian import heisenberg_hamiltonian
from qpe_toolbox.tensor import (
    add_cqubit_mpo,
    controlled_mpo,
    kron_mpos,
    kron_mps,
    state_preparation_mpo,
)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
P0 = np.array([[1, 0], [0, 0]], dtype=complex)
P1 = np.array([[0, 0], [0, 1]], dtype=complex)

# define


def test_kronmps():
    mps1 = qtn.MPS_computational_state("1011")
    mps2 = qtn.MPS_computational_state("001")

    mps_ref = qtn.MPS_computational_state("1011001")
    res = kron_mps(mps1, mps2)
    assert abs(1 - res.overlap(mps_ref)) < 1e-10

    mps1 = qtn.MPS_computational_state("1")
    mps2 = qtn.MPS_computational_state("001")

    mps_ref = qtn.MPS_computational_state("1001")
    res = kron_mps(mps1, mps2)
    assert abs(1 - res.overlap(mps_ref)) < 1e-10

    mps_ref = qtn.MPS_computational_state("0011")
    res = kron_mps(mps2, mps1)
    assert abs(1 - res.overlap(mps_ref)) < 1e-10


def test_kronmpos():
    Id1 = qtn.MatrixProductOperator([np.eye(2)])
    Id2 = qtn.MPO_identity(2)
    Id3 = qtn.MPO_identity(3)

    myId3 = kron_mpos(Id2, Id1)
    assert abs((myId3 - Id3).norm()) < 1e-10

    myId3_bis = kron_mpos(Id1, Id2)
    assert abs((myId3_bis - Id3).norm()) < 1e-10


def test_state_preparation_mpo():
    # DMRG2 returns arrays in the 'lpr' order state_preparation_mpo requires
    ham = heisenberg_hamiltonian(3)
    dmrg = qtn.DMRG2(
        ham.to_mpo(), p0=qtn.MPS_rand_state(ham.n_qubits, bond_dim=2, seed=42)
    )
    dmrg.solve(max_sweeps=8, bond_dims=16, verbosity=0, cutoffs=1e-10)
    gs = dmrg.state
    gs_vec = gs.to_dense().reshape(-1)

    dense = state_preparation_mpo(gs).to_dense()

    n = ham.n_qubits
    e0 = np.zeros(2**n, dtype=complex)
    e0[0] = 1.0
    assert np.allclose(dense @ e0, 2**n * gs_vec, atol=1e-8)

    e1 = np.zeros(2**n, dtype=complex)
    e1[1] = 1.0
    assert np.allclose(dense @ e1, 0, atol=1e-8)

    # any other layout is rejected instead of silently read positionally
    mps = qtn.MPS_rand_state(n, bond_dim=2, seed=42)
    with pytest.raises(ValueError, match="'lpr' order"):
        state_preparation_mpo(mps)
    mps.permute_arrays("lpr")
    dense = state_preparation_mpo(mps).to_dense()
    assert np.allclose(dense @ e0, 2**n * mps.to_dense().reshape(-1), atol=1e-8)


def test_kron_verbosity(capsys):
    # verbosity > 0 prints the resulting tensor shapes
    res = kron_mps(
        qtn.MPS_computational_state("0"),
        qtn.MPS_computational_state("1"),
        verbosity=1,
    )
    assert res.L == 2
    assert capsys.readouterr().out.strip() != ""


def test_kron_weird_shapes():
    # single-site qudit (dim 3) violates the qubit boundary conventions
    qudit_mpo = qtn.MatrixProductOperator([np.eye(3)])
    with pytest.raises(ValueError, match="Weird shape mpo1"):
        kron_mpos(qudit_mpo, qtn.MPO_identity(2))
    with pytest.raises(ValueError, match="Weird shape mpo2"):
        kron_mpos(qtn.MPO_identity(2), qudit_mpo)

    qudit_mps = qtn.MatrixProductState([np.array([1.0, 0, 0])])
    with pytest.raises(ValueError, match="Weird shape mps1"):
        kron_mps(qudit_mps, qtn.MPS_computational_state("0"))
    with pytest.raises(ValueError, match="Weird shape mps2"):
        kron_mps(qtn.MPS_computational_state("0"), qudit_mps)


def test_add_cqubit_mpo():
    # U = X ⊗ Z as a two-site MPO
    def two_site_u():
        return qtn.MatrixProductOperator([X.reshape(1, 2, 2), Z.reshape(1, 2, 2)])

    u_dense = np.kron(X, Z)

    # "before" adds the control as the first qubit: |0><0|⊗I + |1><1|⊗U
    before = add_cqubit_mpo(two_site_u(), "before").to_dense()
    expected_first = np.kron(P0, np.eye(4)) + np.kron(P1, u_dense)
    assert np.allclose(before, expected_first)

    # "after" adds the control as the last qubit: I⊗|0><0| + U⊗|1><1|
    after = add_cqubit_mpo(two_site_u(), "after").to_dense()
    expected_last = np.kron(np.eye(4), P0) + np.kron(u_dense, P1)
    assert np.allclose(after, expected_last)

    with pytest.raises(ValueError, match="Invalid location"):
        add_cqubit_mpo(two_site_u(), "sideways")


def test_controlled_mpo_guards():
    # physical register must sit entirely above the auxiliary register
    with pytest.raises(ValueError, match="min\\(phys_reg\\) > max\\(aux_reg\\)"):
        controlled_mpo(qtn.MPO_identity(2), [0], [1], 0)

    # the auxiliary tensor must be an interior (1, 1, 2, 2) tensor
    with pytest.raises(ValueError, match="Invalid MPO tensor shape"):
        controlled_mpo(qtn.MPO_identity(2), [1], [0], 0)

    # the auxiliary tensor must be the identity
    mpo_bad = qtn.MatrixProductOperator(
        [np.eye(2).reshape(1, 2, 2), X.reshape(1, 1, 2, 2), np.eye(2).reshape(1, 2, 2)]
    )
    with pytest.raises(ValueError, match="Invalid last MPO tensor"):
        controlled_mpo(mpo_bad, [2], [1], 0)


# run
if __name__ == "__main__":
    test_kronmps()
    test_kronmpos()
    test_state_preparation_mpo()
    test_kron_weird_shapes()
    test_add_cqubit_mpo()
    test_controlled_mpo_guards()
