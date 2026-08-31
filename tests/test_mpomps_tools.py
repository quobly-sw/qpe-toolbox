#!/usr/bin/env python3

import numpy as np
import quimb.tensor as qtn

from qpe_toolbox.hamiltonian import heisenberg_hamiltonian
from qpe_toolbox.tensor import kron_mpos, kron_mps, state_preparation_mpo

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
    # state_preparation_mpo expects the boundary-tensor leg order DMRG2
    # produces (phys, bond) / (bond, phys) -- not MPS_rand_state's (bond, phys)
    ham = heisenberg_hamiltonian(3)
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


# run
if __name__ == "__main__":
    test_kronmps()
    test_kronmpos()
    test_state_preparation_mpo()
