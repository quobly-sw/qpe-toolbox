#!/usr/bin/env python3

import numpy as np
import quimb as qu
import quimb.tensor as qtn
from pyscf import gto

from qpe_toolbox.estimation import build_hadamard_test_circuit
from qpe_toolbox.hamiltonian import chemistry_hamiltonian, heisenberg_hamiltonian


def test_heisenberg():
    for n_qubits in [2, 4]:
        heis_ham = heisenberg_hamiltonian(n_qubits)
        heis_mpo = heis_ham.to_mpo()
        heis_dense = heis_ham.to_dense()
        assert np.max(abs(heis_dense - heis_mpo.to_dense())) < 1e-12


def test_molecule_h2():
    mol = gto.M(
        atom=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))],
        basis="sto-3g",
        verbose=0,
    )
    e_hf = -1.1169989968

    _ = chemistry_hamiltonian(mol, hf_mode="uhf", do_fci=True, do_ccsd=True)
    h2_ham = chemistry_hamiltonian(mol, hf_mode="rhf", do_fci=True, do_ccsd=True)
    assert h2_ham.n_qubits == 4
    assert abs(h2_ham.e_ccsd - h2_ham.e_fci) < abs(e_hf - h2_ham.e_fci)

    # DMRG
    h2_mpo = h2_ham.to_mpo()
    dmrg = qtn.DMRG2(h2_mpo, bond_dims=[64], cutoffs=1e-10)
    dmrg.solve(tol=1e-10, verbosity=0)
    e_dmrg = dmrg.energy + h2_ham.e_const
    psi_dmrg = dmrg.state
    assert abs(e_dmrg - h2_ham.e_fci) < 1e-10

    # Exact diag
    eigvals, eigvecs = np.linalg.eigh(h2_ham.to_dense())
    e_ed = eigvals[0] + h2_ham.e_const
    psi_ed = eigvecs[:, 0]
    overlap = abs(np.dot(psi_ed.conj(), psi_dmrg.to_dense()[:, 0])) ** 2
    assert abs(h2_ham.e_fci - e_ed) < 1e-10
    assert abs(1 - overlap) < 1e-10

    # HF modes and encodings
    for hf_mode in ["rhf", "uhf"]:
        for encoding in ["original", "sf", "df", "original"]:
            h2_ham = chemistry_hamiltonian(
                mol,
                hf_mode=hf_mode,
                encoding=encoding,
                do_fci=False,
                do_ccsd=False,
            )


def test_U():
    n_qubits = 4

    H = heisenberg_hamiltonian(n_qubits)
    H_dense = H.to_dense()
    t = 1
    U_dense = qu.expm(-1j * H_dense * t)
    eigvals, eigvecs = np.linalg.eigh(H_dense)
    psi0 = eigvecs[:, 0]
    assert abs(np.angle(psi0.H @ U_dense @ psi0) + eigvals[0]) < 1e-11

    H_mpo = H.to_mpo()
    dmrg = qtn.DMRG2(H_mpo, bond_dims=[10, 20, 40, 100, 100, 200], cutoffs=1e-10)
    dmrg.solve(tol=1e-6)
    E0_dmrg = dmrg.energy
    psi0_mps = dmrg.state

    data_reg = list(range(1, n_qubits + 1))
    U_gate = H.get_U_exact(t, data_reg, controls=[0])
    Z = []
    for theta in [0, -np.pi / 2]:
        circ = build_hadamard_test_circuit(psi0_mps, U_gate, theta)
        probs = circ.compute_marginal(where=[0])
        Z.append(probs[0] - probs[1])
    phi_ref = np.angle(Z[0] + 1j * Z[1])
    assert abs(phi_ref + E0_dmrg) < 1e-6

    r = 1
    dt = t / r
    U_gate = [H.get_trotter_step(dt, data_reg, trotter_order=2)] * r
    Z = []
    for theta in [0, -np.pi / 2]:
        circ = build_hadamard_test_circuit(psi0_mps, U_gate, theta)
        probs = circ.compute_marginal(where=[0])
        Z.append(probs[0] - probs[1])
    phi_ref = np.angle(Z[0] + 1j * Z[1])
    assert np.isclose(phi_ref, 1.6068383462530338, rtol=0, atol=1e-15)


if __name__ == "__main__":
    test_heisenberg()
    test_molecule_h2()
    test_U()
