#!/usr/bin/env python3

import numpy as np
import quimb.tensor as qtn

import qpe_toolbox.estimation as qpe
from qpe_toolbox.hamiltonian import do_dmrg, heisenberg_hamiltonian
from qpe_toolbox.tensor import add_cqubit_mpo, apply_gate_from_mpo, kron_mps

tol = 1e-10

n_qubits = 4
hamiltonian = heisenberg_hamiltonian(n_qubits)
weights, lmb, L, m_L = qpe.get_lcu_weights(hamiltonian)

E0, psi0 = do_dmrg(hamiltonian)

###############################################################################
# define the tests


# I. PREPARE


def test_L_mps():
    L_mps = qpe.build_lcu_prepare_state_mps(hamiltonian, cutoff=tol)
    assert L_mps.max_bond() == 2
    assert (
        abs(1 - L_mps.overlap(qpe.build_lcu_prepare_state_mps(hamiltonian, cutoff=0)))
        ** 2
        < tol
    )


def test_prepare_mpo():
    prep_mpo = qpe.build_lcu_prepare_mpo(hamiltonian)
    zero_mps = qtn.MPS_computational_state("0" * m_L)
    L_mps = qpe.build_lcu_prepare_state_mps(hamiltonian)
    test_mps = prep_mpo.apply(zero_mps)
    assert abs(test_mps.norm() - 1) < 1e-12
    assert abs(abs(test_mps.H @ L_mps) ** 2 - 1) < 1e-12

    for k in range(1, 2**m_L):
        k_mps = qtn.MPS_computational_state(f"{{0:0{m_L}b}}".format(k))
        test_mps = prep_mpo.apply(k_mps)
        assert abs(test_mps.norm()) ** 2 < 1e-12


# II. SELECT


def test_slct_reflection():
    select_mpo = qpe.build_lcu_select_mpo(hamiltonian)
    Id_mpo = select_mpo.apply(select_mpo)
    Id_mpo.compress(cutoff=1e-18)
    err_mpo = Id_mpo - qtn.MPO_identity(m_L + hamiltonian.n_qubits)
    assert abs(err_mpo.norm()) ** 2 < 1e-12


def test_slct_gates_comp_basis():
    select_gates = qpe.lcu_select_gates(hamiltonian)
    select_mpo = qpe.build_lcu_select_mpo(hamiltonian)
    for k in range(2 ** (m_L)):
        psi = qtn.MPS_computational_state(f"{{0:0{m_L}b}}".format(k) + "0" * n_qubits)

        circ = qtn.CircuitMPS(psi0=psi)
        for gate in select_gates:
            circ.apply_gate(gate)
        psi1 = circ.psi
        psi2 = select_mpo.apply(psi)

        assert abs(abs(psi2.H @ psi1) ** 2 - 1) < tol


def test_select_gates_Lpsi():
    _, lmb, _, _ = qpe.get_lcu_weights(hamiltonian)
    L_mps = qpe.build_lcu_prepare_state_mps(hamiltonian)

    Lpsi_mps = kron_mps(L_mps, psi0)

    select_gates = qpe.lcu_select_gates(hamiltonian)
    circ = qtn.CircuitMPS(psi0=Lpsi_mps)
    for gate in select_gates:
        circ.apply_gate(gate)

    assert abs(Lpsi_mps.H @ circ.psi - E0 / lmb) < 1e-12


# III. REFLECTION


def test_RL():
    R_L = qpe.build_lcu_reflection_mpo(hamiltonian)

    Id_test = R_L.apply(R_L)
    error_mpo = Id_test - qtn.MPO_identity(m_L + n_qubits)
    assert abs(error_mpo.norm()) ** 2 < tol

    L_mps = qpe.build_lcu_prepare_state_mps(hamiltonian)
    Lpsi = kron_mps(L_mps, psi0)
    assert abs((R_L.apply(Lpsi) - Lpsi).norm()) ** 2 < tol


# IV. Controlled-Walk operator


def test_controlled_walk():
    L_mps = qpe.build_lcu_prepare_state_mps(hamiltonian)
    psi_init = kron_mps(qtn.MPS_computational_state("1"), kron_mps(L_mps, psi0))

    # Create circuit, define registers
    circ = qtn.CircuitMPS(m_L + n_qubits + 1, psi0=psi_init)
    anc_reg = (0,)

    # SELECT
    select_gates = qpe.lcu_select_gates(hamiltonian)
    for g in select_gates:
        qubits = tuple([k + 1 for k in g.qubits])
        controls = (
            anc_reg
            if g.controls is None
            else tuple([k + 1 for k in g.controls]) + anc_reg
        )
        circ.apply_gate(g.copy_with(qubits=qubits, controls=controls))

    select_Lpsi = circ.psi
    phi = (select_Lpsi - E0 / lmb * psi_init) / np.sqrt(1 - (E0 / lmb) ** 2)

    RL_mpo = qpe.build_lcu_reflection_mpo(hamiltonian)
    cRL_mpo = add_cqubit_mpo(RL_mpo, "before")

    circ_final = apply_gate_from_mpo(circ=circ, mpo=cRL_mpo)
    psi_final = circ_final.psi.copy()

    assert abs(psi_init.H @ psi_final - E0 / lmb) < tol
    assert abs(phi.H @ psi_final + np.sqrt(1 - (E0 / lmb) ** 2)) < tol


###############################################################################
#### run the tests
if __name__ == "__main__":
    # run
    test_L_mps()
    test_prepare_mpo()

    test_slct_reflection()
    test_slct_gates_comp_basis()
    test_select_gates_Lpsi()

    test_RL()
