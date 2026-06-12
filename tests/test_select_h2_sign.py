#!/usr/bin/env python3

from pyscf import gto
from quimb.tensor import CircuitMPS

import qpe_toolbox.estimation as qpe
from qpe_toolbox.hamiltonian import chemistry_hamiltonian, do_dmrg
from qpe_toolbox.tensor import kron_mps


def test_select():
    mol = gto.M(
        atom=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))],
        basis="STO-3G",
        verbose=0,
    )

    H = chemistry_hamiltonian(
        mol, hf_mode="rhf", encoding="original", do_fci=False, do_ccsd=False
    )

    _weights, lmb, _L, _m_L = qpe.get_lcu_weights(H)

    # DMRG E0 and psi0
    E0_dmrg, psi0_mps = do_dmrg(H)

    L_mps = qpe.build_lcu_prepare_state_mps(H)

    Lpsi_mps = kron_mps(L_mps, psi0_mps)

    select_gates = qpe.lcu_select_gates(H)

    circ = CircuitMPS(psi0=Lpsi_mps)
    for gate in select_gates:
        circ.apply_gate(gate)

    psi1 = circ.psi.copy()

    assert abs(Lpsi_mps.overlap(psi1) - E0_dmrg / lmb) < 1e-10


if __name__ == "__main__":
    test_select()
