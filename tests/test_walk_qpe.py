#!/usr/bin/env python3

import qpe_toolbox.estimation as qpe
from qpe_toolbox.hamiltonian import do_dmrg, heisenberg_hamiltonian


def test_walk_qpe():
    n_qbits = 2
    m_ph = 2

    H = heisenberg_hamiltonian(n_qbits)
    E0, psi0_mps = do_dmrg(H)
    lmb = sum([abs(P[0]) for P in H.terms])

    _traces, theta = qpe.run_qpe_lcu_walk_operator(H, psi0_mps, m_ph)
    energy = qpe.get_energy_from_lcu_walk_phase(theta, lmb)
    delta_e = qpe.estimate_lcu_error(m_ph, E0, lmb)

    assert abs(E0 - energy) < delta_e


if __name__ == "__main__":
    test_walk_qpe()
