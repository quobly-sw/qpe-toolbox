#!/usr/bin/env python3

from qpe_toolbox.estimation import lcu_walk_operator as lcu
from qpe_toolbox.hamiltonian import do_dmrg, heisenberg_hamiltonian


def test_walk_qpe():
    n_qbits = 2
    m_ph = 2

    H = heisenberg_hamiltonian(n_qbits)
    E0, psi0_mps = do_dmrg(H)
    lmb = sum([abs(P[0]) for P in H.terms])

    _traces, theta = lcu.qpe_walk(H, psi0_mps, m_ph, verbosity=0)
    energy = lcu.energy_from_theta(theta, lmb)
    delta_e = lcu.energy_error_bound(m_ph, E0, lmb)

    assert abs(E0 - energy) < delta_e


if __name__ == "__main__":
    test_walk_qpe()
