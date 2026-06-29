#!/usr/bin/env python3

import numpy as np

from qpe_toolbox import EXACT
from qpe_toolbox.estimation import robust_phase_estimation, rpe_distance
from qpe_toolbox.hamiltonian import do_dmrg, heisenberg_hamiltonian


def test_rpe():
    n_qubits = 4
    H = heisenberg_hamiltonian(n_qubits)
    E0, psi0 = do_dmrg(H)
    n_repetitions = 7

    theta_list = robust_phase_estimation(H, psi0, n_repetitions, EXACT, EXACT)
    assert abs(rpe_distance(E0, theta_list[-1])) < 2**-n_repetitions


def test_rpe_seed_deterministic():
    H = heisenberg_hamiltonian(4)
    _E0, psi0 = do_dmrg(H)

    a = robust_phase_estimation(H, psi0, 5, EXACT, 4, rng=np.random.default_rng(123))
    b = robust_phase_estimation(H, psi0, 5, EXACT, 4, rng=np.random.default_rng(123))
    c = robust_phase_estimation(H, psi0, 5, EXACT, 4, rng=np.random.default_rng(456))

    assert a == b  # same seed -> identical
    assert a != c  # different seed -> different sampling


if __name__ == "__main__":
    test_rpe()
    test_rpe_seed_deterministic()
