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

    theta_list = robust_phase_estimation(
        H, psi0, n_repetitions, sign_E0=-1, n_steps=EXACT, n_shots=EXACT, verbosity=0
    )
    assert abs(rpe_distance(E0, theta_list[-1])) < 2**-n_repetitions


def test_rpe_seed_deterministic():
    H = heisenberg_hamiltonian(4)
    _E0, psi0 = do_dmrg(H)

    kwargs = {"sign_E0": -1, "n_steps": EXACT, "n_shots": 4}
    a = robust_phase_estimation(H, psi0, 5, rng=np.random.default_rng(123), **kwargs)
    b = robust_phase_estimation(H, psi0, 5, rng=np.random.default_rng(123), **kwargs)
    c = robust_phase_estimation(H, psi0, 5, rng=np.random.default_rng(456), **kwargs)

    assert a == b  # same seed -> identical
    assert a != c  # different seed -> different sampling


if __name__ == "__main__":
    test_rpe()
    test_rpe_seed_deterministic()
