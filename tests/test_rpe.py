#!/usr/bin/env python3

import numpy as np

from qpe_toolbox import EXACT
from qpe_toolbox.estimation import (
    angular_distance,
    robust_phase_estimation,
    rpe_update_theta,
)
from qpe_toolbox.hamiltonian import do_dmrg, heisenberg_hamiltonian


def test_rpe():
    n_qubits = 4
    H = heisenberg_hamiltonian(n_qubits)
    E0, psi0 = do_dmrg(H)
    n_repetitions = 7

    theta_list = robust_phase_estimation(H, psi0, n_repetitions, EXACT, EXACT)
    assert abs(angular_distance(E0, theta_list[-1])) < 2**-n_repetitions


def test_rpe_seed_deterministic():
    H = heisenberg_hamiltonian(4)
    _E0, psi0 = do_dmrg(H)

    a = robust_phase_estimation(H, psi0, 5, EXACT, 4, rng=np.random.default_rng(123))
    b = robust_phase_estimation(H, psi0, 5, EXACT, 4, rng=np.random.default_rng(123))
    c = robust_phase_estimation(H, psi0, 5, EXACT, 4, rng=np.random.default_rng(456))

    assert np.array_equal(a, b)  # same seed -> identical
    assert not np.array_equal(a, c)  # different seed -> different sampling


def test_rpe_update_theta_matches_bruteforce():
    # the closed-form update must agree with an explicit O(2**m) candidate scan
    rng = np.random.default_rng(0)
    for m in range(1, 9):
        for _ in range(20):
            phi_m = rng.uniform(-np.pi, np.pi)
            theta_ref = rng.uniform(-np.pi, np.pi)

            # brute-force reference: wrapped candidate set, nearest by distance
            candidates = (phi_m + 2 * np.pi * np.arange(2**m)) / 2**m
            candidates = (candidates + np.pi) % (2 * np.pi) - np.pi
            theta_brute = candidates[np.argmin(angular_distance(candidates, theta_ref))]

            theta_fast = rpe_update_theta(phi_m, theta_ref, m)
            assert angular_distance(theta_fast, theta_brute) < 1e-12


if __name__ == "__main__":
    test_rpe()
    test_rpe_seed_deterministic()
    test_rpe_update_theta_matches_bruteforce()
