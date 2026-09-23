#!/usr/bin/env python3

import numpy as np
import quimb.tensor as qtn
import scipy.linalg

from qpe_toolbox import EXACT
from qpe_toolbox.estimation import (
    angular_distance,
    robust_phase_estimation,
    rpe_update_theta,
)
from qpe_toolbox.hamiltonian import Hamiltonian, do_dmrg, heisenberg_hamiltonian


def test_rpe():
    n_qubits = 4
    H = heisenberg_hamiltonian(n_qubits)
    E0, psi0 = do_dmrg(H)
    n_repetitions = 7

    # |E0 * t0| < pi for both values, so the m=0 phase is unambiguous
    for t0 in [1.0, 0.5]:
        theta_list = robust_phase_estimation(
            H, psi0, n_repetitions, EXACT, EXACT, t0=t0
        )
        assert abs(angular_distance(E0 * t0, theta_list[-1])) < 2**-n_repetitions


def _product_formula(hamiltonian, terms_and_steps):
    """Dense product of exp(-i dt h_j), applied in the given order."""
    unitary = np.eye(2**hamiltonian.n_qubits)
    for term, dt in terms_and_steps:
        h = Hamiltonian([term], hamiltonian.n_qubits).to_dense()
        unitary = scipy.linalg.expm(-1j * dt * h) @ unitary
    return unitary


def test_rpe_trotter():
    # use custom Hamiltonian to get non-commuting terms even on 2 sites
    # non-commuting terms: the Trotter eigenphase differs from E0 * t0 (by 1.5e-2)
    # and between orders 1 and 2 (by 3.7e-3)
    terms = [(0.7, "ZZ", [0, 1]), (0.5, "X", [0]), (0.4, "X", [1]), (0.3, "Z", [0])]
    hamilt = Hamiltonian(terms, 2)
    E0 = np.linalg.eigvalsh(hamilt.to_dense())[0]
    t0 = 1.0
    n_trotter_steps = 2
    n_repetitions = 5

    dt = t0 / n_trotter_steps
    first_order = [(term, dt) for term in terms]
    half_steps = [(term, dt / 2) for term in terms]
    slices = {1: first_order, 2: half_steps + half_steps[::-1]}
    for trotter_order, trotter_slice in slices.items():
        unitary = np.linalg.matrix_power(
            _product_formula(hamilt, trotter_slice), n_trotter_steps
        )
        eigvals, eigvecs = np.linalg.eig(unitary)
        phases = -np.angle(eigvals)
        k = np.argmin(angular_distance(phases, E0 * t0))
        psi0 = qtn.MatrixProductState.from_dense(eigvecs[:, k])

        theta_list = robust_phase_estimation(
            hamilt,
            psi0,
            n_repetitions,
            n_trotter_steps,
            EXACT,
            t0=t0,
            trotter_order=trotter_order,
        )
        # psi0 is an eigenstate of the Trotterized unitary: with exact
        # probabilities, RPE recovers its eigenphase up to float32 precision
        assert angular_distance(phases[k], theta_list[-1]) < 1e-5


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
    test_rpe_trotter()
    test_rpe_seed_deterministic()
    test_rpe_update_theta_matches_bruteforce()
