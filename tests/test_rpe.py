#!/usr/bin/env python3

from qpe_toolbox.estimation import distance, robust_phase_estimation
from qpe_toolbox.hamiltonian import do_dmrg, heisenberg_hamiltonian


def test_rpe():
    n_qbits = 4
    H = heisenberg_hamiltonian(n_qbits)
    E0, psi0 = do_dmrg(H)
    epsilon = 0.01
    n_shots = 0

    theta_list = robust_phase_estimation(
        H, psi0, epsilon, sign_E0=-1, n_steps=0, n_shots=n_shots, verbosity=0
    )
    assert abs(distance(E0, theta_list[-1])) < epsilon


if __name__ == "__main__":
    test_rpe()
