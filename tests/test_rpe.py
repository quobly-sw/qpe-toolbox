#!/usr/bin/env python3

from qpe_toolbox.estimation import robust_phase_estimation, rpe_distance
from qpe_toolbox.hamiltonian import do_dmrg, heisenberg_hamiltonian


def test_rpe():
    n_qubits = 4
    H = heisenberg_hamiltonian(n_qubits)
    E0, psi0 = do_dmrg(H)
    epsilon = 0.01
    n_shots = 0

    theta_list = robust_phase_estimation(
        H, psi0, epsilon, sign_E0=-1, n_steps=0, n_shots=n_shots, verbosity=0
    )
    assert abs(rpe_distance(E0, theta_list[-1])) < epsilon


if __name__ == "__main__":
    test_rpe()
