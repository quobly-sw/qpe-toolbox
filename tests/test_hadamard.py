#!/usr/bin/env python3
import numpy as np
import quimb.tensor as qtn

from qpe_toolbox import EXACT
from qpe_toolbox.estimation import run_hadamard_test


def _phase_unitary(phi):
    # one-shot generator for U = diag(1, e^{i phi}) on a single data qubit
    return iter([qtn.Gate("PHASE", params=[phi], qubits=[0])])


def test_hadamard_test_phase_gate():
    # |1> is an eigenstate of U with eigenvalue e^{i phi}: X + iY = e^{i phi}
    phi = np.pi / 6
    psi = qtn.MPS_computational_state("1")

    X = run_hadamard_test(psi, _phase_unitary(phi), 0, EXACT)
    Y = run_hadamard_test(psi, _phase_unitary(phi), -np.pi / 2, EXACT)
    assert np.isclose(X + 1j * Y, np.exp(1j * phi))


def test_hadamard_test_shots():
    phi = np.pi / 6
    psi = qtn.MPS_computational_state("1")

    n_shots = 1000
    X = run_hadamard_test(psi, _phase_unitary(phi), 0, n_shots, seed=42)
    Y = run_hadamard_test(psi, _phase_unitary(phi), -np.pi / 2, n_shots, seed=42)
    # statistical error scales as 1 / sqrt(n_shots)
    assert abs(X - np.cos(phi)) < 5 / np.sqrt(n_shots)
    assert abs(Y - np.sin(phi)) < 5 / np.sqrt(n_shots)


if __name__ == "__main__":
    test_hadamard_test_phase_gate()
    test_hadamard_test_shots()
