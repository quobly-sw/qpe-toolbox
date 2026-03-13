#!/usr/bin/env python3

import autoray
import numpy as np
import quimb.tensor as qtn

from qpe_toolbox.circuit import ansatz_circuit, ansatz_circuit_su4, ansatz_circuit_sym
from qpe_toolbox.hamiltonian import heisenberg_hamiltonian

opt = "auto-hq"
rng = np.random.default_rng(42)

n_qubits = 2
hamilt = heisenberg_hamiltonian(n_qubits)
hamilt_mpo = hamilt.to_mpo()

depth = 3


def _loss_circ(circ, mpo):
    psi = circ.psi
    psiH = psi.H
    norm_tn = psiH & psi
    psi.align_(mpo, psiH)
    energy_tn = psiH & mpo & psi
    return autoray.do("real", energy_tn.contract(all, optimize=opt)) / autoray.do(
        "real", norm_tn.contract(all, optimize=opt)
    )


def make_circuit_optimizer(circ, mpo):
    return qtn.TNOptimizer(
        circ,  # the tensor network we want to optimize
        _loss_circ,  # the function we want to minimize
        loss_constants={"mpo": mpo},  # supply U to the loss function as a constant TN
        autodiff_backend="jax",  # use 'autograd' for non-compiled optimization
        optimizer="L-BFGS-B",
        progbar=False,
    )


def test_ansatz_circuit():
    circ = ansatz_circuit(n_qubits, depth, rng=rng)
    assert len(circ.gates) == depth * ((n_qubits - 1) + n_qubits)


def test_ansatz_circuit_su4():
    circ = ansatz_circuit_su4(n_qubits, depth, rng=rng)
    c = 0
    for g in circ.gates:
        if g.label == "SU4":
            c += 1
    assert c == depth * (n_qubits - 1)


def test_ansatz_circuit_sym():
    circ = ansatz_circuit_sym(n_qubits, depth, gate_round=0, rng=rng)
    assert circ.gates[0].label == "X"
    assert circ.gates[-1].label == "RZ"


def test_ansatz_circuit_opt():
    circ = ansatz_circuit(n_qubits, 1, rng=rng)
    circuit_optimizer = make_circuit_optimizer(circ, hamilt_mpo)
    optimal_circuit = circuit_optimizer.optimize(10)
    circ = ansatz_circuit(n_qubits, 2, param_scaling=1e-4, rng=rng)
    circ.set_params(optimal_circuit.get_params())
    circuit_optimizer = make_circuit_optimizer(circ, hamilt_mpo)
    optimal_circuit = circuit_optimizer.optimize(10)

    assert np.isclose(circuit_optimizer.loss, 0.25)


if __name__ == "__main__":
    test_ansatz_circuit()
    test_ansatz_circuit_su4()
    test_ansatz_circuit_sym()
    test_ansatz_circuit_opt()
