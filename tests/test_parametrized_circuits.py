#!/usr/bin/env python3

import autoray
import numpy as np
import pytest
import quimb.tensor as qtn

from qpe_toolbox.circuit import (
    ansatz_circuit,
    ansatz_circuit_su4,
    ansatz_circuit_sym,
    generate_brickwall_circuit,
    generate_rand_circuit,
)
from qpe_toolbox.circuit.parametrized_circuits import (
    one_qubit_layer,
    two_qubit_nn_layer,
    two_qubit_rand_layer,
)
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
        if g.label == "SU4SWAP":
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


def test_one_qubit_layer():
    # parametrized gate, rng defaulted internally
    circ = qtn.Circuit(3)
    one_qubit_layer(circ, "RX")
    assert len(circ.gates) == 3

    # non-parametrized gate
    circ = qtn.Circuit(3)
    one_qubit_layer(circ, "H")
    assert len(circ.gates) == 3

    with pytest.raises(KeyError, match="Unknown gate_label"):
        one_qubit_layer(qtn.Circuit(2), "NOTAGATE")


def test_two_qubit_nn_layer():
    # parametrized gate, rng defaulted internally
    circ = qtn.Circuit(4)
    two_qubit_nn_layer(circ, 0, "RZZ")
    assert len(circ.gates) == 2

    # non-parametrized gate, odd start
    circ = qtn.Circuit(4)
    two_qubit_nn_layer(circ, 1, "CNOT")
    assert len(circ.gates) == 1

    with pytest.raises(KeyError, match="Unknown gate_label"):
        two_qubit_nn_layer(qtn.Circuit(2), 0, "NOTAGATE")


def test_two_qubit_rand_layer():
    # parametrized gate
    circ = qtn.Circuit(4)
    two_qubit_rand_layer(circ, "RZZ", 2, 1.0, rng=np.random.default_rng(42))
    assert len(circ.gates) > 0

    # non-parametrized gate, rng defaulted internally
    circ = qtn.Circuit(4)
    two_qubit_rand_layer(circ, "CNOT", 2, 1.0)

    # zero probability applies nothing
    circ = qtn.Circuit(4)
    two_qubit_rand_layer(circ, "CNOT", 2, 0.0, rng=np.random.default_rng(1))
    assert len(circ.gates) == 0

    with pytest.raises(KeyError, match="Unknown gate_label"):
        two_qubit_rand_layer(qtn.Circuit(2), "NOTAGATE", 1, 1.0)

    # a one-qubit parametrized gate cannot be applied to a pair of qubits
    with pytest.raises(KeyError, match="Unknown gate_label"):
        two_qubit_rand_layer(
            qtn.Circuit(4), "RX", 2, 1.0, rng=np.random.default_rng(42)
        )


def test_generate_brickwall_circuit():
    # valid build, rng defaulted internally
    circ = generate_brickwall_circuit(4, 2, "H", "CNOT")
    assert circ.N == 4
    assert len(circ.gates) > 0

    # explicit rng, and entangling-only layers (no single-body gates)
    entangling_only = generate_brickwall_circuit(
        4, 2, "H", "CNOT", include_1qubit_gates=False, rng=np.random.default_rng(42)
    )
    assert all(g.label == "CNOT" for g in entangling_only.gates)

    with pytest.raises(ValueError, match="single-body gate"):
        generate_brickwall_circuit(4, 1, "CNOT", "CNOT")
    with pytest.raises(ValueError, match="two-body gate"):
        generate_brickwall_circuit(4, 1, "H", "H")


def test_generate_rand_circuit():
    # valid build, rng defaulted internally
    circ = generate_rand_circuit(4, 2, "RX", "RZZ", 2, 1.0)
    assert circ.N == 4

    # explicit rng
    circ = generate_rand_circuit(
        4, 2, "RX", "RZZ", 2, 1.0, rng=np.random.default_rng(42)
    )
    assert circ.N == 4

    with pytest.raises(ValueError, match="single-body gate"):
        generate_rand_circuit(4, 1, "CNOT", "RZZ", 2, 1.0)
    with pytest.raises(ValueError, match="two-body gate"):
        generate_rand_circuit(4, 1, "RX", "RX", 2, 1.0)


def test_ansatz_circuits_with_psi0():
    # psi0 branch + rng defaulted internally, for the three ansatz builders
    for builder in (ansatz_circuit, ansatz_circuit_su4, ansatz_circuit_sym):
        # identical seeded gates on orthogonal inputs must stay orthogonal: this
        # fails if psi0 is used only for its size and its amplitudes are dropped
        circ_00 = builder(
            2, 1, psi0=qtn.MPS_computational_state("00"), rng=np.random.default_rng(42)
        )
        circ_11 = builder(
            2, 1, psi0=qtn.MPS_computational_state("11"), rng=np.random.default_rng(42)
        )
        assert abs(circ_00.psi.overlap(circ_11.psi)) < 1e-10

        with pytest.raises(ValueError, match="expected n_qubits=2"):
            builder(2, 1, psi0=qtn.MPS_computational_state("000"))


def test_ansatz_circuit_sym_gate_round():
    # gate_round != 0 skips the initial X layer applied at gate_round == 0
    circ = ansatz_circuit_sym(2, 1, gate_round=1, rng=np.random.default_rng(42))
    assert all(g.label != "X" for g in circ.gates)


if __name__ == "__main__":
    test_ansatz_circuit()
    test_ansatz_circuit_su4()
    test_ansatz_circuit_sym()
    test_ansatz_circuit_opt()
    test_one_qubit_layer()
    test_two_qubit_nn_layer()
    test_two_qubit_rand_layer()
    test_generate_brickwall_circuit()
    test_generate_rand_circuit()
    test_ansatz_circuits_with_psi0()
    test_ansatz_circuit_sym_gate_round()
