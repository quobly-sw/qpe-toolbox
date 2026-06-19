#!/usr/bin/env python3
import numpy as np
import quimb.tensor as qtn

from qpe_toolbox.circuit import add_gate_controls
from qpe_toolbox.estimation import qpe_circuit, qpe_gates


def _phase_gate_powers(theta, n_phase_bits):
    """Unitaries [U^(2^k)] for U = PHASE(2*pi*theta) acting on one data qubit."""
    return [
        [qtn.Gate("PHASE", params=[2 * np.pi * theta * 2**k], qubits=[0])]
        for k in range(n_phase_bits)
    ]


def _one_data_qubit_circ(n_phase_bits):
    """CircuitMPS with the phase register in |0> and the data qubit in |1>."""
    psi0 = qtn.MPS_computational_state("0" * n_phase_bits + "1")
    return qtn.CircuitMPS(n_phase_bits + 1, psi0=psi0)


def test_qpe_circuit_phase_gate():
    # theta exactly representable on 3 bits: QPE reads it out deterministically
    m = 3
    theta = 5 / 8

    circ0 = _one_data_qubit_circ(m)
    traces, circ = qpe_circuit(circ0, _phase_gate_powers(theta, m))
    probs = np.ravel(circ.compute_marginal(where=list(range(m))))

    assert np.argmax(probs) == 5
    assert np.isclose(probs[5], 1.0)
    # one timing per stage: Hadamard wall, m controlled unitaries, IQFT
    assert len(traces["ctimes"]) == m + 2
    # initial circuit is left unmodified
    assert len(circ0.gates) == 0


def test_qpe_circuit_global_phase():
    # the global phase correction (phi * 2**k on phase qubit k) shifts the measured phase
    m = 3
    theta = 5 / 8
    shift = 2 / 8

    circ0 = _one_data_qubit_circ(m)
    _, circ = qpe_circuit(
        circ0, _phase_gate_powers(theta, m), global_phase=2 * np.pi * shift
    )
    probs = np.ravel(circ.compute_marginal(where=list(range(m))))

    assert np.argmax(probs) == 7
    assert np.isclose(probs[7], 1.0)


def test_qpe_circuit_non_squaring():
    # supplying U^(2^k) directly or as 2^k repetitions of U gives the same output
    m = 3
    theta = 3 / 8

    direct = _phase_gate_powers(theta, m)
    repeated = [
        [qtn.Gate("PHASE", params=[2 * np.pi * theta], qubits=[0])] * 2**k
        for k in range(m)
    ]

    _, circ_direct = qpe_circuit(_one_data_qubit_circ(m), direct)
    _, circ_repeated = qpe_circuit(_one_data_qubit_circ(m), repeated)

    probs_direct = np.ravel(circ_direct.compute_marginal(where=list(range(m))))
    probs_repeated = np.ravel(circ_repeated.compute_marginal(where=list(range(m))))
    assert np.allclose(probs_direct, probs_repeated)
    assert np.argmax(probs_direct) == 3


def test_qpe_gates_lazy():
    # unitaries may be one-shot generators, consumed exactly once
    m = 2
    theta = 1 / 4
    unitaries = [iter(u) for u in _phase_gate_powers(theta, m)]

    gates = list(qpe_gates(unitaries))

    # m Hadamard + m controlled phase + IQFT (m Hadamard + m(m-1)/2 CPHASE)
    assert len(gates) == 3 * m + m * (m - 1) // 2
    # data-register gates are shifted past the phase register and controlled
    # by the corresponding phase qubit
    assert gates[m].qubits == (m,)
    assert gates[m].controls == (0,)
    assert gates[m + 1].qubits == (m,)
    assert gates[m + 1].controls == (1,)
    # round is the QPE stage index: Hadamard wall 0, k-th controlled-U k+1,
    # inverse QFT n_phase_bits+1 (all gates of a stage share one layer index)
    assert gates[0].round == 0 and gates[1].round == 0
    assert gates[m].round == 1
    assert gates[m + 1].round == 2
    assert gates[-1].round == m + 1


def test_add_gate_controls():
    g = qtn.Gate("X", params=[], qubits=[2])
    (cg,) = add_gate_controls([g], [0])
    assert cg.qubits == (2,)
    assert cg.controls == (0,)
    # the original gate is not modified
    assert g.controls is None

    g2 = qtn.Gate("X", params=[], qubits=[3], controls=[1])
    (cg2,) = add_gate_controls([g2], [0, 2])
    assert cg2.controls == (1, 0, 2)
    assert g2.controls == (1,)

    # qubit_shift offsets targets and existing controls, but not the new ones;
    # gate_round is set in the same single copy
    g3 = qtn.Gate("X", params=[], qubits=[3], controls=[1])
    (cg3,) = add_gate_controls([g3], [0], qubit_shift=2, gate_round=5)
    assert cg3.qubits == (5,)
    assert cg3.controls == (3, 0)
    assert cg3.round == 5
    assert g3.qubits == (3,) and g3.controls == (1,)


if __name__ == "__main__":
    test_qpe_circuit_phase_gate()
    test_qpe_circuit_global_phase()
    test_qpe_circuit_non_squaring()
    test_qpe_gates_lazy()
    test_add_gate_controls()
