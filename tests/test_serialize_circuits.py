#!/usr/bin/env python3

import pathlib
import tempfile
from collections import Counter

import numpy as np
import quimb.tensor as qtn
from qiskit_aer import AerSimulator

from qpe_toolbox.circuit import (
    deserialize_to_qiskit_QuantumCircuit,
    deserialize_to_quimb_Circuit,
    deserialize_to_quimb_CircuitMPS,
    dump_quimb_Circuit_to_qasm,
    generate_brickwall_circuit,
    generate_rand_circuit,
    load_qasm_to_quimb_Circuit,
    serialize_from_quimb_Circuit,
)

tol = 1e-2


def _build_circuit():
    circ = qtn.Circuit(3)
    circ.apply_gate("RX", 0.3, 0, gate_round=0)
    circ.apply_gate("CX", 0, 1, gate_round=0)
    circ.apply_gate("RZZ", 0.7, 1, 2, gate_round=1)
    return circ


def test_load_qasm_round_trip(tmp_path):
    circ = _build_circuit()
    base = str(tmp_path / "circ")
    dump_quimb_Circuit_to_qasm(circ, base, save_rounds=True)

    # default: register size read from the QASM header
    rc = load_qasm_to_quimb_Circuit(base)
    assert rc.N == 3
    assert [g.label for g in rc.gates] == ["RX", "CX", "RZZ"]
    assert [g.qubits for g in rc.gates] == [(0,), (0, 1), (1, 2)]
    overlap = abs(circ.psi.to_dense().conj().T @ rc.psi.to_dense())
    assert abs(overlap - 1.0) < 1e-8


def test_load_qasm_with_rounds(tmp_path):
    circ = _build_circuit()
    base = str(tmp_path / "circ")
    dump_quimb_Circuit_to_qasm(circ, base, save_rounds=True)

    # rounds restored from the sidecar file
    rc = load_qasm_to_quimb_Circuit(base, with_rounds=True)
    assert [g.round for g in rc.gates] == [0, 0, 1]

    # max_depth keeps only gates with round < max_depth
    rc_trunc = load_qasm_to_quimb_Circuit(base, with_rounds=True, max_depth=1)
    assert [g.round for g in rc_trunc.gates] == [0, 0]


def test_load_qasm_min_layout(tmp_path):
    circ = _build_circuit()
    base = str(tmp_path / "circ")
    dump_quimb_Circuit_to_qasm(circ, base, save_rounds=True)

    # register size inferred from the maximum qubit index in the gates
    rc = load_qasm_to_quimb_Circuit(base, min_layout=True)
    assert rc.N == 3
    assert [g.label for g in rc.gates] == ["RX", "CX", "RZZ"]


def test_build_save_load_quimb(tmp_path):
    n_qubits = 4
    depth = 2
    rng = np.random.default_rng(666)
    circ_quimb = generate_rand_circuit(n_qubits, depth, "rx", "cu3", 4, 0.75, rng=rng)
    circ_dict = serialize_from_quimb_Circuit(circ_quimb)

    base = str(tmp_path / "quimb_circuit")
    dump_quimb_Circuit_to_qasm(circ_quimb, base, save_rounds=True)
    assert (tmp_path / "quimb_circuit.qasm").exists()
    assert (tmp_path / "quimb_circuit_rounds.txt").exists()

    inferred_depth = max([gate["round"] for gate in circ_dict["gates"]]) + 1
    assert inferred_depth <= depth
    assert any(len(gate["qubits"]) == 2 for gate in circ_dict["gates"])

    circ_Circuit = deserialize_to_quimb_Circuit(circ_dict)
    circ_CircuitMPS = deserialize_to_quimb_CircuitMPS(circ_dict, 2**depth, 10e-10)

    num_samples = 10**4
    counts_Circuit = Counter(circ_Circuit.sample(C=num_samples, seed=42))
    counts_CircuitMPS = Counter(circ_CircuitMPS.sample(C=num_samples, seed=43))
    keys = counts_Circuit.keys() | counts_CircuitMPS.keys()
    diff_count = sum(abs(counts_Circuit[k] - counts_CircuitMPS[k]) for k in keys)

    assert diff_count / num_samples / len(keys) < tol


def test_sample_quimb_qiskit():
    rng = np.random.default_rng(666)
    n_qubits = 5
    depth = 2
    circ_quimb = generate_brickwall_circuit(n_qubits, depth, "rx", "cnot", rng=rng)

    circ_dict = serialize_from_quimb_Circuit(circ_quimb)
    circ_quimb = deserialize_to_quimb_CircuitMPS(
        full_gate_dict=circ_dict, max_bond=2**depth, cutoff=10e-8, perm=True
    )

    num_samples = 10**4
    circ_qiskit = deserialize_to_qiskit_QuantumCircuit(circ_dict, measure=True)
    simulator = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=2**depth,
        matrix_product_state_truncation_threshold=10e-8,
        seed_simulator=1,
    )
    result = simulator.run(circ_qiskit, shots=num_samples).result()
    count_qiskit = Counter({k[::-1]: v for k, v in result.get_counts().items()})

    count_quimb = Counter(circ_quimb.sample(C=num_samples, seed=42))
    keys = count_qiskit.keys() | count_quimb.keys()
    diff_count = sum(abs(count_qiskit[k] - count_quimb[k]) for k in keys)

    assert diff_count / num_samples / len(keys) < tol


if __name__ == "__main__":
    for _test in (
        test_load_qasm_round_trip,
        test_load_qasm_with_rounds,
        test_load_qasm_min_layout,
        test_build_save_load_quimb,
    ):
        with tempfile.TemporaryDirectory() as _d:
            _test(pathlib.Path(_d))
    test_sample_quimb_qiskit()
