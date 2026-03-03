#!/usr/bin/env python3

import os
import tempfile
from collections import Counter

import numpy as np
from qiskit_aer import AerSimulator

from qpe_toolbox.circuit.parametrized_circuits import (
    generate_brickwall_quimb,
    generate_rand_quimb,
)
from qpe_toolbox.circuit.serialize_circuits import (
    deserialize_to_qiskit_QuantumCircuit,
    deserialize_to_quimb_Circuit,
    deserialize_to_quimb_CircuitMPS,
    dump_quimb_Circuit_to_qasm,
    serialize_from_quimb_Circuit,
)

tol = 1e-2


def test_build_save_load_quimb():
    with tempfile.TemporaryDirectory() as tmp_dir:
        orig = os.getcwd()
        os.chdir(tmp_dir)
        try:
            _run_build_save_load_quimb()
        finally:
            os.chdir(orig)


def _run_build_save_load_quimb():
    n_qubits = 4
    depth = 2
    rng = np.random.default_rng(666)
    circ_quimb = generate_rand_quimb(
        n_qubits, depth, "rx", "cu3", 4, 0.75, start_ent=True, rng=rng
    )
    circ_dict = serialize_from_quimb_Circuit(circ_quimb)

    savefile_rad = "quimb_circuit"
    dump_quimb_Circuit_to_qasm(circ_quimb, savefile_rad, save_rounds=True)
    assert os.path.exists(savefile_rad + ".qasm")
    assert os.path.exists(savefile_rad + "_rounds.txt")

    inferred_depth = max([gate["round"] for gate in circ_dict["gates"]]) + 1
    assert inferred_depth <= depth
    first_gate_dict = circ_dict["gates"][0]
    assert len(first_gate_dict["qubits"]) == 2

    circ_Circuit = deserialize_to_quimb_Circuit(circ_dict)
    circ_CircuitMPS = deserialize_to_quimb_CircuitMPS(circ_dict, 2**depth, 10e-10)

    num_samples = 10**4
    counts_Circuit = Counter(circ_Circuit.sample(C=num_samples, seed=42))
    counts_CircuitMPS = Counter(circ_CircuitMPS.sample(C=num_samples, seed=43))
    keys = counts_Circuit.keys() | counts_CircuitMPS.keys()
    diff_count = sum(abs(counts_Circuit[k] - counts_CircuitMPS[k]) for k in keys)

    assert diff_count / num_samples / len(keys) < tol


def test_sample_quimb_qiskit():
    n_qubits = 5
    depth = 2
    circ_quimb = generate_brickwall_quimb(n_qubits, depth, "rx", "cnot")

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
    test_build_save_load_quimb()
    test_sample_quimb_qiskit()
