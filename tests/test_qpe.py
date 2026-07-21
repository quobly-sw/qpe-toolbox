#!/usr/bin/env python3
import json
import os
import tempfile

import numpy as np
import quimb.tensor as qtn

from qpe_toolbox import EXACT
from qpe_toolbox.circuit import deserialize_to_quimb_CircuitMPS, make_circMPS
from qpe_toolbox.circuit.serialize_circuits import (
    serialize_from_quimb_Circuit,
    serialize_from_quimb_gates,
)
from qpe_toolbox.estimation import quantum_phase_estimation as qpe
from qpe_toolbox.hamiltonian import do_dmrg, heisenberg_hamiltonian
from qpe_toolbox.tensor import kron_mps

ham = heisenberg_hamiltonian(2)
E0, psi0 = do_dmrg(ham)
E_target = E0 + 0.2
size_interval = 2

e_const_shifted = 5.0
ham_shifted = heisenberg_hamiltonian(2)
ham_shifted.e_const = e_const_shifted
E_target_shifted = E0 + e_const_shifted + 0.2


def test_qpe():
    circ = make_circMPS(5, psi0)

    _, energy = qpe.qpe_energy(ham, circ, EXACT, E_target, size_interval)

    assert np.isclose(energy, -0.7375)


def test_qpe_with_e_const():
    circ = make_circMPS(5, psi0)

    _, energy = qpe.qpe_energy(
        ham_shifted, circ, EXACT, E_target_shifted, size_interval
    )

    assert np.isclose(energy, -0.7375 + e_const_shifted)


def test_resource_analysis():
    with tempfile.TemporaryDirectory() as tmp_dir:
        orig = os.getcwd()
        os.chdir(tmp_dir)
        try:
            _run_resource_analysis()
        finally:
            os.chdir(orig)


def _run_resource_analysis():
    n_phase_bits = 5

    E_max, evolution_time, global_phase = qpe.set_search_window(
        ham, E_target, size_interval
    )
    n_steps = 4
    trotter_order = 2

    filename = f"QPE_ttr{trotter_order}{n_steps}steps_{ham.n_qubits}qubits_{n_phase_bits}phbits.json"
    traces, gates_list = qpe.qpe_gate_list(
        ham,
        n_phase_bits,
        evolution_time,
        n_steps,
        global_phase,
        trotter_order=trotter_order,
        savefile=filename,
    )

    assert len(gates_list) == 4241

    c = sum(traces["gates_count"].values()) - traces["gates_count"]["SWAP"]
    assert c == len(gates_list)

    psi_init = kron_mps(qtn.MPS_computational_state("0" * n_phase_bits), psi0)

    assert os.path.exists(filename)
    with open(filename) as infile:
        gate_dict = json.load(infile)

    circ2 = deserialize_to_quimb_CircuitMPS(
        gate_dict, max_bond=0, cutoff=1e-10, psi0=psi_init
    )

    probs = circ2.compute_marginal(where=list(range(n_phase_bits)))

    max_prob_state_int = np.argmax(probs)
    theta = max_prob_state_int / 2**n_phase_bits
    energy = E_max - 2 * np.pi * theta / evolution_time

    assert np.isclose(energy, -0.7375)


def test_gate_list_matches_circuit():
    # the lazy gate list and the simulated circuit must hold the same gate sequence
    n_phase_bits = 3
    circ = make_circMPS(n_phase_bits, psi0)

    _, evolution_time, global_phase = qpe.set_search_window(
        ham, E_target, size_interval
    )
    n_trotter_steps = 2

    traces, _ = qpe.qpe_sample(ham, circ, evolution_time, n_trotter_steps, global_phase)
    _, gates_list = qpe.qpe_gate_list(
        ham, n_phase_bits, evolution_time, n_trotter_steps, global_phase
    )

    circuit_dict = serialize_from_quimb_Circuit(traces["circuit"])
    gates_dict = serialize_from_quimb_gates(circ.N, gates_list)
    assert circuit_dict == gates_dict


if __name__ == "__main__":
    test_qpe()
    test_resource_analysis()
    test_gate_list_matches_circuit()
