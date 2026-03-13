#!/usr/bin/env python

import qpe_toolbox.estimation as qpe
from qpe_toolbox.circuit import count_gates_by_qb, make_circ
from qpe_toolbox.hamiltonian import do_dmrg, heisenberg_hamiltonian


def test_count():
    n_trotter_steps = 2
    n_phase_bits = 2
    n_qubits = 4

    h_spin = heisenberg_hamiltonian(n_qubits)
    exact_energy, psi0_mps = do_dmrg(h_spin)
    energy_target = exact_energy + 0.1
    size_interval = 2

    initial_circ = make_circ(n_phase_bits, psi0_mps)
    traces, _energy = qpe.qpe_energy(
        h_spin, initial_circ, n_trotter_steps, energy_target, size_interval
    )
    count = count_gates_by_qb(traces["gates_count"])
    assert count == {"1qb": 540, "2qb": 199, "3+qb": 108}


if __name__ == "__main__":
    test_count()
