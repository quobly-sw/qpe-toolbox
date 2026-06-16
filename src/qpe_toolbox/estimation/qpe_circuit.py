# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------
"""
Generic construction of Quantum Phase Estimation circuits from arbitrary unitaries.

The functions in this module are agnostic to how the controlled unitaries are
obtained. The ``unitaries`` input is a sequence of length ``n_phase_bits``, where
``unitaries[k]`` is an iterable of uncontrolled :quimb-api:`Gate` objects
implementing the unitary controlled by phase qubit ``k`` (in textbook QPE,
:math:`U^{2^k}`), acting on data-register-local qubit indices ``[0, n_data)``.
Entries may be lists or one-shot generators, enabling lazy circuit construction.
Each power can be supplied as an independent gate decomposition, so
non-squaring implementations (e.g. modular exponentiation in Shor's algorithm)
are supported.
"""

import itertools
import time

from quimb.tensor.circuit import parse_to_gate

from qpe_toolbox.circuit import shift_control_gates

from .qft import iqft_swapped


def _controlled_unitary_gates(unitary, n_phase_bits, k_ctrl, global_phase, rounds):
    """
    Generate the gates of one controlled-unitary stage of QPE.

    Yields the optional phase correction on phase qubit ``k_ctrl``, then the
    gates of ``unitary`` shifted past the phase register and controlled by
    ``k_ctrl``. The phase correction is ``global_phase * 2 ** k_ctrl`` (the
    global phase of :math:`U^{2^k}`), omitted when ``global_phase`` is zero.
    Gate rounds are drawn from the shared ``rounds`` counter.
    """
    if global_phase:
        yield parse_to_gate(
            "PHASE", global_phase * 2**k_ctrl, k_ctrl, gate_round=next(rounds)
        )
    for gate in unitary:
        (cgate,) = shift_control_gates((gate,), n_phase_bits, k_ctrl)
        yield cgate.copy_with(round=next(rounds))


def _first_stage_gates_iter(unitaries, global_phase):
    n_phase_bits = len(unitaries)
    for k in range(n_phase_bits):
        yield parse_to_gate("H", k, gate_round=0)
    rounds = itertools.count(1)
    for k in range(n_phase_bits):
        yield from _controlled_unitary_gates(
            unitaries[k], n_phase_bits, k, global_phase, rounds
        )


def qpe_gates(unitaries, *, global_phase=0):
    """
    Generate the full textbook QPE gate sequence without simulating it.

    The sequence is the QPE first stage (Hadamard wall and controlled
    unitaries) followed by the inverse QFT on the phase register. Suitable for
    resource analysis and serialization.

    Parameters
    ----------
    unitaries : sequence of iterable of :quimb-api:`Gate`
        ``unitaries[k]`` is the gate decomposition of the unitary controlled by
        phase qubit ``k`` (in textbook QPE, :math:`U^{2^k}`), acting on
        data-register-local qubit indices ``[0, n_data)`` without controls.
        Entries may be one-shot generators; each is consumed exactly once.
    global_phase : float, default ``0``
        Global phase :math:`\\phi` of the unitary :math:`U`. The phase
        correction applied before the k-th controlled unitary is
        :math:`\\phi \\, 2^k` (the global phase of :math:`U^{2^k}`), and the
        whole correction is omitted when ``global_phase`` is zero.

    Returns
    -------
    generator of :quimb-api:`Gate`
        Lazy gate sequence of the full QPE circuit.
    """
    c_round = 0
    for gate in _first_stage_gates_iter(unitaries, global_phase):
        c_round = gate.round
        yield gate
    phase_reg = list(range(len(unitaries)))
    for gate_id in iqft_swapped(phase_reg):
        c_round += 1
        yield parse_to_gate(*gate_id, gate_round=c_round)


def qpe_first_stage_circuit(initial_circ, unitaries, *, global_phase=0, verbosity=0):
    """
    Apply the QPE first stage (Hadamard wall and controlled unitaries) to a circuit.

    Parameters
    ----------
    initial_circ : :quimb-api:`Circuit` or :quimb-api:`CircuitMPS`
        Circuit preparing the trial state. The phase register occupies qubits
        ``[0, len(unitaries))``, the data register the remaining qubits.
    unitaries : sequence of iterable of :quimb-api:`Gate`
        Same convention as in ``qpe_gates``.
    global_phase : float, default ``0``
        Same convention as in ``qpe_gates``.
    verbosity : int, default ``0``
        Verbosity level. If >= 1, print progress and bond dimension information.

    Returns
    -------
    traces : dict
        Contains computation times, bond dimensions and the next gate round.
    circ : :quimb-api:`Circuit` or :quimb-api:`CircuitMPS`
        Copy of ``initial_circ`` with the first stage applied.
    """
    n_phase_bits = len(unitaries)
    st = time.time()
    ctimes = []
    circ = initial_circ.copy()
    bd_list = [circ.psi.max_bond()]

    # Hadamard wall
    for k in range(n_phase_bits):
        circ.apply_gate("H", k, gate_round=0)
    bd_list.append(circ.psi.max_bond())
    ctimes.append(time.time() - st)

    if verbosity >= 1:
        print(f"Start C-Us, elapsed {ctimes[-1]:.2f} s, bond dim {bd_list[-1]}")

    # Controlled unitaries
    rounds = itertools.count(1)
    for k in range(n_phase_bits):
        for gate in _controlled_unitary_gates(
            unitaries[k], n_phase_bits, k, global_phase, rounds
        ):
            circ.apply_gate(gate)
        bd_list.append(circ.psi.max_bond())
        ctimes.append(time.time() - st)
        if verbosity >= 1:
            print(
                f"Done w/ {k}-th C-U, elapsed {ctimes[-1]:.2f} s, bond dim {bd_list[-1]}"
            )

    traces = {"ctimes": ctimes, "bond_dims": bd_list, "gate_round": next(rounds)}
    return traces, circ


def qpe_circuit(initial_circ, unitaries, *, global_phase=0, verbosity=0):
    """
    Apply the full textbook QPE circuit to an initial circuit.

    The circuit is built from three components: a Hadamard wall on the phase
    register, the controlled unitaries (in textbook QPE, :math:`U^{2^k}`
    controlled by phase qubit ``k``), and the inverse QFT on the phase
    register.

    Parameters
    ----------
    initial_circ : :quimb-api:`Circuit` or :quimb-api:`CircuitMPS`
        Circuit preparing the trial state. The phase register occupies qubits
        ``[0, len(unitaries))``, the data register the remaining qubits.
    unitaries : sequence of iterable of :quimb-api:`Gate`
        Same convention as in ``qpe_gates``.
    global_phase : float, default ``0``
        Same convention as in ``qpe_gates``.
    verbosity : int, default ``0``
        Verbosity level. If >= 1, print progress and bond dimension information.

    Returns
    -------
    traces : dict
        Contains computation times, bond dimensions and the next gate round.
    circ : :quimb-api:`Circuit` or :quimb-api:`CircuitMPS`
        Copy of ``initial_circ`` with the QPE circuit applied. Phase
        probabilities can be obtained with
        ``circ.compute_marginal(where=range(len(unitaries)))``.
    """
    st = time.time()
    traces, circ = qpe_first_stage_circuit(
        initial_circ, unitaries, global_phase=global_phase, verbosity=verbosity
    )

    phase_reg = list(range(len(unitaries)))
    c_round = traces["gate_round"]
    for gate_id in iqft_swapped(phase_reg):
        circ.apply_gate(*gate_id, gate_round=c_round)
        c_round += 1
    traces["gate_round"] = c_round
    traces["bond_dims"].append(circ.psi.max_bond())
    traces["ctimes"].append(time.time() - st)

    return traces, circ
