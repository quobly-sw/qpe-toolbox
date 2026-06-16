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


def _qpe_stages(unitaries, global_phase, *, with_iqft):
    """
    Yield the textbook QPE circuit as a sequence of stages.

    Each stage is an iterable of :quimb-api:`Gate` objects: the Hadamard wall,
    then one controlled unitary per phase qubit, then (when ``with_iqft``) the
    inverse QFT on the phase register. Gate rounds are shared across stages so
    the flattened sequence matches the applied circuit gate-for-gate.
    """
    n_phase_bits = len(unitaries)
    yield [parse_to_gate("H", k, gate_round=0) for k in range(n_phase_bits)]
    rounds = itertools.count(1)
    for k in range(n_phase_bits):
        yield _controlled_unitary_gates(
            unitaries[k], n_phase_bits, k, global_phase, rounds
        )
    if with_iqft:
        yield (
            parse_to_gate(*gate_id, gate_round=next(rounds))
            for gate_id in iqft_swapped(list(range(n_phase_bits)))
        )


def _apply_stages(initial_circ, stages, verbosity):
    """
    Apply QPE ``stages`` to a copy of ``initial_circ``, recording per-stage traces.

    Returns the ``traces`` dict (per-stage computation times and bond
    dimensions, including the initial bond dimension) and the updated circuit.
    """
    st = time.time()
    circ = initial_circ.copy()
    bd_list = [circ.psi.max_bond()]
    ctimes = []
    for i, stage in enumerate(stages):
        for gate in stage:
            circ.apply_gate(gate)
        bd_list.append(circ.psi.max_bond())
        ctimes.append(time.time() - st)
        if verbosity >= 1:
            print(
                f"Done w/ stage {i}, elapsed {ctimes[-1]:.2f} s, bond dim {bd_list[-1]}"
            )
    return {"ctimes": ctimes, "bond_dims": bd_list}, circ


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
    for stage in _qpe_stages(unitaries, global_phase, with_iqft=True):
        yield from stage


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
        Contains per-stage computation times and bond dimensions.
    circ : :quimb-api:`Circuit` or :quimb-api:`CircuitMPS`
        Copy of ``initial_circ`` with the first stage applied.
    """
    stages = _qpe_stages(unitaries, global_phase, with_iqft=False)
    return _apply_stages(initial_circ, stages, verbosity)


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
        Contains per-stage computation times and bond dimensions.
    circ : :quimb-api:`Circuit` or :quimb-api:`CircuitMPS`
        Copy of ``initial_circ`` with the QPE circuit applied. Phase
        probabilities can be obtained with
        ``circ.compute_marginal(where=range(len(unitaries)))``.
    """
    stages = _qpe_stages(unitaries, global_phase, with_iqft=True)
    return _apply_stages(initial_circ, stages, verbosity)
