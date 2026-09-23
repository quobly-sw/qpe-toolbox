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

import time

import quimb.tensor as qtn

from qpe_toolbox.circuit import shift_control_gates

from .qft import iqft_swapped


def _generate_hadamard_wall(n_phase_bits):
    """Generate the Hadamard wall on the phase register (layer round 0)."""
    for k in range(n_phase_bits):
        yield qtn.circuit.parse_to_gate("H", k, gate_round=0)


def _generate_controlled_unitary(unitary, n_phase_bits, k_ctrl, global_phase):
    """
    Generate the gates of the unitary controlled by phase qubit ``k_ctrl``.

    Yields the optional phase correction on phase qubit ``k_ctrl``, then the
    gates of ``unitary`` shifted past the phase register and controlled by
    ``k_ctrl``. The phase correction is ``global_phase * 2 ** k_ctrl`` (the
    global phase of :math:`U^{2^k}`), omitted when ``global_phase`` is zero.
    All gates share the layer round ``k_ctrl + 1`` (the Hadamard wall is 0).
    """
    gate_round = k_ctrl + 1
    if global_phase:
        yield qtn.circuit.parse_to_gate(
            "PHASE", global_phase * 2**k_ctrl, k_ctrl, gate_round=gate_round
        )
    yield from shift_control_gates(unitary, n_phase_bits, k_ctrl, gate_round=gate_round)


def _generate_iqft(n_phase_bits):
    """Generate the inverse QFT on the phase register (layer round n_phase_bits + 1)."""
    for gate_id in iqft_swapped(list(range(n_phase_bits))):
        yield qtn.circuit.parse_to_gate(*gate_id, gate_round=n_phase_bits + 1)


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
    n_phase_bits = len(unitaries)
    yield from _generate_hadamard_wall(n_phase_bits)
    for k in range(n_phase_bits):
        yield from _generate_controlled_unitary(
            unitaries[k], n_phase_bits, k, global_phase
        )
    yield from _generate_iqft(n_phase_bits)


def qpe_circuit(
    initial_circ, unitaries, *, global_phase=0, with_iqft=True, verbosity=0
):
    """
    Apply the textbook QPE circuit to an initial circuit.

    The circuit is built from a Hadamard wall on the phase register and the
    controlled unitaries (in textbook QPE, :math:`U^{2^k}` controlled by phase
    qubit ``k``), optionally followed by the inverse QFT on the phase register.

    Parameters
    ----------
    initial_circ : :quimb-api:`Circuit` or :quimb-api:`CircuitMPS`
        Circuit preparing the trial state. The phase register occupies qubits
        ``[0, len(unitaries))``, the data register the remaining qubits.
    unitaries : sequence of iterable of :quimb-api:`Gate`
        Same convention as in ``qpe_gates``.
    global_phase : float, default ``0``
        Same convention as in ``qpe_gates``.
    with_iqft : bool, default ``True``
        If ``True``, apply the inverse QFT on the phase register after the
        controlled unitaries. If ``False``, stop after the controlled unitaries
        (the QPE first stage), leaving the phase register in the Fourier basis.
    verbosity : int, default ``0``
        Verbosity level. If >= 1, print progress and bond dimension information.

    Returns
    -------
    traces : dict
        Contains per-stage computation times and bond dimensions.
    circ : :quimb-api:`Circuit` or :quimb-api:`CircuitMPS`
        Copy of ``initial_circ`` with the QPE circuit applied. When
        ``with_iqft`` is ``True``, phase probabilities can be obtained with
        ``circ.compute_marginal(where=range(len(unitaries)))``.

    Notes
    -----
    Whether the circuit is contracted lazily or eagerly is governed by the
    *class* of ``initial_circ``, not by any argument here:

    - With a :quimb-api:`Circuit`, gates accumulate lazily into an uncontracted
      tensor network (no truncation). The returned ``circ.psi`` is that network;
      the contraction strategy is deferred to post-processing, e.g.
      ``circ.compute_marginal(..., optimize=..., rehearse=...)`` or handing
      ``circ.psi`` to a contraction-path optimizer.
    - With a :quimb-api:`CircuitMPS`, each gate is applied eagerly to the MPS
      with SVD truncation (controlled by the ``CircuitMPS`` ``max_bond`` /
      ``cutoff``), i.e. the circuit is simulated as it is built.

    In both cases the per-stage ``bond_dims`` trace is read from
    ``circ.psi.max_bond()``, which is cheap and does not force a contraction.
    """
    n_phase_bits = len(unitaries)
    st = time.time()
    circ = initial_circ.copy()
    bond_dims = [circ.psi.max_bond()]
    ctimes = []

    def apply_layer(label, gates):
        for gate in gates:
            circ.apply_gate(gate)
        bond_dims.append(circ.psi.max_bond())
        ctimes.append(time.time() - st)
        if verbosity >= 1:
            print(
                f"Done w/ {label}, elapsed {ctimes[-1]:.2f} s, bond dim {bond_dims[-1]}"
            )

    apply_layer("Hadamard wall", _generate_hadamard_wall(n_phase_bits))
    for k in range(n_phase_bits):
        apply_layer(
            f"{k}-th controlled-U",
            _generate_controlled_unitary(unitaries[k], n_phase_bits, k, global_phase),
        )
    if with_iqft:
        apply_layer("inverse QFT", _generate_iqft(n_phase_bits))

    return {"ctimes": ctimes, "bond_dims": bond_dims}, circ
