# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

import numpy as np
import quimb.tensor as qtn


def count_gates(circ):
    """
    Count quantum gates appearing in a circuit or gate-instruction list.

    Gates are grouped by label, with controlled gates prefixed by
    ``'C'`` for each control qubit (e.g. ``'CCX'`` for a doubly controlled X).

    An estimate of the number of SWAP gates required to implement the
    circuit on a 1D nearest-neighbor architecture is also included.

    Parameters
    ----------
    circ : :quimb-api:`Circuit`, :quimb-api:`CircuitMPS`, or list
        Circuit object or list of gate instructions of the form
        ``(label, params, qubits, controls)``.

    Returns
    -------
    dict
        Dictionary mapping gate labels to their counts. The key ``'SWAP'``
        contains the estimated number of SWAP gates required due to
        connectivity constraints.

    Notes
    -----
    - SWAP counting assumes a linear qubit layout with nearest-neighbor
      connectivity.
    - For quimb circuits, gate information is extracted from ``circ.gates``.
    """
    gates_count = {"SWAP": 0}
    if isinstance(circ, list):
        gates_list = circ
    elif isinstance(circ, qtn.Circuit):
        gates_list = circ.gates
    else:
        raise TypeError("input must be Circuit or list")

    for gate in gates_list:
        prefix = "C" * len(gate.controls) if gate.controls is not None else ""
        label = prefix + gate.label
        gates_count[label] = gates_count.get(label, 0) + 1
        gates_count["SWAP"] += count_swaps(gate.qubits, gate.controls)

    return gates_count


def count_swaps(qubits, controls):
    """
    Estimate the number of SWAP gates required for a gate.

    The estimate assumes a 1D nearest-neighbor qubit layout and counts
    the minimal number of SWAPs required to bring all involved qubits
    next to a common interaction point.

    Parameters
    ----------
    qubits : int or iterable of of int or None
        Target qubit(s) of the gate.
    controls : iterable of int or None
        Control qubits.

    Returns
    -------
    int
        Estimated number of SWAP gates required.

    Notes
    -----
    - This is a heuristic estimate and does not correspond to an
      explicit routing algorithm.
    """
    if qubits is None:
        qubits = []
    elif isinstance(qubits, (np.integer, int)):
        qubits = [qubits]
    if controls is None:
        controls = []

    total_qubits = list(qubits) + list(controls)
    res = np.min(
        [
            np.sum([max(distance_qubits(i, j) - 1, 0) for j in total_qubits])
            for i in total_qubits
        ]
    )

    return int(res)


def distance_qubits(i, j):
    """
    Compute the distance between two qubits.

    This function defines the qubit connectivity metric and can be
    customized for different hardware topologies.

    Parameters
    ----------
    i, j : int
        Qubit indices.

    Returns
    -------
    int
        Distance between qubits ``i`` and ``j``.
    """
    return abs(i - j)


def count_gates_by_qb(gate_count):
    """
    Group gate counts by the number of qubits they act on.

    Gates are classified as:
    - ``'1qb'``: single-qubit gates
    - ``'2qb'``: two-qubit gates (including singly controlled gates)
    - ``'3+qb'``: gates acting on three or more qubits

    Parameters
    ----------
    gate_count : dict
        Dictionary mapping gate labels to counts as produced by
        :func:`qpe_toolbox.estimation.quantum_phase_estimation.qpe_energy`.

    Returns
    -------
    dict
        Dictionary with keys ``'1qb'``, ``'2qb'``, and ``'3+qb'``.
    """
    count = {"1qb": 0, "2qb": 0, "3+qb": 0}
    for label in gate_count:
        c = 0
        i = 0
        while label[i].upper() == "C":
            c += 1
            i += 1
        g_key = f"{c + 1}qb" if c <= 1 else "3+qb"
        count[g_key] += gate_count[label]
    return count
