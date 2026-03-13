# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

import numpy as np


def qft_swapped(wires):
    """
    Generate the sequence of gates for a recursive QFT with reversed ordering.

    The QFT is implemented recursively without final swaps.
    This routine is suitable for constructing circuits with qubit indices
    in reversed order.

    Parameters
    ----------
    wires : list[int]
        List of qubit indices on which to apply the QFT.

    Returns
    -------
    list[tuple]
        List of gate instructions in the form
        ``(gate_label, *params, qubits...)``, where single-qubit gates are
        ``('H', q)``, and controlled-phase gates are
        ``('CPHASE', angle, target, control)``.

    Notes
    -----
    - Recursive implementation.
    - No final SWAP gates are included.
    - Use ``qft`` for standard bit ordering with swaps.
    """
    routine = [("H", wires[0])]
    n = len(wires)

    # If there is only one qubit, then the QFT is a simple H gate
    if n == 1:
        return routine

    for i in range(1, n):
        routine.append(("CPHASE", np.pi / 2**i, wires[i], wires[0]))
    routine += qft_swapped(wires[1:])

    return routine


def iqft_swapped(wires):
    """
    Generate the inverse QFT gate sequence with reversed bit ordering.

    Parameters
    ----------
    wires : list[int]
        List of qubit indices on which to apply the inverse QFT.

    Returns
    -------
    list[tuple]
        Gate instructions for the inverse QFT, in the same format as ``qft_sw``.

    Notes
    -----
    - Controlled-phase gates have their angles negated relative to ``qft_sw``.
    - No final SWAP gates are included.
    """
    qftsw_routine = qft_swapped(wires)
    depth = len(qftsw_routine)
    result = []
    for i in range(depth - 1, -1, -1):
        gate = qftsw_routine[i]
        if gate[0] == "CPHASE":
            result.append(("CPHASE", -gate[1], gate[2], gate[3]))
        else:
            result.append(gate)
    return result


def qft(wires):
    """
    Generate the full Quantum Fourier Transform (QFT) gate sequence.

    Parameters
    ----------
    wires : list[int]
        List of qubit indices on which to apply the QFT.

    Returns
    -------
    list[tuple]
        Gate instructions including Hadamard, controlled-phase, and final
        SWAP gates for normal bit ordering.

    Notes
    -----
    - Combines ``qft_sw`` with SWAP gates to reorder qubits to standard output order.
    """
    routine = qft_swapped(wires)
    n = len(wires)

    for i in range(n // 2):
        routine.append(("SWAP", wires[i], wires[-i - 1]))

    return routine


def iqft(wires):
    """
    Generate the inverse Quantum Fourier Transform (IQFT) gate sequence.

    Parameters
    ----------
    wires : list[int]
        List of qubit indices on which to apply the IQFT.

    Returns
    -------
    list[tuple]
        Gate instructions for the inverse QFT, including SWAPs.

    Notes
    -----
    - The angles of controlled-phase gates are negated relative to the QFT.
    - Bit ordering should be checked if interfacing with other routines.
    """

    qft_routine = qft(wires)
    depth = len(qft_routine)
    iqft_routine = []

    for i in range(depth - 1, -1, -1):
        gate = qft_routine[i]
        if gate[0] == "CPHASE":
            iqft_routine.append(("CPHASE", -gate[1], gate[2], gate[3]))
        else:
            iqft_routine.append(gate)

    return iqft_routine


def count_gates_qft_swapped(m):
    """
    Compute the number of gates in the recursive QFT (reversed bit ordering).

    Parameters
    ----------
    m : int
        Number of qubits in the QFT.

    Returns
    -------
    dict[str, int]
        Dictionary with keys:
        - ``'H'`` : number of Hadamard gates
        - ``'CPHASE'`` : number of controlled-phase gates

    Notes
    -----
    - SWAP gates are **not** counted.
    - Gate count corresponds to the reversed-output QFT.
    """
    return {"H": m, "CPHASE": (m * (m - 1)) // 2}
