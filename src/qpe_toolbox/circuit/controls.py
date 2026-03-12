# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

"""
Routines for quimb circuits and/or gates
"""


def shift_control_gates(gates, m_aux, k_ctrl):
    """
    Shift gate targets and controls to account for an auxiliary qubit register.

    This function is used when an auxiliary (ancilla) register occupies the
    first ``m_aux`` qubits of a circuit. All gates acting on the data register
    are shifted by ``m_aux`` in their qubit indices, and an additional control
    qubit from the auxiliary register is added to each gate.

    Parameters
    ----------
    gates : iterable of :quimb-api:`Gate`
        List of gates to shift.
    m_aux : int
        Number of auxiliary qubits occupying the lowest indices ``[0, m_aux - 1]``.
    k_ctrl : int
        Index of the control qubit within the auxiliary register.
        Must satisfy ``0 <= k_ctrl < m_aux``.

    Returns
    -------
    controlled_gates : list of :quimb-api:`Gate`
        New list of gate objects with shifted qubit indices and added control.

    Raises
    ------
    ValueError
        If ``k_ctrl`` lies outside the auxiliary register.

    Notes
    -----
    - Target qubits ``q`` are mapped to ``q + m_aux``.
    - Existing control qubits are also shifted by ``m_aux``.
    - If a gate originally has no controls, it becomes singly controlled
      by ``k_ctrl``.
    - The original gate objects are not modified.

    """
    if not 0 <= k_ctrl < m_aux:
        raise ValueError(
            f"control qubit k_ctrl={k_ctrl} outside of auxiliary register [0,{m_aux}["
        )
    controlled_gates = []
    for g in gates:
        qubits = tuple(k + m_aux for k in g.qubits)
        controls = (
            (k_ctrl,)
            if g.controls is None
            else (*(k + m_aux for k in g.controls), k_ctrl)
        )
        controlled_gates.append(g.copy_with(qubits=qubits, controls=controls))
    return controlled_gates
