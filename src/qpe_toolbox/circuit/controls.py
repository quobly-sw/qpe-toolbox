# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------
"""Routines for quimb circuits and/or gates."""


def add_gate_controls(gates, controls):
    """
    Add control qubits to a list of gates.

    Parameters
    ----------
    gates : iterable of :quimb-api:`Gate`
        Gates to control.
    controls : sequence of int
        Control qubits, appended to the existing controls of each gate.

    Returns
    -------
    controlled_gates : list of :quimb-api:`Gate`
        New list of gate objects with the additional controls.

    Notes
    -----
    - Qubit indices are not modified; use ``shift_control_gates`` to also shift
      gates past an auxiliary register.
    - The original gate objects are not modified.
    """
    controls = tuple(controls)
    controlled_gates = []
    for g in gates:
        new_controls = controls if g.controls is None else (*g.controls, *controls)
        controlled_gates.append(g.copy_with(controls=new_controls))
    return controlled_gates


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
            f"control qubit k_ctrl={k_ctrl} outside of auxiliary register [0, {m_aux})"
        )
    shifted_gates = []
    for g in gates:
        qubits = tuple(k + m_aux for k in g.qubits)
        controls = None if g.controls is None else tuple(k + m_aux for k in g.controls)
        shifted_gates.append(g.copy_with(qubits=qubits, controls=controls))
    return add_gate_controls(shifted_gates, (k_ctrl,))
