# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------
"""Routines for quimb circuits and/or gates."""


def add_gate_controls(gates, controls, *, qubit_shift=0, gate_round=None):
    """
    Add controls to a list of gates, optionally shifting and setting their round.

    Each gate is copied once: its target and control qubits are shifted by
    ``qubit_shift``, ``controls`` are appended, and the gate round is set when
    ``gate_round`` is given.

    Parameters
    ----------
    gates : iterable of :quimb-api:`Gate`
        Gates to control.
    controls : sequence of int
        Control qubits, appended to the existing controls of each gate.
    qubit_shift : int, default ``0``
        Offset added to every target and existing-control qubit index. The
        appended ``controls`` are taken as already-final indices and not shifted.
    gate_round : int or None, default ``None``
        Gate round assigned to each copy. When ``None`` the existing round is
        left untouched.

    Returns
    -------
    controlled_gates : list of :quimb-api:`Gate`
        New list of gate objects with the additional controls.

    Notes
    -----
    - Use ``shift_control_gates`` for the common case of shifting gates past an
      auxiliary register and controlling on one of its qubits.
    - The original gate objects are not modified.
    """
    controls = tuple(controls)
    extra = {} if gate_round is None else {"round": gate_round}
    controlled_gates = []
    for g in gates:
        qubits = tuple(k + qubit_shift for k in g.qubits)
        base = () if g.controls is None else tuple(k + qubit_shift for k in g.controls)
        controlled_gates.append(
            g.copy_with(qubits=qubits, controls=(*base, *controls), **extra)
        )
    return controlled_gates


def shift_control_gates(gates, m_aux, k_ctrl, *, gate_round=None):
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
    gate_round : int or None, default ``None``
        Gate round assigned to each shifted gate. When ``None`` the existing
        round is left untouched.

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
    return add_gate_controls(gates, (k_ctrl,), qubit_shift=m_aux, gate_round=gate_round)
