# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

import quimb.tensor as qtn


def _build_init_mps(n_phase_bits, psi_mps):
    """
    Build the initial MPS for a circuit with a phase register prepended.

    Constructs a product MPS of ``n_phase_bits`` zero qubits followed by
    the data register ``psi_mps``, with qubit indices shifted accordingly.

    Parameters
    ----------
    n_phase_bits : int
        Number of qubits in the phase (auxiliary) register.
    psi_mps : :quimb-api:`MatrixProductState`
        Initial state of the data register.

    Returns
    -------
    :quimb-api:`MatrixProductState`
        Combined MPS of total length ``n_phase_bits + psi_mps.L``.

    """
    n_qubits = psi_mps.L
    psi_aux = psi_mps.copy()

    phase_reg_mps = qtn.MPS_computational_state("0" * n_phase_bits)

    psi_aux.reindex_({f"k{i}": f"k{i + n_phase_bits}" for i in range(n_qubits)})
    psi_aux.retag_({f"I{i}": f"I{i + n_phase_bits}" for i in range(n_qubits)})

    init_mps = qtn.MatrixProductState.new(
        L=n_qubits + n_phase_bits, cyclic=False, site_ind_id="k{}", site_tag_id="I{}"
    )
    for t in phase_reg_mps.tensors:
        init_mps &= t
    for t in psi_aux.tensors:
        init_mps &= t

    return init_mps


def make_circ(n_phase_bits, psi_mps):
    """
    Initialize a quimb quantum circuit with a phase register and a data register.

    The circuit is initialized in a product state consisting of:

    * A phase (auxiliary) register of ``n_phase_bits`` qubits
      in the computational ``|0...0⟩`` state.
    * A data register initialized in the given matrix product state ``psi_mps``.

    The phase register occupies the *lowest* qubit indices, while the data
    register is shifted to higher indices.

    Parameters
    ----------
    n_phase_bits : int
        Number of qubits in the phase (auxiliary) register.
    psi_mps : :quimb-api:`MatrixProductState`
        Initial state of the data register as an MPS. Its sites will be
        reindexed and retagged to follow the phase register.

    Returns
    -------
    :quimb-api:`Circuit`
        Quantum circuit initialized with the combined MPS state.

    Notes
    -----
    - The physical qubits of ``psi_mps`` are shifted by ``n_phase_bits`` to
      avoid index collisions.
    - The resulting circuit has ``n_phase_bits + psi_mps.L`` qubits in total.

    """
    return qtn.Circuit(psi0=_build_init_mps(n_phase_bits, psi_mps))


def make_circMPS(n_phase_bits, psi_mps, *, cutoff=1e-10, max_bond=None):
    """
    Initialize a quimb ``CircuitMPS`` with a phase and data register.

    This function is equivalent to :func:`make_circ`, but returns a
    ``CircuitMPS`` object, enabling controlled bond-dimension growth
    and truncation during circuit simulation.

    Parameters
    ----------
    n_phase_bits : int
        Number of qubits in the phase (ancilla) register.
    psi_mps : :quimb-api:`MatrixProductState`
        Initial data-register state as an MPS.
    cutoff : float, optional
        Singular-value truncation threshold used during tensor compression.
        Default is ``1e-10``.
    max_bond : int or None, optional
        Maximum allowed bond dimension. If ``None``, no explicit limit
        is imposed.

    Returns
    -------
    :quimb-api:`CircuitMPS`
        Quantum circuit initialized with an MPS backend.

    Notes
    -----
    - ``CircuitMPS`` is more efficient than ``Circuit`` for large systems
      or deep circuits, at the cost of controlled approximation.
    - The phase register qubits precede the data register in qubit ordering.

    """
    return qtn.CircuitMPS(
        psi0=_build_init_mps(n_phase_bits, psi_mps), cutoff=cutoff, max_bond=max_bond
    )
