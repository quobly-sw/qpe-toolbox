# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

from collections import Counter

import numpy as np
import quimb.tensor as qtn

from qpe_toolbox.circuit import make_circMPS


def build_hadamard_test_circuit(init_mps, U_gate, theta):
    r"""
    Construct the quantum circuit implementing the Hadamard test.

    The circuit prepares an ancilla qubit, applies a controlled unitary
    :math:`U`, applies a phase rotation on the ancilla, and measures the
    ancilla in the X basis.

    This circuit can be used to estimate the real or imaginary part of
    :math:`\bra{\psi} U \ket{\psi}` by choosing appropriate
    values of ``theta``.

    Parameters
    ----------
    init_mps : :quimb-api:`MatrixProductState`
        Initial state :math:`\ket{\psi}` of the data register.
    U_gate : :quimb-api:`Gate` or list
        Unitary operator to be tested.
        If a ``Gate`` instance is provided, it must act on all data qubits
        and be controlled by the ancilla qubit.
        If a list is provided, it is interpreted as a sequence of Trotter
        slices, each slice being a list of gate specifications.
    theta : float
        Phase angle applied to the ancilla qubit.
        Typical values:
        - ``0`` for estimating the real part
        - ``-π/2`` for estimating the imaginary part

    Returns
    -------
    circ : :quimb-api:`CircuitMPS`
        Circuit implementing the Hadamard test.

    """
    n_qubits = init_mps.L
    circ = make_circMPS(n_phase_bits=1, psi_mps=init_mps)
    data_reg = list(range(1, n_qubits + 1))

    circ.apply_gate("H", 0)

    if isinstance(U_gate, qtn.circuit.Gate):
        if (U_gate.controls != (0,)) or (U_gate.qubits != tuple(data_reg)):
            raise ValueError("Invalid U_gate")
        circ.apply_gate(U_gate)
    elif isinstance(U_gate, list):
        for trotter_slice in U_gate:
            for gate in trotter_slice:
                circ.apply_gate(*gate, controls=[0])
    else:
        raise TypeError("Invalid U_gate type")

    circ.apply_gate("PHASE", theta, 0)
    circ.apply_gate("H", 0)
    return circ


def run_hadamard_test(init_mps, U_gate, theta, n_shots, *, seed=42):
    r"""
    Run the Hadamard test circuit and estimate the expectation value
    :math:`Z(\theta)`.

    The returned value is

    .. math::

        Z(\theta) = P(0) - P(1)
                  = \mathrm{Re} \left[e^{i\theta} \bra{\psi} U \ket{\psi} \right]

    where :math:`P(0)` and :math:`P(1)` are the probabilities of measuring
    the ancilla qubit in states :math:`\ket{0}` and :math:`\ket{1}`.

    Parameters
    ----------
    init_mps : :quimb-api:`MatrixProductState`
        Initial state :math:`\ket{\psi}` of the data register.
    U_gate : :quimb-api:`Gate` or list
        Unitary operator used in the Hadamard test.
        See ``build_circuit`` method for accepted formats.
    theta : float
        Phase angle applied to the ancilla qubit.
    n_shots : int
        Number of measurement shots.
        - If ``0`` or ``inf``, probabilities are computed exactly.
        - If finite, probabilities are estimated by sampling.

    Returns
    -------
    Z : float
        Estimated value of :math:`Z(\theta) = P(0) - P(1)`.

    """
    circ = build_hadamard_test_circuit(init_mps, U_gate, theta)
    aux_ind = 0
    if (n_shots == 0) or np.isposinf(n_shots):
        probs = circ.compute_marginal(where=[aux_ind])
    else:
        count = Counter(circ.sample(C=n_shots, seed=seed))
        probs = [0.0, 0.0]
        for b, c in count.items():
            probs[int(b[aux_ind])] += c / n_shots
    return probs[0] - probs[1]
