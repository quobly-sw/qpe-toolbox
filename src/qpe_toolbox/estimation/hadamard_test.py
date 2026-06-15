# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

from collections import Counter

from quimb.tensor import Gate

from qpe_toolbox import EXACT
from qpe_toolbox.circuit import make_circMPS

from .qpe_circuit import qpe_circuit


def build_hadamard_test_circuit(init_mps, unitary, theta):
    r"""
    Construct the quantum circuit implementing the Hadamard test.

    The circuit prepares an ancilla qubit, applies a phase rotation and a
    controlled unitary :math:`U`, and measures the ancilla in the X basis.
    It is the single-phase-qubit Quantum Phase Estimation circuit, built with
    ``qpe_circuit``. The phase rotation is applied before the controlled
    unitary; both commute, so the output is identical to the textbook
    ordering where the rotation comes after.

    This circuit can be used to estimate the real or imaginary part of
    :math:`\bra{\psi} U \ket{\psi}` by choosing appropriate
    values of ``theta``.

    Parameters
    ----------
    init_mps : :quimb-api:`MatrixProductState`
        Initial state :math:`\ket{\psi}` of the data register.
    unitary : :quimb-api:`Gate` or iterable of :quimb-api:`Gate`
        The unitary :math:`U`, acting on data-register-local qubit indices
        ``[0, n_data)`` without controls (same convention as in
        ``qpe_circuit``). Either a single gate (e.g. an exact :math:`U`) or its
        gate decomposition as an iterable of gates (e.g. a Trotterized
        :math:`U`). An iterable may be a one-shot generator, consumed exactly
        once.
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
    unitary_gates = [unitary] if isinstance(unitary, Gate) else unitary
    circ0 = make_circMPS(n_phase_bits=1, psi_mps=init_mps)
    _, circ = qpe_circuit(circ0, [unitary_gates], global_phases=[theta])
    return circ


def run_hadamard_test(init_mps, unitary, theta, n_shots, *, seed=None):
    r"""
    Run the Hadamard test circuit and estimate the expectation value :math:`Z(\theta)`.

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
    unitary : :quimb-api:`Gate` or iterable of :quimb-api:`Gate`
        The unitary :math:`U` used in the Hadamard test, either a single gate
        (e.g. an exact :math:`U`) or its gate decomposition as an iterable of
        gates (e.g. a Trotterized :math:`U`). See
        ``build_hadamard_test_circuit`` for the convention. A one-shot
        generator is consumed by this call; build a fresh one per call.
    theta : float
        Phase angle applied to the ancilla qubit.
    n_shots : int or qpe_toolbox.EXACT
        Number of measurement shots. If ``EXACT``, probabilities are computed exactly,
        else probabilities are estimated by sampling.
    seed : None or int, optional
        A random seed, passed to ``numpy.random.seed`` if given.

    Returns
    -------
    Z : float
        Estimated value of :math:`Z(\theta) = P(0) - P(1)`.
    """
    circ = build_hadamard_test_circuit(init_mps, unitary, theta)
    aux_ind = 0
    if n_shots is EXACT:
        probs = circ.compute_marginal(where=[aux_ind])
    else:
        count = Counter(circ.sample(C=n_shots, seed=seed))
        probs = [0.0, 0.0]
        for b, c in count.items():
            probs[int(b[aux_ind])] += c / n_shots
    return probs[0] - probs[1]
