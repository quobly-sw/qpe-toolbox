# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------
"""Trotter-Suzuki decompositions of Hamiltonian time evolution, as MPOs or gate sequences."""

import numpy as np
import quimb as qu
import quimb.tensor as qtn
from tqdm import tqdm


# --------------------------------------------------------------------------
def exp_Pauli_string_as_MPO(term, dt, n_qubits):
    r"""
    Construct the MPO representation of the unitary exponential of a Pauli string.

    Given a Hamiltonian term of the form:

        H = c * P

    where ``c`` is a scalar coefficient and ``P`` is a tensor product of Pauli
    operators acting on a subset of qubits, this function builds the Matrix
    Product Operator (MPO) corresponding to:

        exp(i * dt * c * P)

    using the identity:

        exp(i α P) = cos(α) I + i sin(α) P

    where ``P^2 = I``.

    The Pauli string is expanded to the full system size by inserting identity
    operators on inactive qubits.

    Parameters
    ----------
    term : tuple
        A tuple ``(coeff, pauli_string, active_qubits)`` where:

        - ``coeff`` (float): Scalar coefficient multiplying the Pauli string.
        - ``pauli_string`` (str): String of Pauli operators (e.g. ``"ZYXXZ"``).
        - ``active_qubits`` (list[int]): Indices of qubits where the Pauli
          operators act. The length must match ``pauli_string``.

    dt : float
        Evolution parameter (e.g. time or rotation angle).

    n_qubits : int
        Total number of qubits in the system.

    Returns
    -------
    qtn.MatrixProductOperator in lrud format
        MPO representing the operator:

            exp(i * dt * coeff * P)

        where ``P`` is the full Pauli string embedded in the ``n_qubits`` system.

    Raises
    ------
    ValueError
        If the length of ``pauli_string`` does not match the number of
        ``active_qubits``.
    """

    string_coeff, pauli_string, active_qubits = term
    id4 = qu.identity(2).reshape(1, 1, 2, 2)

    pauli_string_tensors = []
    pauli_weight = 0
    for qubit in range(n_qubits):
        if qubit in active_qubits:
            pauli_string_tensors.append(
                qu.pauli(pauli_string[pauli_weight]).reshape(1, 1, 2, 2)
            )
            pauli_weight += 1
        else:
            pauli_string_tensors.append(id4)

    # Fix the legs on the edges
    pauli_string_tensors[0] = pauli_string_tensors[0].reshape(1, 2, 2)
    pauli_string_tensors[-1] = pauli_string_tensors[-1].reshape(1, 2, 2)
    string_mpo = qtn.MatrixProductOperator(arrays=pauli_string_tensors)
    id_mpo = string_mpo.identity()

    id_mpo[0] *= np.cos(dt * string_coeff)
    string_mpo[0] *= 1j * np.sin(dt * string_coeff)

    exp_pauli_string_mpo = id_mpo.add_MPO(string_mpo)
    exp_pauli_string_mpo.compress(cutoff=1e-6, max_bond=2)

    return exp_pauli_string_mpo


def trotter1_approx_as_MPO(
    hamiltonian, dt, *, cutoff=1e-10, max_bond=None, reverse_order=False
):
    r"""
    Construct the first-order Trotter-Suzuki approximation as a Matrix Product Operator (MPO).

    This function builds an MPO representation of the first-order product
    formula for the time-evolution operator

    .. math::

        U(dt) \approx \prod_j e^{-i dt \, H_j},

    where ``hamiltonian.terms = [H_0, H_1, ..., H_{m-1}]`` is a decomposition
    of the Hamiltonian into terms that can each be exponentiated individually
    as MPOs.

    Parameters
    ----------
    hamiltonian : :class:`~src.hamiltonian.hamiltonian.Hamiltonian`
        Includes Pauli strings, positions and couplings.
    dt : float
        Time step used in the Trotter approximation.
    cutoff : float, optional
        Singular value truncation threshold used during MPO compression.
        Default is ``1e-10``.
    max_bond : int, optional
        Maximum allowed bond dimension during MPO compression.
        Default is ``None`` (no limit).
    reverse_order : bool, optional
        If ``False`` (default), terms are applied in forward order.
        after the first term. If ``True``, terms are applied in reverse index order.

    Returns
    -------
    :quimb-api:`MatrixProductOperator`
        MPO representation of the first-order Trotter approximation.
    """

    ham_terms = hamiltonian.terms
    n_qubits = hamiltonian.n_qubits

    if reverse_order:
        init_term = len(ham_terms) - 1
        trange_counter = tqdm(reversed(range(len(ham_terms) - 1)))
    else:
        init_term = 0
        trange_counter = tqdm(range(1, len(ham_terms)))

    trotter1_mpo = exp_Pauli_string_as_MPO(ham_terms[init_term], -dt, n_qubits)
    for i in trange_counter:
        new_factor_mpo = exp_Pauli_string_as_MPO(ham_terms[i], -dt, n_qubits)
        trotter1_mpo = trotter1_mpo.apply(
            new_factor_mpo, compress=True, cutoff=cutoff, max_bond=max_bond
        )
        trange_counter.set_description(
            f"{'': <4}Bond dimension (Trotter 1): {trotter1_mpo.max_bond()}"
        )

    return trotter1_mpo


def trotter2_approx_as_MPO(
    hamiltonian, dt, *, cutoff=1e-10, max_bond=None, verbosity=0
):
    r"""
    Construct the second-order symmetric Trotter-Suzuki approximation as an MPO.

    This function builds the second-order product formula

    .. math::

        U(dt) \approx U_1(dt/2)\,U_1^{\mathrm{rev}}(dt/2),

    where :math:`U_1` is the first-order Trotter approximation and
    :math:`U_1^{\mathrm{rev}}` uses the reverse operator ordering.

    Parameters
    ----------
    hamiltonian : :class:`~src.hamiltonian.hamiltonian.Hamiltonian`
        Includes Pauli strings, positions and couplings.
    dt : float or complex
        Time step used in the Trotter approximation.
    cutoff : float, optional
        Singular value truncation threshold used during MPO compression.
        Default is ``1e-10``.
    max_bond : int, optional
        Maximum allowed bond dimension during MPO compression.
        Default is ``None`` (no limit).
    verbosity : int, default ``0``
        Verbosity level. If >= 1, print progress.

    Returns
    -------
    :quimb-api:`MatrixProductOperator`
        MPO representation of the second-order Trotter approximation.
    """

    if verbosity >= 1:
        print(f"{'': <2}Building 2nd order Trotter")
        print(f"{'': <4}Building 1st order Trotter (1st half)")

    layer1_mpo = trotter1_approx_as_MPO(
        hamiltonian, dt / 2, cutoff=cutoff, max_bond=max_bond
    )

    if verbosity >= 1:
        print(rf"{'': <4}Building 1st order Trotter (2nd half)")

    layer2_mpo = trotter1_approx_as_MPO(
        hamiltonian, dt / 2, cutoff=cutoff, max_bond=max_bond, reverse_order=True
    )
    return layer1_mpo.apply(layer2_mpo, compress=True, cutoff=cutoff, max_bond=max_bond)


def trotter4_approx_as_MPO(
    hamiltonian, dt, *, cutoff=1e-10, max_bond=None, verbosity=0
):
    r"""
    Construct the fourth-order Trotter-Suzuki approximation as an MPO.

    This function implements the standard symmetric fourth-order composition

    .. math::

        U_4(dt) =
        U_2(s\,dt)\,
        U_2((1 - 2s)\,dt)\,
        U_2(s\,dt),

    where :math:`U_2` is the second-order Trotter approximation and

    .. math::

        s = \frac{1}{2 - 2^{1/3}}

    is the symmetry factor.

    Parameters
    ----------
    hamiltonian : :class:`~src.hamiltonian.hamiltonian.Hamiltonian`
        Includes Pauli strings, positions and couplings.
    dt : float
        Time step used in the Trotter approximation.
    cutoff : float, optional
        Singular value truncation threshold used during MPO compression.
        Default is ``1e-10``.
    max_bond : int, optional
        Maximum allowed bond dimension during MPO compression.
        Default is ``None`` (no limit).
    verbosity : int, default ``0``
        Verbosity level. If >= 1, print progress.

    Returns
    -------
    :quimb-api:`MatrixProductOperator`
        MPO representation of the fourth-order Trotter approximation.
    """

    sym_factor = 1.0 / (2.0 - 2 ** (1.0 / 3.0))

    if verbosity >= 1:
        print("Building 4th order Trotter")
        print(rf"{'': <2}Building 2nd order Trotter (1st and 3rd layers)")

    layer1_3_mpo = trotter2_approx_as_MPO(
        hamiltonian,
        dt * sym_factor,
        cutoff=cutoff,
        max_bond=max_bond,
        verbosity=verbosity,
    )

    if verbosity >= 1:
        print(rf"{'': <2}Building 2nd order Trotter (2nd layer)")

    layer2_mpo = trotter2_approx_as_MPO(
        hamiltonian,
        dt * (1 - 2 * sym_factor),
        cutoff=cutoff,
        max_bond=max_bond,
        verbosity=verbosity,
    )

    if verbosity >= 1:
        print(f"{'': <2}Multiplying the 3 MPO layers")

    trotter4_mpo = layer1_3_mpo.apply(
        layer2_mpo, compress=True, cutoff=cutoff, max_bond=max_bond
    )
    trotter4_mpo = trotter4_mpo.apply(
        layer1_3_mpo, compress=True, cutoff=cutoff, max_bond=max_bond
    )
    if verbosity >= 1:
        print(f"{'': <4}Final bond dimension:", trotter4_mpo.max_bond())

    return trotter4_mpo


def trotter_approx_as_MPO(
    hamiltonian, dt, *, trotter_order=1, cutoff=1e-10, max_bond=None, verbosity=0
):
    r"""
    Construct a Trotter-Suzuki approximation of a Hamiltonian evolution operator as an MPO.

    This function dispatches to a specific Trotter-Suzuki decomposition
    according to the requested approximation order. The Hamiltonian is assumed
    to provide a decomposition into elementary terms through ``hamiltonian.terms``.

    Depending on ``trotter_order``, the approximation is built using:

    .. math::

        U(dt) \approx
        \begin{cases}
            \text{1st-order product formula}, & \text{if trotter\_order} = 1, \\
            \text{2nd-order symmetric formula}, & \text{if trotter\_order} = 2, \\
            \text{4th-order Suzuki formula}, & \text{if trotter\_order} = 4.
        \end{cases}

    Parameters
    ----------
    hamiltonian : :class:`~src.hamiltonian.hamiltonian.Hamiltonian`
        Includes Pauli strings, positions and couplings.
    dt : float
        Time step used in the Trotter approximation.
    trotter_order : {1, 2, 4}, optional
        Order of the Trotter-Suzuki decomposition. Default is ``1``.
    cutoff : float, optional
        Singular value truncation threshold used during MPO compression.
        Default is ``1e-10``.
    max_bond : int, optional
        Maximum allowed MPO bond dimension during compression.
        Default is ``None`` (no limit).
    verbosity : int, default ``0``
        Verbosity level. If >= 1, print progress.

    Returns
    -------
    :quimb-api:`MatrixProductOperator`
        Compressed MPO representation of the Trotterized time-evolution operator.

    Raises
    ------
    ValueError
        If the requested ``trotter_order`` is not implemented.
    """
    if trotter_order == 1:
        return trotter1_approx_as_MPO(hamiltonian, dt, cutoff=cutoff, max_bond=max_bond)

    if trotter_order == 2:
        return trotter2_approx_as_MPO(
            hamiltonian, dt, cutoff=cutoff, max_bond=max_bond, verbosity=verbosity
        )

    if trotter_order == 4:
        return trotter4_approx_as_MPO(
            hamiltonian, dt, cutoff=cutoff, max_bond=max_bond, verbosity=verbosity
        )
    raise ValueError(f"Order {trotter_order} not implemented")


def rotation_gates(term, dt, qubit_reg):
    """
    Generate a gate sequence for exponentiating a Pauli-string term.

    Implements

    .. math::
        e^{-i dt \\theta P}

    where :math:`P` is a tensor product of Pauli operators and :math:`\\theta` is the associated
    coefficient in the term. The implementation uses basis rotations, ``CNOT`` chains, and a
    single ``RZ`` rotation.

    Parameters
    ----------
    term : tuple
        Hamiltonian term ``(theta, pauli_string, qubits)``.
    dt : float
        Time step or Trotter slice.
    qubit_reg : sequence of int
        Mapping from logical qubit indices to circuit qubits.

    Returns
    -------
    list
        Abstract quantum gate instructions suitable for circuit construction.
    """
    (theta, pauli_string, qubits) = term
    routine = []

    # Rotations: H for X gates and RX(pi/2) for Y gates
    for op, qubit in zip(pauli_string, qubits, strict=True):
        if op.upper() == "X":
            routine.append(("H", qubit_reg[qubit]))
        if op.upper() == "Y":
            routine.append(("RX", np.pi / 2, qubit_reg[qubit]))

    # CNOTs
    for j in range(len(pauli_string) - 1):
        routine.append(("CNOT", qubit_reg[qubits[j]], qubit_reg[qubits[j + 1]]))

    # RZ gate
    routine.append(
        ("RZ", 2 * theta * dt, qubit_reg[qubits[-1]])
    )  ## RZ(alpha) = exp(-1j * alpha/2 * sigma_z)

    # CNOTs back
    for j in range(len(pauli_string) - 1, 0, -1):
        routine.append(("CNOT", qubit_reg[qubits[j - 1]], qubit_reg[qubits[j]]))

    # Rotations back
    for op, qubit in zip(pauli_string, qubits, strict=True):
        if op.upper() == "X":
            routine.append(("H", qubit_reg[qubit]))

        if op.upper() == "Y":
            routine.append(("RX", -np.pi / 2, qubit_reg[qubit]))

    return routine
