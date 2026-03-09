# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

import numpy as np
import quimb as qu
import quimb.tensor as qtn
from quimb.operator import SparseOperatorBuilder


def heisenberg_hamiltonian(n_qubits, *, coupling_strength=1, spin=1 / 2):
    """
    Construct a 1D nearest-neighbor Heisenberg Hamiltonian with open boundaries.

    The Hamiltonian is given by

    .. math::
        H = \\sum_{i=0}^{N-2} \\frac{J s}{2}
        (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})

    where only spin-1/2 systems are supported.

    Parameters
    ----------
    n_qubits : int
        Number of spins (qubits) in the chain.
    coupling_strength : float, optional
        Exchange coupling constant :math:`J`. Default is 1.
    spin : float, optional
        Spin quantum number :math:`s`. Only ``spin=1/2`` is implemented.

    Returns
    -------
    Hamiltonian
        Heisenberg Hamiltonian represented as a qubit operator.

    Raises
    ------
    ValueError
        If ``spin`` is not equal to 1/2.

    """
    terms = []
    if spin != 1 / 2:
        raise ValueError(f"spin {spin} not implemented. Defined only for spin 1/2")
    for i in range(n_qubits - 1):
        for op in ["xx", "yy", "zz"]:
            terms.append((1 / 2.0 * coupling_strength * spin, op, [i, i + 1]))
    return Hamiltonian(terms, n_qubits)


def do_dmrg(hamiltonian):
    """
    Perform a DMRG ground-state calculation using quimb.

    Based on quimb.tensor.DMRG2

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian to be converted into an MPO.

    Returns
    -------
    E0 : float
        Ground-state energy.
    psi0 : quimb.tensor.MatrixProductState
        Ground-state wavefunction as a Matrix Product State.

    """
    mpo = hamiltonian.to_mpo()
    dmrg = qtn.DMRG2(mpo)
    dmrg.solve()
    E0 = np.real(dmrg.energy)
    psi0 = dmrg.state
    psi0.permute_arrays("lrp")

    return E0, psi0


class Hamiltonian:
    """
    Qubit Hamiltonian represented as a sum of Pauli strings.

    This class provides utilities to convert the Hamiltonian into dense,
    sparse, or MPO representations, as well as to construct
    exact or Trotterized time-evolution operators.

    Parameters
    ----------
    terms : list of tuple
        Hamiltonian terms in the form
        ``(coefficient, pauli_string, qubits)``, e.g. ``(0.5, "xy", [0, 1])``.
    n_qubits : int
        Total number of qubits.

    """

    def __init__(self, terms, n_qubits):
        self.terms = terms
        self.n_qubits = n_qubits

    def to_dense(self):
        """
        Convert the Hamiltonian to a dense matrix representation.

        Returns
        -------
        quimb.qarray
            Dense Hermitian matrix of shape ``(2**n_qubits, 2**n_qubits)``.

        """
        h_dense = np.zeros([2**self.n_qubits, 2**self.n_qubits], dtype="complex")
        for coeff, paulis, qubits in self.terms:
            ops = [qu.identity(2)] * self.n_qubits
            for sigma, k in zip(paulis, qubits, strict=True):
                ops[k] = qu.pauli(sigma)
            h_dense += coeff * qu.kron(*ops)
        return qu.qarray(h_dense)

    def to_builder(self):
        """
        Convert the Hamiltonian to a sparse operator builder.

        Returns
        -------
        quimb.operator.SparseOperatorBuilder
            Builder object that can generate sparse matrices or MPOs.

        """
        builder = SparseOperatorBuilder()
        for coeff, paulis, qubits in self.terms:
            builder.add_term(
                coeff,
                *(
                    (sigma.lower(), k)
                    for (sigma, k) in zip(paulis, qubits, strict=True)
                ),
            )
        return builder

    def to_mpo(self):
        """
        Convert the Hamiltonian into a matrix product operator (MPO).

        Returns
        -------
        quimb.tensor.MatrixProductOperator
            MPO representation of the Hamiltonian.

        """
        return self.to_builder().build_mpo()

    def get_U_exact(self, t, data_reg, controls):
        """
        Construct the exact time-evolution operator as a quantum gate.

        Computes

        .. math::
            U(t) = e^{-i H t}

        using dense matrix exponentiation.

        Parameters
        ----------
        t : float
            Evolution time.
        data_reg : sequence of int
            Qubit register on which the Hamiltonian acts.
        controls : sequence of int or None
            Control qubits for the gate.

        Returns
        -------
        quimb.tensor.Gate
            Exact multi-qubit unitary gate.

        """
        if len(data_reg) != self.n_qubits:
            raise ValueError("Invalid data_reg size")
        h_dense = self.to_dense()
        U = qu.expm(-1j * h_dense * t)
        return qtn.Gate.from_raw(U, qubits=data_reg, controls=controls)

    def get_trotter_step(self, dt, data_reg, trotter_order):
        """
        Construct a Trotterized time-evolution circuit (one step).

        Parameters
        ----------
        dt : float
            Time step.
        data_reg : sequence of int
            Qubit register.
        trotter_order : int
            Trotter order (1 or 2).

        Returns
        -------
        list
            List of abstract gate instructions implementing the Trotter step.

        Raises
        ------
        ValueError
            If the Trotter order is not implemented.

        """
        if len(data_reg) != self.n_qubits:
            raise ValueError("Invalid data_reg size")
        if trotter_order == 1:
            program = []
            for term in self.terms:
                program += rotation_gates(term, delta=dt, qubit_reg=data_reg)
            return program
        if trotter_order == 2:
            program = []
            for term in self.terms:
                program += rotation_gates(term, delta=dt / 2, qubit_reg=data_reg)
            for term in reversed(self.terms):
                program += rotation_gates(term, delta=dt / 2, qubit_reg=data_reg)
            return program
        raise ValueError(f"order {trotter_order} not implemented")


def rotation_gates(term, delta, qubit_reg):
    """
    Generate a gate sequence for exponentiating a Pauli-string term.

    Implements

    .. math::
        e^{-i \\delta \\theta P}

    where ``P`` is a tensor product of Pauli operators, using basis
    rotations, CNOT chains, and a single ``RZ`` rotation.

    Parameters
    ----------
    term : tuple
        Hamiltonian term ``(theta, pauli_string, qubits)``.
    delta : float
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
        if op in ("x", "X"):
            routine.append(("H", qubit_reg[qubit]))
        if op in ("y", "Y"):
            routine.append(("RX", np.pi / 2, qubit_reg[qubit]))

    # CNOTs
    for j in range(len(pauli_string) - 1):
        routine.append(("CNOT", qubit_reg[qubits[j]], qubit_reg[qubits[j + 1]]))

    # RZ gate
    routine.append(
        ("RZ", 2 * theta * delta, qubit_reg[qubits[-1]])
    )  ## RZ(alpha) = exp(-1j * alpha/2 * sigma_z)

    # CNOTs back
    for j in range(len(pauli_string) - 1, 0, -1):
        routine.append(("CNOT", qubit_reg[qubits[j - 1]], qubit_reg[qubits[j]]))

    # Rotations back
    for op, qubit in zip(pauli_string, qubits, strict=True):
        if op in ("x", "X"):
            routine.append(("H", qubit_reg[qubit]))

        if op in ("y", "Y"):
            routine.append(("RX", -np.pi / 2, qubit_reg[qubit]))

    return routine
