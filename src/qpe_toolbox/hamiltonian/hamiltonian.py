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


def heisenberg_hamiltonian(n_qubits, *, coupling_strength=1.0):
    """
    Construct a 1D nearest-neighbor spin 1/2 Heisenberg Hamiltonian with open boundaries.

    The Hamiltonian is given by

    .. math::
        H = \\sum_{i=0}^{N-2} \\frac{J}{4}
        (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})

    where only spin-1/2 systems are supported. The normalization is chosen such that
    ``heisenberg_hamiltonian(2)`` is the standard :math:`\\mathbf{S}\\cdot\\mathbf{S}` operator
    with eigenvalues ``(-3/4, 1/4, 1/4, 1/4)``.

    Parameters
    ----------
    n_qubits : int
        Number of spins (qubits) in the chain.
    coupling_strength : float, optional
        Exchange coupling constant :math:`J`. Default is 1.0.

    Returns
    -------
    Hamiltonian
        Heisenberg Hamiltonian represented as a qubit operator.
    """
    terms = []
    for i in range(n_qubits - 1):
        for op in ["xx", "yy", "zz"]:
            # convention: S^α = σ^α / 2
            # use Pauli matrices as terms and set coefficient to J/4
            terms.append((coupling_strength / 4, op, [i, i + 1]))
    return Hamiltonian(terms, n_qubits)


def do_dmrg(hamiltonian):
    """
    Perform a DMRG ground-state calculation using quimb.

    Based on quimb :quimb-api:`DMRG2`

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian to be converted into an MPO.

    Returns
    -------
    E0 : float
        Ground-state energy.
    psi0 : :quimb-api:`MatrixProductState`
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
    terms : sequence of tuple
        Hamiltonian terms in the form
        ``(coefficient, pauli_string, qubits)``, e.g. ``(0.5, "xy", [0, 1])``.
    n_qubits : int
        Total number of qubits.
    """

    def __init__(self, terms, n_qubits):
        self._terms = list(terms)
        self._n_qubits = int(n_qubits)

    @property
    def terms(self):
        return self._terms

    @property
    def n_terms(self):
        return len(self._terms)

    @property
    def n_qubits(self):
        return self._n_qubits

    @property
    def shape(self):
        return (2**self._n_qubits, 2**self._n_qubits)

    def __repr__(self):
        return f"Hamiltonian(n_qubits={self._n_qubits}, n_terms={self.n_terms})"

    def __str__(self):
        lines = [
            f"Hamiltonian(n_qubits={self._n_qubits}, n_terms={self.n_terms}) with terms:"
        ]
        for coeff, paulis, qubits in self._terms:
            lines.append(f"  {coeff:+6g} {paulis.upper()} @ {qubits}")
        return "\n".join(lines)

    def to_dense(self):
        """
        Convert the Hamiltonian to a dense matrix representation.

        Returns
        -------
        :quimb:`quimb.qarray <autoapi/quimb/index.html#quimb.qarray>`
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
        :quimb:`quimb.operator.SparseOperatorBuilder <autoapi/quimb/operator/index.html#quimb.operator.SparseOperatorBuilder>`
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
        :quimb-api:`MatrixProductOperator`
            MPO representation of the Hamiltonian.
        """
        return self.to_builder().build_mpo()

    def get_U_exact(self, evolution_time, data_reg, controls):
        """
        Construct the exact time-evolution operator as a quantum gate.

        Computes

        .. math::
            U(t) = e^{-i H t}

        using dense matrix exponentiation.

        Parameters
        ----------
        evolution_time : float
            Evolution time.
        data_reg : sequence of int
            Qubit register on which the Hamiltonian acts.
        controls : sequence of int or None
            Control qubits for the gate.

        Returns
        -------
        :quimb-api:`Gate`
            Exact multi-qubit unitary gate.
        """
        if len(data_reg) != self.n_qubits:
            raise ValueError("Invalid data_reg size")
        h_dense = self.to_dense()
        U = qu.expm(-1j * h_dense * evolution_time)
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
                program += rotation_gates(term, dt, data_reg)
            return program
        if trotter_order == 2:
            program = []
            for term in self.terms:
                program += rotation_gates(term, dt / 2, data_reg)
            for term in reversed(self.terms):
                program += rotation_gates(term, dt / 2, data_reg)
            return program
        raise ValueError(f"order {trotter_order} not implemented")


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
