# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

import itertools
import time

import numpy as np
import quimb as qu
import quimb.tensor as qtn

from qpe_toolbox.circuit import shift_control_gates
from qpe_toolbox.tensor import apply_gate_from_mpo, controlled_mpo, kron_mpos, kron_mps

from .qft import iqft_swapped

#####################################
### L register and PREPARE oracle ###
#####################################


def get_lcu_weights(hamiltonian):
    """
    Compute the weights for the linear combination of unitaries (LCU) representation
    of a Hamiltonian, including normalization factor and ancilla register size.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian from the QPE-Toolbox ``Hamiltonian`` class.

    Returns
    -------
    weights : list of float
        Absolute values of the Hamiltonian coefficients, extended with zeros to
        match a power-of-two length.
    lmb : float
        Sum of absolute values of Hamiltonian coefficients (normalization factor).
    L : int
        Number of original Hamiltonian terms.
    m_L : int
        Number of qubits required for the auxiliary L-register,
        i.e., ceil(log2(L)).

    """
    weights = [abs(P[0]) for P in hamiltonian.terms]

    lmb = sum(weights)
    L = len(weights)
    if L < 2:
        raise ValueError("Need at least 2 terms in Hamiltonian")

    # number of ancilla qubits for the PREPARE oracle
    m_L = int(np.ceil(np.log2(L)))
    # complete the weights with zeros when L is not a power of 2
    weights.extend([0] * (2**m_L - L))
    return weights, lmb, L, m_L


def build_lcu_prepare_state_mps(hamiltonian, cutoff=1e-10):
    r"""
    Construct the normalized MPS representing the L register state :math:`\ket{\mathcal{L}}`

    .. math::
        \ket{\mathcal{L}} = sum_\ell sqrt(w_\ell / lambda) \ket{\ell}

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian from the QPE-Toolbox ``Hamiltonian`` class.
    cutoff : float, default ``1e-10``
        Singular value cutoff for MPS compression.

    Returns
    -------
    L_mps : quimb.tensor.MPS
        MPS representing the state :math:`\ket{\mathcal{L}}`.

    """
    weights, lmb, L, m_L = get_lcu_weights(hamiltonian)

    # Initialize |L> state MPS
    L_mps = np.sqrt(weights[0] / lmb) * qtn.MPS_computational_state("0" * m_L)
    for i in range(1, L):
        L_mps += np.sqrt(weights[i] / lmb) * qtn.MPS_computational_state(f"{i:0{m_L}b}")
    # Check normalization
    if not np.isclose(L_mps.norm(), 1.0, atol=cutoff):
        raise ValueError("Invalid MPS normalization")
    L_mps.compress(cutoff=cutoff)
    return L_mps


def build_lcu_prepare_mpo(hamiltonian, cutoff=1e-10):
    r"""
    Construct the PREPARE oracle MPO :math:`\ket{0}\bra{\mathcal{L}}`.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian from the QPE-Toolbox ``Hamiltonian`` class.
    cutoff : float, default ``1e-10``
        Cutoff for MPO compression.

    Returns
    -------
    prep_mpo : quimb.tensor.MatrixProductOperator
        MPO implementing the PREPARE oracle.

    """
    weights, lmb, _, m_L = get_lcu_weights(hamiltonian)

    zero_mps = qtn.MPS_computational_state("0" * m_L)
    zero_mpo = zero_mps.partial_trace_to_mpo(keep=list(range(m_L)))
    prep_mpo = np.sqrt(weights[0] / lmb) * zero_mpo

    for k, w in enumerate(weights[1:]):
        if w > 0:
            prep_mpo = prep_mpo + np.sqrt(w / lmb) * _prepare_computational_state(
                k + 1, m_L
            )
            prep_mpo.compress(cutoff=cutoff)
        elif w < 0:
            raise ValueError("negative weight")
    return prep_mpo


def _prepare_computational_state(k, n):
    """Get the MPO for :math:`\\ketbra{k}{0}` where :math:`0 \\leq k < 2^n`"""
    if not (0 <= k < 2**n):
        raise ValueError("k must be between 0 and 2**n")
    bitstring = f"{k:0{n}b}"
    arrays = []
    for idx, ik in enumerate(bitstring):
        if idx == 0 or idx == n - 1:
            aux = np.zeros([1, 2, 2], dtype=complex)  # internal, upper, lower
            aux[0, int(ik), 0] = 1
        elif idx < n - 1:
            aux = np.zeros(
                [1, 1, 2, 2], dtype=complex
            )  # left internal, right internal, upper, lower
            aux[0, 0, int(ik), 0] = 1
        arrays.append(aux)

    return qtn.MatrixProductOperator(arrays, shape="lrud")


###############################################################################
##################### SELECT oracle ###########################################
###############################################################################


# gates instruction implementation of SELECT
def lcu_select_gates(hamiltonian):
    """
    Construct the full list of gate instructions for the SELECT oracle.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object from the QPE-Toolbox ``Hamiltonian`` class

    Returns
    -------
    gates : list of quimb.tensor.circuit.Gate
        Gate sequence implementing the SELECT oracle.

    """
    L = len(hamiltonian.terms)
    return list(
        itertools.chain.from_iterable(_gates_llxHl(hamiltonian, i) for i in range(L))
    )


def _gates_llxHl(hamiltonian, l_term):
    r"""
    Construct gate instructions for controlled :math:`H_\ell`
    (conditioned on L-register).

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object from the QPE-Toolbox ``Hamiltonian`` class.
    l_term : int
        Index of Hamiltonian term.

    Returns
    -------
    gates : list of quimb.tensor.circuit.Gate
        Gates implementing controlled :math:`H_\ell`.

    """
    L = len(hamiltonian.terms)
    if not (0 <= l_term < L):
        raise ValueError("Invalid Hamiltonian term index")

    m_L = int(np.ceil(np.log2(L)))

    phys_reg = list(range(m_L, m_L + hamiltonian.n_qubits))
    l_reg = list(range(m_L))

    gates = []
    bitstring = f"{{0:0{m_L}b}}".format(l_term)
    for i, b in enumerate(bitstring):
        if int(b) == 0:
            gates.append(qtn.Gate("x", params=[], qubits=[l_reg[i]]))

    H_l = hamiltonian.terms[l_term][1]
    Hl_qubits = [phys_reg[qubit_idx] for qubit_idx in hamiltonian.terms[l_term][2]]
    prefactor = hamiltonian.terms[l_term][0]

    if prefactor < 0:
        gates.append(
            qtn.Gate("phase", params=[np.pi], qubits=[l_reg[0]], controls=l_reg[1:])
        )
    for i, g in enumerate(H_l):
        gates.append(qtn.Gate(g, params=[], qubits=[Hl_qubits[i]], controls=l_reg))

    for i, b in enumerate(bitstring):
        if int(b) == 0:
            gates.append(qtn.Gate("x", params=[], qubits=[l_reg[i]]))

    return gates


# MPO representation of SELECT oracle
def _build_Hl_mpo(hamiltonian, l_term):
    r"""
    Build MPO for the l-th Pauli string of a Hamiltonian

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian describing the system.
    l_term : int
        Index of the Hamiltonian term.

    Returns
    -------
    Hl_mpo : quimb.tensor.MatrixProductOperator
        MPO representing the l-th term of the Hamiltonian.

    """
    if l_term >= len(hamiltonian.terms):
        return qtn.MPO_identity(hamiltonian.n_qubits)
    if hamiltonian.terms[l_term][0] == 0:
        return qtn.MPO_identity(hamiltonian.n_qubits)

    P = hamiltonian.terms[l_term]
    prefactor = P[0]
    paulis = P[1]
    qubits = P[2]

    arrays = []
    for i in range(hamiltonian.n_qubits):
        if i in qubits:
            ind_i = qubits.index(i)
            mat = np.sign(prefactor) * qu.pauli(paulis[ind_i])
        else:
            mat = qu.identity(2)
        if (i == 0) or (i == (hamiltonian.n_qubits - 1)):
            aux = np.zeros([1, 2, 2], dtype=complex)
            aux[0, :, :] = mat
        else:
            aux = np.zeros([1, 1, 2, 2], dtype=complex)
            aux[0, 0, :, :] = mat
        arrays.append(aux)

    return qtn.MatrixProductOperator(arrays, shape="lrud")


def _build_llxHl_mpo(hamiltonian, l_term):
    r"""
    Construct the MPO for :math:`\ket{\ell}\bra{\ell} \otimes H_\ell`,
    used in the SELECT oracle.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian describing the system.
    l_term : int
        Index of the Hamiltonian term.

    Returns
    -------
    llxHl_mpo : quimb.tensor.MatrixProductOperator
        MPO representing :math:`\ket{\ell}\bra{\ell} \otimes H_\ell`.

    """
    m_L = int(np.ceil(np.log2(len(hamiltonian.terms))))
    l_mps = qtn.MPS_computational_state(f"{{0:0{m_L}b}}".format(l_term))
    l_mpo = l_mps.partial_trace_to_mpo(keep=list(range(m_L)))

    Hl_mpo = _build_Hl_mpo(hamiltonian, l_term)

    return kron_mpos(l_mpo, Hl_mpo)


def build_lcu_select_mpo(hamiltonian, cutoff=1e-10):
    r"""
    Construct the MPO implementing the SELECT oracle.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian describing the system.
    cutoff : float, default ``1e-10``
        MPO compression cutoff.

    Returns
    -------
    select_mpo : quimb.tensor.MatrixProductOperator
        MPO implementing :math:`SELECT = \sum_\ell \ket{\ell}\bra{\ell} \otimes H_\ell`.

    """
    L = len(hamiltonian.terms)
    m_L = int(np.ceil(np.log2(L)))

    select_mpo = _build_llxHl_mpo(hamiltonian, 0)
    for l_term in range(1, 2**m_L):
        aux = _build_llxHl_mpo(hamiltonian, l_term)
        select_mpo = aux + select_mpo
        select_mpo.compress(cutoff=cutoff)

    return select_mpo


###############################################################################
############### WALK operator #################################################
###############################################################################


def build_lcu_reflection_mpo(hamiltonian, cutoff=1e-10):
    r"""
    Construct the reflection operator :math:`\mathcal{R}_L` for the L register.

    .. math::
        \mathcal{R}_L = 2 \ket{\mathcal{L}}\bra{\mathcal{L}}\otimes\mathbb{1} - \mathbb{1}

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object from the QPE-Toolbox ``Hamiltonian`` class.
    cutoff : float, default ``1e-10``
        MPO compression cutoff.

    Returns
    -------
    R_L : quimb.tensor.MatrixProductOperator
        MPO representing the reflection

    """
    L_mps = build_lcu_prepare_state_mps(hamiltonian)
    m_L = L_mps.L
    L_mpo = L_mps.partial_trace_to_mpo(keep=list(range(m_L)))
    L_mpo.compress(cutoff=cutoff)

    n_qb = hamiltonian.n_qubits
    R_L = 2 * kron_mpos(L_mpo, qtn.MPO_identity(n_qb)) - qtn.MPO_identity(m_L + n_qb)
    R_L.compress(cutoff=cutoff)

    return R_L


###############################################################################
############## QPE ############################################################
###############################################################################


def run_qpe_lcu_walk_operator(
    H, psi0_mps, m_ph, *, max_bond=0, cutoff=1e-10, verbosity=0
):
    """
    Perform LCU and quantum phase estimation (QPE) using the walk operator.

    Parameters
    ----------
    H : Hamiltonian
        Hamiltonian object from the QPE-Toolbox ``Hamiltonian`` class.
    psi0_mps : quimb.tensor.MPS
        Initial state of the physical register.
    m_ph : int
        Number of phase estimation qubits.
    max_bond : int, default ``0``
        Maximum MPS bond dimension.
    cutoff : float, default ``1e-10``
        Truncation cutoff for MPS compression.
    verbosity : int, default ``0``
        Verbosity level. If >= 1, print result summary. If >= 2, print
        detailed progress messages.

    Returns
    -------
    traces : dict
        Contains timing information: {'ctimes': [...]}.
    theta : float
        Estimated phase from the walk operator.

    """
    st = time.time()
    ctimes = []
    m_L = int(np.ceil(np.log2(len(H.terms))))
    regs = _get_registers_qpe_lcu(H.n_qubits, m_L, m_ph)

    _, circ = qpe_first_stage_walk(
        H, psi0_mps, m_ph, regs, max_bond=max_bond, cutoff=cutoff, verbosity=verbosity
    )

    circ.apply_gates(iqft_swapped(regs["phase"]))
    ctimes.append(time.time() - st)
    if verbosity >= 2:
        msg = f"Start sampling, bond dim={circ.psi.max_bond()}, {ctimes[-1]:.1f}s"
        print(msg, end="\r")
        len_prev_msg = len(msg)
    probs = circ.compute_marginal(where=regs["phase"])
    ctimes.append(time.time() - st)

    if verbosity >= 1:
        if verbosity >= 2:
            print(" " * len_prev_msg, end="\r")
            print(f"Done sampling {ctimes[-1]:.1f}s", end="\r")
        print("binary" + " " * 6 + "\t ket" + " " * 4 + "\t phase  \t prob")
        for state_int, prob in sorted(
            enumerate(np.ravel(probs)), key=lambda x: x[1], reverse=True
        )[:3]:
            print(
                f"{format(state_int, f'0{m_ph}b'):<12}",
                f"{'|' + str(state_int) + '>':<8}",
                f"{state_int / 2**m_ph:<6.4f}",
                f"{prob:<6.4f}",
                sep=" \t ",
            )

    max_prob_state_int = np.argmax(probs)
    theta = max_prob_state_int / 2**m_ph

    traces = {"ctimes": ctimes}

    return traces, theta


def qpe_first_stage_walk(
    H, psi0_mps, m_ph, regs, *, max_bond=0, cutoff=1e-10, verbosity=0
):
    """
    LCU and first stage of QPE using walk operator: apply Hadamard wall
    and controlled-W sequence.

    Parameters
    ----------
    H : Hamiltonian
        Hamiltonian from the QPE-Toolbox ``Hamiltonian`` class.
    psi0_mps : quimb.tensor.MPS
        Initial state of the physical register.
    m_ph : int
        Number of phase qubits.
    regs : dict
        Dictionary with registers: {'phase':..., 'L':..., 'phys':...}.
    max_bond : int, default ``0``
        Maximum MPS bond dimension.
    cutoff : float, default ``1e-10``
        MPO/MPS compression cutoff.
    verbosity : int, default ``0``
        Verbosity level. If >= 2, print progress messages.

    Returns
    -------
    traces : dict
        Contains timing information: {'ctimes': [...]}.
    circ : quimb.tensor.CircuitMPS
        Circuit representing the applied QPE first stage.

    """
    st = time.time()
    traces = {"ctimes": []}
    m_L = int(np.ceil(np.log2(len(H.terms))))

    Id_ph = (
        qtn.MatrixProductOperator([np.eye(2)]) if m_ph == 1 else qtn.MPO_identity(m_ph)
    )
    RL_mpo = kron_mpos(Id_ph, build_lcu_reflection_mpo(H))
    select_gates = lcu_select_gates(H)

    L_mps = build_lcu_prepare_state_mps(H)
    phase_zeros = qtn.MPS_computational_state("0" * m_ph)
    psi_init = kron_mps(phase_zeros, kron_mps(L_mps, psi0_mps))
    circ = qtn.CircuitMPS(
        m_ph + m_L + H.n_qubits, psi0=psi_init, max_bond=max_bond, cutoff=cutoff
    )

    # Hadamard wall
    for k in regs["phase"]:
        circ.apply_gate("H", k)
    # sequence of controlled-W
    for k in regs["phase"]:
        cRLk_mpo = controlled_mpo(
            RL_mpo, phys_reg=regs["L"] + regs["phys"], aux_reg=regs["phase"], k_ctrl=k
        )
        cRLk_mpo.compress(cutoff=cutoff)
        cSELECTk = shift_control_gates(select_gates, m_aux=m_ph, k_ctrl=k)
        for _ in range(2**k):
            # W
            ## SELECT
            for g in cSELECTk:
                circ.apply_gate(g)
            ## controlled-RL
            circ = apply_gate_from_mpo(circ, cRLk_mpo, compress=True, max_bond=max_bond)
        traces["ctimes"].append(time.time() - st)
        if verbosity >= 2:
            print(
                f"End k={k}, bond dim={circ.psi.max_bond()}, {time.time() - st:.1f}s",
                end="\r",
            )

    return traces, circ


def _get_registers_qpe_lcu(n_qubits, m_L, m_ph):
    """
    Return dictionary of qubit registers for the phase, L,
    and physical registers in LCU QPE.

    Parameters
    ----------
    n_qubits : int
        Number of physical qubits.
    m_L : int
        Number of L register qubits.
    m_ph : int
        Number of phase estimation qubits.

    Returns
    -------
    regs : dict
        Dictionary with keys 'phase', 'L', 'phys' giving tuples of qubit indices.

    """
    regs = {}
    regs["phase"] = tuple(range(m_ph))
    regs["L"] = tuple(range(m_ph, m_ph + m_L))
    regs["phys"] = tuple(range(m_ph + m_L, m_ph + m_L + n_qubits))
    return regs


def get_energy_from_lcu_walk_phase(theta, lmb):
    r"""
    Get the energy from the eigenphase of the LCU Walk operator.

    ..math:
        E = \lambda \cos(2 \pi \theta)

    Parameters
    ----------
    theta : float
        Eigenphase of the Walk operator.
    lmb : float
        One-norm of LCU weights.

    Returns
    -------
    energy : float
        Estimated energy.

    """
    return lmb * np.cos(2 * np.pi * theta)


def estimate_lcu_error(m_ph, E0, lmb):
    """
    Estimate the error bound in energy in LCU QPE from finite number of phase qubits.

    Parameters
    ----------
    m_ph : int
        Number of phase estimation qubits.
    E0 : float
        Ground state energy estimate.
    lmb : float
        LCU normalization factor.

    Returns
    -------
    delta_E : float
        Estimated upper bound on the energy error.

    """
    return lmb * np.sqrt(1 - (E0 / lmb) ** 2) * 2 * np.pi / 2**m_ph
