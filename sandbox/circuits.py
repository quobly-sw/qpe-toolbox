import numpy as np
import quimb.tensor as qtn
from layout import RegisterLayout

from qpe_toolbox.tensor import (
    apply_gate_from_mpo,
    controlled_mpo,
    kron_mpos,
    kron_mps,
)


def build_lcu_qpe_circuit(lcu, mps_init, nbit_qpe, *, max_bond=0, cutoff=1e-12):
    """
    Build the QPE circuit with controlled walk operators, up to IQFT.
    """
    regs = RegisterLayout(
        nbit_qpe=nbit_qpe,
        nbit_aux=lcu.nbit_aux,
        nbit_sys=lcu.nbit_sys,
    )

    mpoR = lcu.reflection_mpo
    mpoR = kron_mpos(qtn.MPO_identity(nbit_qpe), mpoR)
    mpoS = lcu.select_mpo
    mpoS = kron_mpos(qtn.MPO_identity(nbit_qpe), mpoS)

    mpsV = lcu.prepare_mps
    mps0 = qtn.MPS_computational_state("0" * nbit_qpe)
    circ = qtn.CircuitMPS(
        regs.total_qubits,
        psi0=kron_mps(mps0, kron_mps(mpsV, mps_init)),
        max_bond=max_bond,
        cutoff=cutoff,
    )

    # Hadamard wall
    for k in regs.qpe:
        circ.apply_gate("H", k)

    def apply_mpo(c, mpo):
        psi = mpo.apply(c.psi, compress=True, max_bond=max_bond, cutoff=cutoff)
        return qtn.CircuitMPS(psi0=psi, max_bond=max_bond, cutoff=cutoff)

    # Sequence of controlled-W
    for k in regs.qpe:
        if k == 0:
            circ.apply_gate("X", 0)
            c_mpoR = controlled_mpo(
                mpoR, phys_reg=regs.aux_plus_system, aux_reg=regs.qpe, k_ctrl=0
            )
            circ = apply_mpo(circ, c_mpoR)
            c_mpoS = controlled_mpo(
                mpoS, phys_reg=regs.aux_plus_system, aux_reg=regs.qpe, k_ctrl=0
            )
            circ = apply_mpo(circ, c_mpoS)
            circ.apply_gate("X", 0)
        else:
            c_mpoR = controlled_mpo(
                mpoR, phys_reg=regs.aux_plus_system, aux_reg=regs.qpe, k_ctrl=k
            )
            circ = apply_mpo(circ, c_mpoR)

            for _ in range(2 ** (k - 1)):
                circ = apply_mpo(circ, mpoR)
                circ = apply_mpo(circ, mpoS)

            circ = apply_mpo(circ, c_mpoR)
    return circ, regs


def build_lcu_hadamard_test_circuit(
    lcu, mps_init, order, theta, *, max_bond=0, cutoff=1e-12
):
    """
    Build circuit for Hadamard test on unitary.
    """
    nbit_qpe = 1
    regs = RegisterLayout(
        nbit_qpe=nbit_qpe,
        nbit_aux=lcu.nbit_aux,
        nbit_sys=lcu.nbit_sys,
    )

    psi0 = qtn.MPS_computational_state("0" * nbit_qpe)
    psi0 = kron_mps(psi0, mps_init)

    circ = qtn.CircuitMPS(
        regs.total_qubits, psi0=psi0, max_bond=max_bond, cutoff=cutoff
    )

    for k in regs.qpe:
        circ.apply_gate("H", k)

    mpoR = lcu.reflection_mpo
    mpoR = kron_mpos(qtn.MatrixProductOperator([np.eye(2)]), mpoR)
    mpoS = lcu.select_mpo
    mpoS = kron_mpos(qtn.MatrixProductOperator([np.eye(2)]), mpoS)

    def apply_mpo(c, mpo):
        psi = mpo.apply(c.psi, compress=True, max_bond=max_bond, cutoff=cutoff)
        return qtn.CircuitMPS(psi0=psi, max_bond=max_bond, cutoff=cutoff)

    if order == 1:
        circ.apply_gate("X", 0)
        c_mpoR = controlled_mpo(
            mpoR, phys_reg=regs.aux_plus_system, aux_reg=regs.qpe, k_ctrl=0
        )
        circ = apply_mpo(circ, c_mpoR)
        c_mpoS = controlled_mpo(
            mpoS, phys_reg=regs.aux_plus_system, aux_reg=regs.qpe, k_ctrl=0
        )
        circ = apply_mpo(circ, c_mpoS)
        circ.apply_gate("X", 0)

    elif order % 2 == 0:
        c_mpoR = controlled_mpo(
            mpoR, phys_reg=regs.aux_plus_system, aux_reg=regs.qpe, k_ctrl=0
        )
        circ = apply_mpo(circ, c_mpoR)
        for _ in range(order // 2):
            circ = apply_mpo(circ, mpoR)
            circ = apply_mpo(circ, mpoS)
        circ = apply_mpo(circ, c_mpoR)

    else:
        circ.apply_gate("X", 0)
        c_mpoR = controlled_mpo(
            mpoR, phys_reg=regs.aux_plus_system, aux_reg=regs.qpe, k_ctrl=0
        )
        circ = apply_mpo(circ, c_mpoR)
        c_mpoS = controlled_mpo(
            mpoS, phys_reg=regs.aux_plus_system, aux_reg=regs.qpe, k_ctrl=0
        )
        circ = apply_mpo(circ, c_mpoS)
        circ.apply_gate("X", 0)

        c_mpoR = controlled_mpo(
            mpoR, phys_reg=regs.aux_plus_system, aux_reg=regs.qpe, k_ctrl=0
        )
        circ = apply_mpo(circ, c_mpoR)
        for _ in range(order // 2):
            circ = apply_mpo(circ, mpoR)
            circ = apply_mpo(circ, mpoS)
        circ = apply_mpo(circ, c_mpoR)

    circ.apply_gate("PHASE", theta, 0)
    circ.apply_gate("H", 0)
    return circ


def build_hadamard_test_circuit(
    lcu, mps_init, order, theta, *, max_bond=0, cutoff=1e-12
):
    """
    Build circuit for Hadamard test on unitary.
    """
    n_qubits = mps_init.L
    psi_init = kron_mps(qtn.MPS_computational_state("0"), mps_init)

    circ = qtn.CircuitMPS(n_qubits + 1, psi0=psi_init, max_bond=max_bond, cutoff=cutoff)

    data_reg = list(range(1, n_qubits + 1))
    circ.apply_gate("H", 0)

    cU = kron_mpos(qtn.MatrixProductOperator([np.eye(2)]), lcu.walk_mpo)
    cU = controlled_mpo(cU, phys_reg=data_reg, aux_reg=[0], k_ctrl=0)

    for _ in range(order):
        circ = apply_gate_from_mpo(circ, cU, compress=True, max_bond=max_bond)

    circ.apply_gate("PHASE", theta, 0)
    circ.apply_gate("H", 0)
    return circ
