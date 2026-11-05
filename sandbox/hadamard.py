import numpy as np
import quimb.tensor as qtn

from qpe_toolbox.tensor import apply_gate_from_mpo, controlled_mpo, kron_mpos, kron_mps


def build_hadamard_test_circuit(
    mps_init, unitary, order, theta, *, max_bond=0, cutoff=1e-12
):
    """
    Build circuit for Hadamard test on unitary.
    """
    n_qubits = mps_init.L
    psi_init = kron_mps(qtn.MPS_computational_state("0"), mps_init)

    circ = qtn.CircuitMPS(n_qubits + 1, psi0=psi_init, max_bond=max_bond, cutoff=cutoff)

    data_reg = list(range(1, n_qubits + 1))
    circ.apply_gate("H", 0)

    cU = kron_mpos(qtn.MatrixProductOperator([np.eye(2)]), unitary)
    cU = controlled_mpo(cU, phys_reg=data_reg, aux_reg=[0], k_ctrl=0)

    for _ in range(2**order):
        circ = apply_gate_from_mpo(circ, cU, compress=True, max_bond=max_bond)

    circ.apply_gate("PHASE", theta, 0)
    circ.apply_gate("H", 0)
    return circ


def run_hadamard_test(
    mps_init,
    unitary,
    order,
    theta,
    n_shots,
    *,
    rng=None,
    max_bond=0,
    cutoff=1e-12,
):
    circ = build_hadamard_test_circuit(
        mps_init, unitary, order, theta, max_bond=max_bond, cutoff=cutoff
    )
    aux_ind = 0

    if n_shots == 0:
        probs = circ.compute_marginal(where=[aux_ind])
        return probs[0] - probs[1]

    popcount = 0
    for bitstring in circ.sample(n_shots, seed=rng):
        popcount += int(bitstring[aux_ind])
    return (n_shots - 2 * popcount) / n_shots
