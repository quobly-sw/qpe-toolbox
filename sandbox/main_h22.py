import autoray
import numpy as np
import quimb.tensor as qtn
from config import RPEConfig
from estimation import run_rmpe_lcu
from lcu import LCU
from pyscf_driver import PySCFDriver, get_interaction_operator, get_quimb_hamiltonian
from quimb.tensor import DMRG2

from qpe_toolbox.hamiltonian import Hamiltonian


def get_orbital_number_operator(i, n_qubits):
    """Single-orbital number operator n_i."""
    terms = [
        (-0.5, "z", [i]),
        (0.5, "I" * n_qubits, list(range(n_qubits))),
    ]
    return Hamiltonian(terms, n_qubits)


def get_number_operator(n_qubits):
    """Total particle-number operator N = sum_i a_i^† a_i in quimb format."""
    terms = [(-0.5, "z", [i]) for i in range(n_qubits)]
    terms.append((n_qubits / 2.0, "I" * n_qubits, list(range(n_qubits))))
    return Hamiltonian(terms, n_qubits)


def get_number_operator_squared(n_qubits):
    """
    Total particle-number-squared operator N^2 = (sum_i a_i^† a_i)^2
    in quimb Hamiltonian format, using the same JW convention as
    get_quimb_hamiltonian (Z strings, e_const stored separately).
    """
    terms = []

    # - (n_qubits / 2) * sum_i Z_i
    for i in range(n_qubits):
        terms.append((-0.5 * n_qubits, "z", [i]))

    # + (1/2) * sum_{i<j} Z_i Z_j
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            terms.append((0.5, "zz", [i, j]))

    terms.append(
        (n_qubits * (n_qubits + 1) / 4.0, "I" * n_qubits, list(range(n_qubits)))
    )
    return Hamiltonian(terms, n_qubits)


def get_number_minus_target_squared_terms(n_qubits, n_target):
    """
    Terms for (N - n_target)^2 in quimb SparseOperatorBuilder format:
        (coeff, pauli_string, qubit_indices)
    with the scalar piece carried by an explicit identity term.
    """
    terms = []

    # Linear: (n - n_q/2) * sum_i Z_i
    c1 = n_target - 0.5 * n_qubits
    if abs(c1) > 1e-15:
        for i in range(n_qubits):
            terms.append((c1, "z", [i]))

    # Two-body: (1/2) * sum_{i<j} Z_i Z_j
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            terms.append((0.5, "zz", [i, j]))

    # Constant
    c0 = n_qubits * (n_qubits + 1) / 4.0 - n_target * n_qubits + n_target**2
    terms.append((c0, "I" * n_qubits, list(range(n_qubits))))
    return Hamiltonian(terms, n_qubits)


def eval_N(mps, n_qubits):
    mpsH = mps.H
    mpoN = get_number_operator(n_qubits).to_mpo()
    mps.align_(mpoN, mpsH)
    tn = mpsH & mpoN & mps
    return autoray.do("real", tn.contract(all))


def eval_E(mps, mpoE):
    mpsH = mps.H
    mps.align_(mpoE, mpsH)
    tn = mpsH & mpoE & mps
    return autoray.do("real", tn.contract(all))


def main():
    geometry = [("H", (0, 0, 0)), ("H", (0, 0, 0.7414))]
    basis = "sto-3g"

    driver = PySCFDriver(geometry, basis, charge=0, multiplicity=1, verbose=1)
    # print("Computing natural orbitals from CCSD...")
    # noons_ccsd = driver.transform_to_natural_orbitals(method='ccsd')
    # print("CCSD natural orbital occupation numbers:", np.ravel(noons_ccsd))
    driver.apply_symmetry_shift(method="L-BFGS-B", tol=1e-10)

    molecule = driver.get_molecular_data()
    interaction_op = get_interaction_operator(
        molecule, n_active_electrons=None, n_active_orbitals=None
    )

    n_qubits = molecule.n_qubits
    print(n_qubits)

    mat_H = get_quimb_hamiltonian(interaction_op, n_qubits, cutoff=1e-6)
    mpo_H = mat_H.to_mpo()
    mpo_C = (
        mpo_H.copy() + 10 * get_number_minus_target_squared_terms(n_qubits, 2).to_mpo()
    )

    dmrg0 = DMRG2(mpo_C)
    dmrg0.solve(max_sweeps=16, tol=1e-8, bond_dims=100, verbosity=0)
    mps_0 = dmrg0.state
    e0 = eval_E(mps_0, mpo_H)  # + np.real(mat_H.e_const)

    print("DMRG0 energy (including nuclear repulsion):", e0 + np.real(mat_H.e_const))
    print("DMRG0 energy:", e0)
    print("Number of e = ", eval_N(mps_0, n_qubits))
    print()

    mpo_0 = mps_0.partial_trace_to_mpo(keep=list(range(n_qubits)))
    dmrg1 = DMRG2(mpo_C + 10 * mpo_0)
    dmrg1.solve(max_sweeps=16, tol=1e-8, bond_dims=100, verbosity=0)
    mps_1 = dmrg1.state
    e1 = eval_E(mps_1, mpo_H)  # + np.real(mat_H.e_const)
    print("DMRG1 energy (including nuclear repulsion):", e1 + np.real(mat_H.e_const))
    print("DMRG1 energy:", e1)
    print("Number of e = ", eval_N(mps_1, n_qubits))
    print()

    # Create LCU instance
    lcu = LCU.from_hamiltonian(mat_H)
    print(
        "norm1 =",
        lcu.norm1,
        "nbit_aux =",
        lcu.nbit_aux,
        "nbit_sys =",
        lcu.nbit_sys,
        "n_term =",
        lcu.n_term,
    )

    # RPE
    mps_0 = 1 / np.sqrt(2) * mps_0 + 1 / np.sqrt(2) * mps_1
    mps_0 = mps_0 + 0.1 * qtn.MPS_rand_state(n_qubits, 1)
    mps_0 = mps_0 / mps_0.norm()
    ovlp = (dmrg0.state.H & mps_0).contract()
    print("Overlap = ", ovlp)

    rpe_config = RPEConfig(n_repetitions=5, n_shots=0, verbosity=1)
    theta_rpe = run_rmpe_lcu(lcu, rpe_config, mps_0, k_state=2)
    energy_rpe = lcu.norm1 * np.cos(theta_rpe)  # [-1,:])
    energy_rpe = np.sort(energy_rpe, axis=1)
    print(energy_rpe)


if __name__ == "__main__":
    main()
