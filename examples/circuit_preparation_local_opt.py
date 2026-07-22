# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Circuit Preparation via Local Optimization
#
# In this notebook, we demonstrate a classical pre-processing technique to prepare a quantum circuit that approximates the ground state of a given Hamiltonian. This circuit can then be used as an initial state in quantum phase estimation (QPE) or other quantum algorithms. The method performs a **local optimization** of the unitaries in the circuit to maximize the overlap with a target state (here, obtained from DMRG).
#
# We consider the 1D transverse-field Ising model. The ansatz is a quantum circuit built from SU(4) gates (two-qubit gates) arranged in a brick-wall pattern.#
# In the follwoing, we will:
# 1. Define the Hamiltonian and compute its ground state using DMRG.
# 2. Initialize a random SU(4) circuit with a given depth.
# 3. Perform the **local** optimization to fit the circuit to the target state.
# 4. Evaluate the energy and fidelity of the optimized circuit.
# 5. Demonstrate a **sequential depth optimization**: starting from depth 1, we optimize, then increase the depth and re-optimize, reusing the parameters from the shallower circuit. This often yields better convergence.

# %%
import os

os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import autoray
import numpy as np
import quimb as qu
from quimb.tensor import DMRG2

from qpe_toolbox.circuit import ansatz_circuit_su4, tn_fit
from qpe_toolbox.hamiltonian import Hamiltonian

# %% [markdown]
# ## Main Example: Transverse-Field Ising Chain
#
# We will use the 1D Heisenberg (or transverse-field Ising) model as a test case. The Hamiltonian is:
#
# $$ H = \sum_{i} g_x X_i + \sum_{i} g_{zz} Z_i Z_{i+1} $$
#
# with $g_x = -1.1$ and $g_{zz} = -1.0$. We take $n=16$ sites.
#
# First, we compute the ground state using DMRG (via `quimb.tensor.DMRG2`). This gives us the target state `GS` and the exact energy.

# %%
# --- Hamiltonian ---
n_qubits = 8
list_paulis = ["I", "X", "Y", "Z"]
gx, gzz = -1.1, -1.0
terms = []
for x in range(n_qubits):
    terms.append((gx, "x", [x]))
for x in range(n_qubits - 1):
    terms.append((gzz, "zz", [x, x + 1]))
ham = Hamiltonian(terms, n_qubits)
mpo = ham.to_mpo()

dmrg = DMRG2(mpo)
dmrg.solve(max_sweeps=16, tol=1e-8, bond_dims=64, verbosity=0)
GS = dmrg.state
dmrg_energy = np.real(dmrg.energy)
print(f"*** DMRG reference energy: {dmrg_energy:12.10f}")
print()

# %% [markdown]
# ## Optimization of a Depth-6 Circuit
#
# We initialize a random SU(4) brick-wall circuit of depth 6. The `ansatz_circuit_su4` function from `qpe_toolbox.circuit` generates a circuit with random parameters (here scaled by 0.01). We then compute the energy and fidelity (overlap squared) with the DMRG state.
#
# Next, we call `tn_fit` to optimize the tensor network representing the circuit. Note that we do not restrict the tensors to be unitary; the optimization may produce non-unitary tensors. However, in practice the optimized tensors often remain close to unitary if the target state is reachable with the given ansatz structure.
#
# We perform 100,000 sweeps (or until convergence) with a tolerance of 1e-8.

# %%
# Optimization: depth 6
depth = 6
circ = qu.tensor.Circuit(n_qubits)
circ = ansatz_circuit_su4(
    n_qubits=n_qubits, depth=depth, param_scaling=0.01, parametrize=False
)

tn = circ.psi
tn_fit(tn, GS, tags="SU4", steps=100000, tol=1e-8)

ovlp = (dmrg.state.H & tn).contract()
tnH = tn.H
tn.align_(mpo, tnH)
energy_tn = tnH & mpo & tn
ene = autoray.do("real", energy_tn.contract(all))

print(
    f"Depth = {depth:2d}   Energy = {ene:12.8f}   Error = {np.abs(1 - ene / dmrg_energy):10.3e}   1-F = {1 - np.abs(ovlp) ** 2:10.3e}"
)
print()

# %% [markdown]
# ## Sequential Depth Optimization
#
# A more robust strategy is to start with a shallow circuit (depth 1) and progressively increase the depth, each time using the optimized parameters from the previous depth as the initial guess. This helps to avoid local minima and often leads to better convergence, especially for larger systems.
#
# We demonstrate this by starting with depth 1, then depth 2, and so on up to `depth=6`. At each step, we optimize for 10,000 sweeps (or until tolerance). The parameters (tensor entries) are copied from the previous optimized circuit to the new, deeper circuit.

# %%
# Sequential optimization
circ = ansatz_circuit_su4(
    n_qubits=n_qubits, depth=1, param_scaling=0.01, parametrize=False
)

tn = circ.psi
tn_fit(tn, GS, tags="SU4", steps=10000, tol=1e-8)

ovlp = (dmrg.state.H & tn).contract()
tnH = tn.H
tn.align_(mpo, tnH)
energy_tn = tnH & mpo & tn
ene = autoray.do("real", energy_tn.contract(all))

print(
    f"Depth = {1:2d}   Energy = {ene:12.8f}   Error = {np.abs(1 - ene / dmrg_energy):10.3e}   1-F = {1 - np.abs(ovlp) ** 2:10.3e}"
)

for ii in range(2, depth + 1):
    circ = ansatz_circuit_su4(n_qubits, ii, param_scaling=1.0, parametrize=False)

    tn_old = tn
    tn = circ.psi
    tn.set_params(tn_old.get_params())  # reuse parameters from previous depth

    tn_fit(tn, GS, tags="SU4", steps=10000, tol=1e-8)

    ovlp = (dmrg.state.H & tn).contract()
    tnH = tn.H
    tn.align_(mpo, tnH)
    energy_tn = tnH & mpo & tn
    ene = autoray.do("real", energy_tn.contract(all))

    print(
        f"Depth = {ii:2d}   Energy = {ene:12.8f}   Error = {np.abs(1 - ene / dmrg_energy):10.3e}   1-F = {1 - np.abs(ovlp) ** 2:10.3e}"
    )
print()
