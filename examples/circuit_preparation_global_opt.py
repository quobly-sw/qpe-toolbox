# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Circuit Preparation via Global Optimization
#
# We show how to approximate the ground state of a transverse-field Ising (TFI) Hamiltonian
# using a parametrized quantum circuit. Three different optimization strategies are compared:
#
# 1. **Standard L-BFGS** on a fixed-depth circuit.
# 2. **Basin-hopping** - a global optimization method that helps escape local minima.
# 3. **Sequential layer-by-layer optimization**, where the parameters of a shallower circuit
#    are used to initialize a deeper one.
#
# The TFI model is defined on a 1D chain with open boundaries. The variational ansatz is a
# brick-wall circuit of SU(4) gates (each acting on two neighbouring qubits). All parameters are optimized simultaneously using automatic
# differentiation via JAX.

# %%
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JAX_ENABLE_X64"] = "True"

import autoray
import numpy as np
from quimb.tensor import DMRG2, TNOptimizer

# Local imports from qpe_toolbox
from qpe_toolbox.circuit import ansatz_circuit_su4
from qpe_toolbox.hamiltonian import Hamiltonian

# %% [markdown]
# ## Hamiltonian: 1D Transverse-Field Ising (TFI) model
#
# We consider a chain of $n$ spins with open boundaries. The Hamiltonian reads
#
# $$
#
# H = g_x \sum_{i} X_i + g_{zz} \sum_{i} Z_i Z_{i+1},
#
# $$
#
# with $g_x = -1.1$ and $g_{zz} = -1.0$. This model is non-integrable and features
# a quantum phase transition at $|g_x| = |g_{zz}|$; we work in the ferromagnetic regime.

# %%
def loss_circ(circ, mpo):
    """
    Loss function: expectation value of the MPO Hamiltonian with respect to
    the state produced by the circuit.

    The contraction is performed with automatic differentiation via JAX.
    """
    psi = circ.psi
    psiH = psi.H
    norm_tn = psiH & psi
    psi.align_(mpo, psiH)
    energy_tn = psiH & mpo & psi
    energy = autoray.do("real", energy_tn.contract())
    norm = autoray.do("real", norm_tn.contract())
    return (energy / norm).real


def make_circuit_optimizer(circ, mpo):
    """
    Factory to create a TNOptimizer for a circuit with the given MPO.

    The optimizer uses L-BFGS-B (or basin-hopping) and JAX for gradients.
    """
    return TNOptimizer(
        circ,
        loss_circ,
        loss_constants={"mpo": mpo},
        autodiff_backend="jax",
        optimizer="L-BFGS-B",
        progbar=False,
    )


# %% [markdown]
# ## DMRG reference ground state
#
# We first obtain a high-quality reference using DMRG with bond dimension 64.
# This serves as the exact (within numerical precision) ground state to compare
# against.

# %%
# --- Hamiltonian definition ---
n_qubits = 8
gx, gzz = -1.1, -1.0
terms = []
for x in range(n_qubits):
    terms.append((gx, "x", [x]))  # transverse field
for x in range(n_qubits - 1):
    terms.append((gzz, "zz", [x, x + 1]))  # nearest-neighbour coupling
ham = Hamiltonian(terms, n_qubits)
mpo = ham.to_mpo()

# --- DMRG ---
dmrg = DMRG2(mpo)
dmrg.solve(max_sweeps=16, tol=1e-8, bond_dims=64, verbosity=0)
GS = dmrg.state
dmrg_energy = np.real(dmrg.energy)
print("*** DMRG reference energy:", dmrg_energy)
print()

# %% [markdown]
# ## 1. Direct L-BFGS optimization
#
# We use the `ansatz_circuit_su4` circuit, which consists of layers of SU(4)
# two-qubit gates arranged in a brick-wall pattern. Each SU(4) gate is parameterized
# by 15 real numbers.
#
# The L-BFGS optimizer runs for up to 10 000 iterations.

# %%
# --- L-BFGS ---
depth = 6
circ = ansatz_circuit_su4(n_qubits, depth)
circ_opt = make_circuit_optimizer(circ, mpo)
optimal_circ = circ_opt.optimize(n=10000, tol=1e-8)
ovlp = (dmrg.state.H & optimal_circ.psi).contract()
print(
    f" # parameters = {circ_opt.d: 6d}",
    f" Energy = {circ_opt.loss: >12.8f}",
    f" Error = {np.abs(1 - circ_opt.loss / dmrg_energy): >10.3e}",
    f" 1-F = {1 - np.abs(ovlp) ** 2: >10.3e}",
)
print()

# %% [markdown]
# ## 2. Basin-hopping optimization
#
# Basin-hopping is a global optimisation algorithm that performs random steps
# in parameter space, followed by a local minimisation. It can help to escape
# poor local minima. Here we use 5000 iterations with 10 hops per step and a
# temperature of 0.1 (controlling the acceptance probability of uphill moves).

# %%
# --- Basin hopping ---
depth = 6
circ = ansatz_circuit_su4(n_qubits, depth)
circ_opt = make_circuit_optimizer(circ, mpo)
optimal_circ = circ_opt.optimize_basinhopping(n=5000, nhop=10, temperature=0.1)
ovlp = (dmrg.state.H & optimal_circ.psi).contract()
print(
    f" # parameters = {circ_opt.d: 4d}",
    f" Energy = {circ_opt.loss: >12.8f}",
    f" Error = {np.abs(1 - circ_opt.loss / dmrg_energy): >10.3e}",
    f" 1-F = {1 - np.abs(ovlp) ** 2: >10.3e}",
)
print()

# %% [markdown]
# ## 3. Sequential layer-wise optimization
#
# Starting from a depth-1 circuit, we optimize it, then add one layer at a time,
# using the previously found parameters as initial guess for the deeper circuit.
# This can lead to faster convergence because the optimisation landscape for
# deeper circuits is easier to navigate when starting close to a good shallow
# solution.
#
# %%
# --- Sequential optimization ---
circ = ansatz_circuit_su4(n_qubits, 1)
circ_opt = make_circuit_optimizer(circ, mpo)
optimal_circ = circ_opt.optimize(n=10000, tol=1e-8)
ovlp = (dmrg.state.H & optimal_circ.psi).contract()
print(
    f" # parameters = {circ_opt.d: 4d}",
    f" Energy = {circ_opt.loss: >12.8f}",
    f" Error = {np.abs(1 - circ_opt.loss / dmrg_energy): >10.3e}",
    f" 1-F = {1 - np.abs(ovlp) ** 2: >10.3e}",
)

for ii in range(2, depth + 1):
    circ = ansatz_circuit_su4(n_qubits, ii, param_scaling=1e-1)
    circ.set_params(optimal_circ.get_params())
    circ_opt = make_circuit_optimizer(circ, mpo)
    optimal_circ = circ_opt.optimize(n=10000, tol=1e-8)
    ovlp = (dmrg.state.H & optimal_circ.psi).contract()
    print(
        f" # parameters = {circ_opt.d: 4d}",
        f" Energy = {circ_opt.loss: >12.8f}",
        f" Error = {np.abs(1 - circ_opt.loss / dmrg_energy): >10.3e}",
        f" 1-F = {1 - np.abs(ovlp) ** 2: >10.3e}",
    )
print()

# %% [markdown]
# ## Results and discussion
#
# The table above reports the number of parameters, the achieved energy,
# the relative error with respect to the DMRG energy, and the infidelity
# $1 - F$ (where $F = |\langle \psi_{\mathrm{DMRG}} | \psi_{\mathrm{circ}} \rangle|^2$).
#
# The trial state prepared by the circuit can then be used as input for
# Quantum Phase Estimation (QPE) to refine the energy estimate, as shown
# in the `textbook_qpe` tutorial.
