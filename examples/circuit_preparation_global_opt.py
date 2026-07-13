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
# # Variational Circuit Preparation for the Transverse-Field Ising Model
#
# We show how to approximate the ground state of a transverse-field Ising (TFI) Hamiltonian,
# expressed as a Matrix-Product-Operator (MPO), using a parametrized quantum circuit.
# Three different optimization strategies are compared:
#
# 1. **Standard L-BFGS** on a fixed-depth circuit.
# 2. **Basin-hopping** to escape local minima.
# 3. **Sequential layer-by-layer optimization**, where the parameters learned for a shallower
#    circuit are used to initialize a deeper one.
#
# This notebook reproduces some of the ideas from R. Haghshenas et al.,
# [*Variational Power of Quantum Circuit Tensor Networks*](https://link.aps.org/doi/10.1103/PhysRevX.12.011047) (2022),
# and follows the coding style of the `qpe_toolbox` examples.

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
# We consider a chain of $n=16$ spins with open boundaries. The Hamiltonian reads
#
# $$
# H = g_x \sum_{i} X_i + g_{zz} \sum_{i} Z_i Z_{i+1},
#
# $$
#
# with $g_x = -1.1$ and $g_{zz} = -1.0$. This model is non-integrable and has a
# quantum phase transition at $|g_x| = |g_{zz}|$. We work in the ferromagnetic regime.

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
# two-qubit gates interleaved with single-qubit rotations. For a depth-8 circuit
# (16 qubits) we have roughly $16 \times 15 \times 2 = 480$ parameters.
# The optimizer runs for up to 10000 iterations.

# %%
# --- L-BFGS ---
depth = 6
circ = ansatz_circuit_su4(n_qubits, depth)
circ_opt = make_circuit_optimizer(circ, mpo)
optimal_circ = circ_opt.optimize(n=10000, tol=1e-8)
ovlp = (dmrg.state.H & optimal_circ.psi).contract()
print(
    f" # = {circ_opt.d: 5d}",
    f" Energy = {circ_opt.loss:.8f}",
    f" Error = {np.abs(1 - circ_opt.loss / dmrg_energy):.3e}",
    f" 1-F = {1 - np.abs(ovlp) ** 2:.3e}",
)
print()

# %% [markdown]
# ## 2. Basin-hopping optimization
#
# Basin-hopping is a global optimisation algorithm that performs random steps
# in parameter space, followed by a local minimisation. It can help to escape
# poor local minima. Here we use 1000 iterations with 10 hops per step.

# %%
# --- Basin hopping ---
depth = 6
circ = ansatz_circuit_su4(n_qubits, depth)
circ_opt = make_circuit_optimizer(circ, mpo)
optimal_circ = circ_opt.optimize_basinhopping(n=1000, nhop=20, temperature=0.1)
ovlp = (dmrg.state.H & optimal_circ.psi).contract()
print(
    f" # = {circ_opt.d: 5d}",
    f" Energy = {circ_opt.loss:.8f}",
    f" Error = {np.abs(1 - circ_opt.loss / dmrg_energy):.3e}",
    f" 1-F = {1 - np.abs(ovlp) ** 2:.3e}",
)
print()

# %% [markdown]
# ## 3. Sequential layer-wise optimization
#
# Starting from depth-1, we optimize the circuit, then add one layer at a time,
# using the previously found parameters as initial guess for the deeper circuit.
# This often leads to better convergence because the optimisation landscape
# for deeper circuits is easier to navigate when starting close to a good
# shallow solution.

# %%
# --- Sequential optimization ---
circ = ansatz_circuit_su4(n_qubits, 1)
circ_opt = make_circuit_optimizer(circ, mpo)
optimal_circ = circ_opt.optimize(n=10000, tol=1e-8)
ovlp = (dmrg.state.H & optimal_circ.psi).contract()
print(
    f" # = {circ_opt.d: 5d}",
    f" Energy = {circ_opt.loss:.8f}",
    f" Error = {np.abs(1 - circ_opt.loss / dmrg_energy):.3e}",
    f" 1-F = {1 - np.abs(ovlp) ** 2:.3e}",
)

for ii in range(2, depth + 1):
    circ = ansatz_circuit_su4(n_qubits, ii, param_scaling=1e-1)
    circ.set_params(optimal_circ.get_params())
    circ_opt = make_circuit_optimizer(circ, mpo)
    optimal_circ = circ_opt.optimize(n=10000, tol=1e-8)
    ovlp = (dmrg.state.H & optimal_circ.psi).contract()
    print(
        f" # = {circ_opt.d: 5d}",
        f" Energy = {circ_opt.loss:.8f}",
        f" Error = {np.abs(1 - circ_opt.loss / dmrg_energy):.3e}",
        f" 1-F = {1 - np.abs(ovlp) ** 2:.3e}",
    )
print()

# %% [markdown]
# ## Results and discussion
#
# The table above reports the number of parameters, the achieved energy,
# the relative error, and the infidelity $1 - F$ (where $F$ is the overlap
# squared with the DMRG ground state). Typically, the sequential strategy
# yields the best compromise between circuit depth and fidelity.
#
# The trial state prepared by the circuit can then be used as input for
# Quantum Phase Estimation (QPE) to refine the energy estimate, as shown
# in the `textbook_qpe` and `variational_circuit_preparation` examples.
