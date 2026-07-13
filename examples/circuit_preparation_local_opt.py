# ---
# jupyter:
#   jupytext:
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
import tqdm
from quimb.tensor import DMRG2
from quimb.tensor.contraction import contract_strategy

from qpe_toolbox.circuit import ansatz_circuit_su4
from qpe_toolbox.hamiltonian import Hamiltonian

# %% [markdown]
# ## Tensor Network Fitting Algorithm
#
# The core of the optimization is the function `_tn_fit_core` and the main wrapper `tn_fit`. Given a tensor network `tn` that we want to fit to a target tensor network `tn_target`, the algorithm proceeds as follows:
#
# - We tag each tensor in `tn` that is to be optimized with a unique tag (e.g., `"__VARi__"`). By default, all tensors are optimized.
# - We form the overlap tensor network `tnAB = tn_fit * conj(tn_target)`, where `tn_fit` is a copy of `tn` with the variable tags added. The contraction of this network gives the overlap `<tn_fit | tn_target>`.
# - For each variable tensor, we compute the **environment** tensor `b` by contracting all other tensors of `tnAB` except the variable tensor. This environment is the partial derivative of the overlap with respect to that tensor.
# - The optimal update for the variable tensor (maximizing the overlap, given the rest) is given by the **polar decomposition** of the environment: we reshape `b` to a matrix (with left and right indices combined), perform an SVD `b = U S V^dagger`, and set the new tensor to `U V^dagger` (the unitary part). This is the orthogonal projection that maximizes the overlap.
# - We iterate over all variable tensors (one sweep) and repeat for a number of steps or until convergence (change in overlap below tolerance).
#
# The wrapper `tn_fit` handles tagging, combining with the target (conjugated), and calling the core routine.

# %%
def _tn_fit_core(
    var_tags,
    tnAB,
    tol,
    steps,
    *,
    progbar=False,
):
    """Core optimization loop for tensor network fitting."""
    xp = tnAB.get_namespace()

    # Precompute environment contractions for each variable tensor.
    env_contractions = []
    for tg in var_tags:
        # The variable tensor is identified by the tag "__KET__" and the variable tag.
        tb = tnAB["__KET__", tg]
        lix = tb.left_inds
        rix = tuple(x for x in tb.inds if x not in tb.left_inds)
        # The rest of the network (all tensors except this one) forms the environment.
        b_tn = tnAB.select((tg,), "!all")
        env_contractions.append((tb, b_tn, lix, rix))

    if tol != 0.0 or progbar:
        old_d = float("inf")

    if progbar:
        pbar = tqdm.trange(steps)
    else:
        pbar = range(steps)

    for _ in pbar:
        # Sweep over all variable tensors.
        for tb, b_tn, lix, rix in env_contractions:
            # Contract the environment to a dense matrix.
            b = b_tn.to_dense(rix, lix)
            # SVD of the environment.
            u, _, v = xp.linalg.svd(b)
            # Optimal update: U @ V^dagger.
            x = u @ v
            x_r = xp.reshape(x, tb.shape)
            tb.modify(data=xp.conj(x_r))

        # Check convergence.
        if (tol != 0.0) or progbar:
            dagx = autoray.dag(x)
            d = float(0)
            if x.ndim == 2:
                d = xp.trace(xp.real(dagx @ b))
            else:
                d = xp.real(dagx @ b)

            if abs(d - old_d) < tol:
                break
            old_d = d

        if progbar:
            pbar.set_description(f"{d:.4g}")


def tn_fit(
    tn,
    tn_target,
    tags=None,
    steps=100,
    tol=1e-8,
    contract_optimize="auto-hq",
    *,
    progbar=False,
    **kwargs,
):
    """
    Fit tensor network `tn` to target tensor network `tn_target`.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to be optimized (in-place).
    tn_target : TensorNetwork
        The target tensor network (usually a state we want to approximate).
    tags : str or list of str, optional
        Tags selecting which tensors of `tn` to optimize. If None, all tensors are optimized.
    steps : int
        Number of sweeps.
    tol : float
        Convergence tolerance on the change of the overlap.
    contract_optimize : str
        Contraction strategy for the environments.
    progbar : bool
        Whether to show a progress bar.
    """
    tn_fit = tn.copy()
    tn_fit.add_tag("__KET__")

    # Tag the tensors to be optimized.
    if tags is None:
        to_tag = tn_fit.tensors
    else:
        to_tag = tn_fit.select_tensors(tags, "any")

    var_tags = []
    for i, t in enumerate(to_tag):
        var_tag = f"__VAR{i}__"
        t.add_tag(var_tag)
        var_tags.append(var_tag)

    # Form the overlap network: <tn_fit | tn_target>
    tn_target_conj = tn_target.conj(mangle_inner=True)
    tnAB = tn_fit.combine(tn_target_conj, virtual=True, check_collisions=False)

    with contract_strategy(contract_optimize):
        _tn_fit_core(
            var_tags=var_tags,
            tnAB=tnAB,
            tol=tol,
            steps=steps,
            progbar=progbar,
            **kwargs,
        )

    # Copy optimized data back to the original tensor network.
    for t1, t2 in zip(tn, tn_fit, strict=True):
        t2.transpose_like_(t1)
        t1.modify(data=t2.data)

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
