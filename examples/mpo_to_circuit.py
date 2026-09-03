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
# # MPO to circuit transpilation
#
# This notebook reproduces the algorithms of [Causer et al.](https://arxiv.org/abs/2312.14245), which itself builds on the method of [Vidal](https://arxiv.org/abs/0707.1454v1) (note that we reference the first version because it is substantially different from the later versions and contains the appropriate information to follow the procedure).
#
# Other teams built on top of it, adding variants and state preparation, in [Kanno et al., 2024](https://www.pnas.org/doi/abs/10.1073/pnas.2425026122) and [Kanno et al., 2026](https://arxiv.org/abs/2601.15616).
#
# ---

# %% [markdown]
# The goal of this notebook is to illustrate the transpilation of an MPO unitary operator ($\mathrm{M}_{\mathrm{ref}}$ representing the unitary time evolution induced by a Hamiltonian during a time $\Delta t$) into a nearest-neighbor brickwall circuit that can be run on a QPU ($U_{\mathrm{bw}}$):

# %% [markdown]
# <img src="./figures/MPO_to_circuit_transpilation/transpil_Uref.svg" align="center">

# %% [markdown]
# We impose an Ansatz circuit made out of three rows of entangling gates on even-odd-even links. Note that in some references the prescription for increasing the depth/layer counting is just a row of even or of odd entangling gates, while other references consider that two consecutive rows even-odd constitute a layer

# %% [markdown]
# <img src="./figures/MPO_to_circuit_transpilation/transpil_Ubw.svg" align="center">

# %% [markdown]
# In order to translate the circuit that best reproduces the action of the unitary MPO, we maximize the overlap between the two unitaries. If this cost function is maximized, then $U_{\mathrm{bw}}$ will act on another state or operator in the same way as $\mathrm{M}_{\mathrm{ref}}$ does.
#
# In this case, the cost function can be computed as a fully contracted tensor network with a cylindrical topology. An illustration of such a cost function for the previously introduced ansatz circuit is:

# %% [markdown]
# <img src="./figures/MPO_to_circuit_transpilation/transpil_cost.svg" align="center">

# %%
import os

os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
from quimb.tensor import DMRG2, MPS_rand_state

from qpe_toolbox.circuit import init_cost_tn, transpile_mpo_to_circuit
from qpe_toolbox.hamiltonian import Hamiltonian, trotter_approx_as_MPO
from qpe_toolbox.tensor import state_preparation_mpo

# %%
plt.rcParams.update({"font.size": 12})

# %% [markdown]
# ## Dynamics Induced by the Next-Nearest-Neighbor Ising Model

# %%
L = 11
gx, gzz, gz1z = 0.2, 0.5, 0.1
terms_NNIM = []
for x in range(L):
    terms_NNIM.append((gx, "x", [x]))
for x in range(L - 1):
    terms_NNIM.append((gzz, "zz", [x, x + 1]))
for x in range(L - 2):
    terms_NNIM.append((gz1z, "zz", [x, x + 2]))

ham_NNIM = Hamiltonian(terms_NNIM, L)

# %%
trotter_mpo_ham_NNIM = trotter_approx_as_MPO(
    ham_NNIM,
    0.5,
    trotter_order=4,
    cutoff=1e-12,
    max_bond=64,
    verbosity=1,
)

# %% [markdown]
# We now initialize the cost function.

# %%
depth = 3

for boundary_bool in [False, True]:
    cost_tn = init_cost_tn(
        ref_mpo=trotter_mpo_ham_NNIM,
        depth=depth,
        param_scaling=1e-1,
        closed=boundary_bool,
        rng=np.random.default_rng(42),
    )
    cost_tn.draw([f"ROUND_{i}" for i in range(depth)], show_inds=True, show_tags=False)

# %% [markdown]
# Now we optimize the overlap as suggested in the context of [algorithms for entanglement renormalization](https://arxiv.org/abs/0707.1454v1): to solve the *Constrained Linear* problem where the isometry to be found decomposes in a given circuit structure, we solve a series of *Unconstrained Linear* problems for each of the gates forming the circuit. The latter is known to have an analytical solution.
#
# The procedure goes as follows: first, we select a given qubit (say, site 0) and target the optimization of all the gates exposed to that qubit. In the following picture we highlight in yellow the set of gates we are referring to:

# %% [markdown]
# <img src="./figures/MPO_to_circuit_transpilation/transpil_opt_pos0.svg" align="center">

# %% [markdown]
# Second, we contract the subnetwork that is not being updated. That contraction is the right environment of qubit 0, $R_0$; such an environment can be built recursively, starting from the right environment of qubit 1, $R_1$, and so on until the last qubit:

# %% [markdown]
# <img src="./figures/MPO_to_circuit_transpilation/transpil_build_Rs_1st_sweep.svg" align="center">

# %% [markdown]
# By knowing the first right environment (that of the second-to-last qubit, $R_{\mathrm{n_qubits-1}}$, which coincides with the last tensor of $\mathrm{M}_{\mathrm{ref}}$ ) and the tensors required to transfer from one environment to another, we can build all the right environments for the first sweep. Similarly, this can be done for left environments:

# %% [markdown]
# Once the environment structure is found, we need to find the optimal gate to update a particular two-qubit unitary. To do so, we contract all the tensors around it. For example, say we want to update the first unitary at depth 0 between qubits 0 and 1, $U^{(0)}_{0,1}$; we contract all the tensors around it into its environment $E^{(0)}_{0,1}$ (this $E$ environment contains the $L$ and $R$ of each site, together with the corresponding tensor from the reference unitary at that site).

# %% [markdown]
# <img src="./figures/MPO_to_circuit_transpilation/transpil_sgu1.svg" align="center">

# %% [markdown]
# The environment $E$ is not necessarily a unitary object. To re-introduce unitarity, we can compute the SVD decomposition of it and remove the singular values. This substitution $E^{(0)}_{0,1} \longrightarrow U^{\dagger (0) \mathrm{opt}}_{0,1}=u v^\dagger$ is indeed the exact analytical solution of the *Unconstrained Linear* problem:

# %% [markdown]
# <img src="./figures/MPO_to_circuit_transpilation/transpil_sgu2.svg" align="center">

# %% [markdown]
# Note that the single-gate update above is the same update
# used by the generic `tn_fit` routine in the [`circuit_preparation_opt`](./circuit_preparation_opt.ipynb)
# tutorial's Local Optimization section. `tn_fit` handles an
# arbitrary tensor-network topology at the cost of fully re-contracting each tensor's environment from scratch on
# every sweep. On this other hand, this notebook is specialized for a 1D chain and caches the left/right partial
# contractions (`contracted_envs`), updating only the one environment adjacent to each optimized gate.
#
# *Causer et al.* find that the model is prone to get stuck on local minima, even when starting from different initial circuits. We will take care of this this by running the same optimization with different seeds of the Ansatz:

# %%
rtol = 1e-6
n_sweeps_max = 250
n_seeds = 4
depth = 5
overlaps5 = np.empty(n_seeds)

for seed in range(n_seeds):
    cost_tn, contracted_envs, overlaps5[seed] = transpile_mpo_to_circuit(
        trotter_mpo_ham_NNIM,
        depth,
        rtol,
        n_sweeps_max,
        param_scaling=1e-1,
        closed=True,
        rng=np.random.default_rng(seed),
    )
    print(f"seed {seed}: overlap = {overlaps5[seed]:.6f}")

# %% [markdown]
# Indeed here we observe that for one seed, the optimization gets stuck at a small overlap `~0.7`. *Causer et al.* overcome the local minimum issue by designing a circuit Ansatz that looks like the second-order Trotter expansion of the circuit, where some SWAPs are held fixed and only the remaining gates need to be optimized.
#
# Let us now consider different dephts:


# %%
depths = np.arange(1, 6)
overlaps = np.zeros((5, n_seeds))
for i in range(4):
    print(f"depth = {depths[i]}")
    for seed in range(n_seeds):
        cost_tn, contracted_envs, overlaps[i, seed] = transpile_mpo_to_circuit(
            trotter_mpo_ham_NNIM,
            depths[i],
            rtol,
            n_sweeps_max,
            param_scaling=1e-1,
            closed=True,
            rng=np.random.default_rng(seed),
        )

print("depths = 5")
print(f"overlaps = {overlaps5}")
overlaps[4] = overlaps5

# %%
fig, ax = plt.subplots()
ax.plot(depths, 1 - overlaps, ls="", color="k", marker="o", ms=4)
ax.plot(depths, 1 - overlaps.max(axis=1), "rv-", ms=6)
ax.set_ylim(0, 0.012)  # cannot display overlap=0.7 on scale
ax.grid(visible=True, alpha=0.3)
ax.set_xlabel("depth")
ax.set_ylabel("1 - overlap")
ax.set_title(f"MPO transpilation L = {L}");

# %% [markdown]
# ## State Preparation

# %% [markdown]
# With the same cost function we can target also the problem of finding a good preparation circuit for some initial state in the form of an MPS. We just need to build $\text{M}_{\mathrm{ref}}$ to represent the transition from an empty quantum register into the target MPS $|0 \rangle^{\otimes L} \langle \Psi_{\mathrm{ref}} |$.
#
# Note that in this case, the cost function will unfold into a square tensor network with open boundary conditions, since the outer product used to build the reference MPO is just an encoding of the tensors.
#
# <img src="./figures/MPO_to_circuit_transpilation/transpil_state_prep.svg" align="center">

# %%
# Let us pick a target state: the ground state of the next-nearest-neighbor Ising Hamiltonian defined above, in MPS form.
ham_NNIM_mpo = ham_NNIM.to_mpo()
p0 = MPS_rand_state(L, bond_dim=2, seed=42)
dmrg = DMRG2(ham_NNIM_mpo, p0=p0)
dmrg.solve(max_sweeps=16, bond_dims=64, verbosity=1, cutoffs=1e-12)
GS = dmrg.state

# build an MPO by taking the outer producht with an empty register
GS_mpo = state_preparation_mpo(state_mps=GS)

# %%
# optimize the reference MPO using the same routine as above
n_sweeps_max = 100
rtol = 1e-6
state_depths = np.arange(1, 6)
n_seeds = 4
state_overlaps = np.empty((state_depths.size, n_seeds))
for i, depth in enumerate(state_depths):
    print(f"{depth =}")
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        cost_tn, contracted_envs, state_overlaps[i, seed] = transpile_mpo_to_circuit(
            GS_mpo, depth, rtol, n_sweeps_max, param_scaling=1e-1, closed=True, rng=rng
        )

# %%
fig, ax = plt.subplots()
ax.plot(state_depths, 1 - state_overlaps, ls="", color="k", marker="o", ms=4)
ax.plot(state_depths, 1 - state_overlaps.max(axis=1), "rv-", ms=6)
ax.set_xlabel("depth")
ax.set_ylabel("1 - overlap")
ax.grid(visible=True, alpha=0.3)
ax.set_title(f"State Preparation L = {L}");
