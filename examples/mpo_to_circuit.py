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

import copy

from quimb.tensor import DMRG2, MPS_rand_state

from qpe_toolbox.circuit.mpo_circuit_transpilation import (
    build_first_sweep,
    find_transfer_structure,
    init_cost_tn,
    optimize_single_gate_update,
    state_preparation_mpo,
    trotter_approx_as_MPO,
)
from qpe_toolbox.hamiltonian import Hamiltonian

# %% [markdown]
# ## Dynamics induced by the next-nearest-neighbor Ising model

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
    order=4,
    dt=0.5,
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

# %%
dict_transf = find_transfer_structure(n_qubits=L, cost_tn=cost_tn)

# %%
dict_contr_envs = build_first_sweep(
    n_qubits=L, cost_tn=cost_tn, dict_transf=dict_transf, drop_tags=True
)

# %% [markdown]
# Once the environment structure is found, we need to find the optimal gate to update a particular two-qubit unitary. To do so, we contract all the tensors around it. For example, say we want to update the first unitary at depth 0 between qubits 0 and 1, $U^{(0)}_{0,1}$; we contract all the tensors around it into its environment $E^{(0)}_{0,1}$ (this $E$ environment contains the $L$ and $R$ of each site, together with the corresponding tensor from the reference unitary at that site).

# %% [markdown]
# <img src="./figures/MPO_to_circuit_transpilation/transpil_sgu1.svg" align="center">

# %% [markdown]
# The environment $E$ is not necessarily a unitary object. To re-introduce unitarity, we can compute the SVD decomposition of it and remove the singular values. This substitution $E^{(0)}_{0,1} \longrightarrow U^{\dagger (0) \mathrm{opt}}_{0,1}=u v^\dagger$ is indeed the exact analytical solution of the *Unconstrained Linear* problem:

# %% [markdown]
# <img src="./figures/MPO_to_circuit_transpilation/transpil_sgu2.svg" align="center">

# %%
rtol = 1e-6
n_sweeps_max = 100

cp_cost_tn = cost_tn.copy(deep=True)
cp_dict_transf = copy.deepcopy(dict_transf)
cp_dict_contr_envs = copy.deepcopy(dict_contr_envs)

opt_cost_tn, opt_dict_contr_envs = optimize_single_gate_update(
    n_qubits=L,
    cost_tn=cp_cost_tn,
    rtol=rtol,
    n_sweeps_max=n_sweeps_max,
    dict_transf=cp_dict_transf,
    dict_contr_envs=cp_dict_contr_envs,
)

# retrieve the optimal circuit tensor network
opt_circuit_tn = opt_cost_tn.copy(deep=True)
opt_circuit_tn.delete(tags=("MPO"))

# %% [markdown]
# *Causer et al.* find that the model is prone to get stuck on local minima, even when starting from different initial circuits by changing the seed of the Ansatz:

# %%
list_seeds = [1, 2, 3]
for seed in list_seeds:
    cost_tn = init_cost_tn(
        ref_mpo=trotter_mpo_ham_NNIM,
        depth=depth,
        param_scaling=1e-1,
        closed=True,
        seed=seed,
    )

    dict_transf = find_transfer_structure(n_qubits=L, cost_tn=cost_tn)

    dict_contr_envs = build_first_sweep(
        n_qubits=L, cost_tn=cost_tn, dict_transf=dict_transf, drop_tags=True
    )

    opt_cost_tn, opt_dict_contr_envs = optimize_single_gate_update(
        n_qubits=L,
        cost_tn=cost_tn,
        rtol=rtol,
        n_sweeps_max=n_sweeps_max,
        dict_transf=dict_transf,
        dict_contr_envs=dict_contr_envs,
    )

# %% [markdown]
# The way *Causer et al.* overcome this issue is by designing a circuit Ansatz that looks like the second-order Trotter expansion of the circuit, where some SWAPs are held fixed and only the remaining gates need to be optimized.


# %% [markdown]
# ## State Preparation

# %% [markdown]
# With the same cost function we can target also the problem of finding a good preparation circuit for some initial state in the form of an MPS. We just need to build $\text{M}_{\mathrm{ref}}$ to represent the transition from an empty quantum register into the target MPS $|0 \rangle^{\otimes L} \langle \Psi_{\mathrm{ref}} |$.
#
# Note that in this case, the cost function will unfold into a square tensor network with open boundary conditions, since the outer product used to build the reference MPO is just an encoding of the tensors.
#
# <img src="./figures/MPO_to_circuit_transpilation/transpil_state_prep.svg" align="center">

# %%
# We pick a target state: the ground state of the next-nearest-neighbor Ising Hamiltonian defined above, in MPS form.
# `DMRG2` itself takes no seed; we seed reproducibility through its `p0` starting guess instead.
ham_NNIM_mpo = ham_NNIM.to_mpo()
p0 = MPS_rand_state(L, bond_dim=2, seed=42)
dmrg = DMRG2(ham_NNIM_mpo, p0=p0)
dmrg.solve(max_sweeps=16, bond_dims=64, verbosity=1, cutoffs=1e-12)
GS = dmrg.state

# We build the transition MPO from the empty register to the target MPS
GS_mpo = state_preparation_mpo(state_mps=GS)

# We feed the new reference MPO into the same routine as above
n_sweeps_max = 1000
rtol = 1e-6
list_depths = [1, 2, 3, 4]
for depth in list_depths:
    cost_tn_closed = init_cost_tn(
        ref_mpo=GS_mpo,
        depth=depth,
        param_scaling=1e-1,
        closed=True,
        seed=37,
    )

    dict_transf = find_transfer_structure(n_qubits=L, cost_tn=cost_tn_closed)

    dict_contr_envs = build_first_sweep(
        n_qubits=L, cost_tn=cost_tn_closed, dict_transf=dict_transf, drop_tags=True
    )
    print(f"Best overlap for depth {depth}:")
    opt_cost_tn, opt_dict_contr_envs = optimize_single_gate_update(
        n_qubits=L,
        cost_tn=cost_tn_closed,
        rtol=rtol,
        n_sweeps_max=n_sweeps_max,
        dict_transf=dict_transf,
        dict_contr_envs=dict_contr_envs,
    )
