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
# based on [original paper](https://arxiv.org/abs/2312.14245) which at the same time uses the method from [Vidal](https://arxiv.org/abs/0707.1454v1) (note that we reference the first version because it is substantially different from the later versions)
# other teams built on top of it adding some variants and state preparation in [paper 1](https://www.pnas.org/doi/abs/10.1073/pnas.2425026122) and [paper 2](https://arxiv.org/abs/2601.15616)

# %% [markdown]
# The goal of this notebook is to illustrate the transpilation of an MPO unitary operator ($U_{\mathrm{ref}}$ representing the time evolution induced by some Hamiltonian during a time $\Delta t$) into a nearest-neighbor brickwall circuit that can be run on some QPU ($U_{\mathrm{bw}}$):

# %% [markdown]
# <img src="./figures/MPO_to_circuit_transpilation/transpil_Uref.svg" align="center">

# %% [markdown]
# We impose an Ansatz circuit made out of three rows of entangling gates on even-odd-even links. Note that in some references the prescription for increasing the depth/layer counting is just a row of even or of odd entangling gates, while other references consider than two consecutive rows even-odd constitute a layer

# %% [markdown]
# <img src="./figures/MPO_to_circuit_transpilation/transpil_Ubw.svg" align="center">

# %% [markdown]
# In order to translate the circuit that best reproduces the action of the unitary MPO, we maximize the overlap between the two unitaries. If this cost function is maximized, then $U_{\mathrm{bw}}$ will match $U^\dagger_{\mathrm{ref}}$. 
#
# In this case, the cost function is a fully contracted tensor network with a cylindrical topology. An illustration of such a cost function for the formerly introduced ansatz circuit is:

# %% [markdown]
# <img src="./figures/MPO_to_circuit_transpilation/transpil_cost.svg" align="center">

# %% [markdown]
# With the same cost function we can target also the problem of transpiling an initial state in the form of an MPO into a preparation circuit applied on the empty quantum register. To do this, one just needs to build $U_{\mathrm{ref}}$ such that each tensor is an outer product of the local tensor of the MPS encoding the state $\Psi_{\mathrm{ref}}$ and a qubit initialized at $|0\rangle$.
#
# Note that in this case, the the cost function will unfold into a simple square tensor network, since the outer product used to build the reference MPO is just an encoding of the tensors.

# %% [markdown]
# <img src="./figures/MPO_to_circuit_transpilation/transpil_state_prep.svg" align="center">

# %%
import os

os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import copy

from qpe_toolbox.circuit.mpo_circuit_transpilation import (
    build_first_sweep,
    find_transfer_structure,
    init_cost_tn,
    optimize_single_gate_update,
    trotter_approx_as_MPO,
)
from qpe_toolbox.hamiltonian import Hamiltonian

list_paulis = ["I", "X", "Y", "Z"]

# %% [markdown]
# **NEXT-NEAREST-NEIGHBOR ISING MODEL**

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

# %% [markdown]
# in the following function: do we keep the verbose? if we keep it, do we change it?

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
# in the following I initialize the cost function

# %%
depth = 5

cost_tn_open = init_cost_tn(
    unitary_mpo=trotter_mpo_ham_NNIM,
    depth=depth,
    param_scaling=1e-1,
    factorize=False,
    closed=False,
)
cost_tn_open.draw([f"ROUND_{i}" for i in range(depth)], show_inds=True, show_tags=False)

cost_tn_closed = init_cost_tn(
    unitary_mpo=trotter_mpo_ham_NNIM,
    depth=depth,
    param_scaling=1e-1,
    factorize=False,
    closed=True,
)
cost_tn_closed.draw(
    [f"ROUND_{i}" for i in range(depth)], show_inds=False, show_tags=False
)

# %% [markdown]
# Now we optimize the overlap as suggested in the context of [algorithms for entanglement renormalization](https://arxiv.org/abs/0707.1454v1): to solve the *Constrained Linear* problem where the isometry to be found decomposes in a given circuit structure, we solve a series of *Unconstrained Linear* problems for each of the gates forming the circuit. The later is known to have an analytical solution.
#
# The procedure goes as follows: we fix ourselves on some qubit position (say, site 0) and target the optimization of all the gates exposed to that qubit. In the following picture we highlight in yellow the set of gates we are referring to:

# %% [markdown]
# <img src="./figures/MPO_to_circuit_transpilation/transpil_opt_pos0.svg" align="center">

# %% [markdown]
# Now we need to contract all the subnetwork that is not being updated. That contraction is the right environment of qubit 0, $R_0$; such an environment can be built at the same time from the right environment of qubit 1, $R_1$, and so un until the last qubit:

# %% [markdown]
# <img src="./figures/MPO_to_circuit_transpilation/transpil_build_Rs_1st_sweep.svg" align="center">

# %% [markdown]
# By knowing the first right environment (that of the second-to-last qubit, $R_{\mathrm{n_qubits-1}}$, which coincides with the last tensor of $U_{\mathrm{ref}}$ ) and the tensors required to transfer from one environment to another, we can build all the right environments for the first sweep. Conversely, this can be done for left environments:

# %%
dict_transf = find_transfer_structure(n_qubits=L, cost_tn=cost_tn_closed)

# %%
dict_contr_envs = build_first_sweep(
    n_qubits=L, cost_tn=cost_tn_closed, dict_transf=dict_transf, drop_tags=True
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
# optimize for "depth" ("2*depth" in Causer et al. paper)

list_n_sweeps = [10, 20, 50, 100]

for n_sweeps in list_n_sweeps:
    cp_cost_tn_closed = cost_tn_closed.copy(deep=True)
    cp_dict_transf = copy.deepcopy(dict_transf)
    cp_dict_contr_envs = copy.deepcopy(dict_contr_envs)

    opt_cost_tn, opt_dict_contr_envs = optimize_single_gate_update(
        n_qubits=L,
        cost_tn=cp_cost_tn_closed,
        n_sweeps=n_sweeps,
        dict_transf=cp_dict_transf,
        dict_contr_envs=cp_dict_contr_envs,
    )

# %% [markdown]
# In *Causer et al.* they find that the model is prone to get stuck on local minima, even when starting from different initial circuits by changing the seed of the Ansatz:

# %%
n_sweeps = 50
depth = 2
list_seeds = [1, 2, 3, 4, 5]
for seed in list_seeds:
    cost_tn_closed = init_cost_tn(
        unitary_mpo=trotter_mpo_ham_NNIM,
        depth=depth,
        param_scaling=1e-1,
        factorize=False,
        closed=True,
        seed=seed,
    )

    dict_transf = find_transfer_structure(n_qubits=L, cost_tn=cost_tn_closed)

    dict_contr_envs = build_first_sweep(
        n_qubits=L, cost_tn=cost_tn_closed, dict_transf=dict_transf, drop_tags=True
    )

    opt_cost_tn, opt_dict_contr_envs = optimize_single_gate_update(
        n_qubits=L,
        cost_tn=cost_tn_closed,
        n_sweeps=n_sweeps,
        dict_transf=dict_transf,
        dict_contr_envs=dict_contr_envs,
    )

# %% [markdown]
# The way *Causer et al.* overcome this issue is by designing a circuit Ansatz that looks like the second order Trotter expansion of the circuit, where some SWAPs are fixed and only a subset of gates needs to be fixed.
#
# We do not want to use model-specific optimizations so far, so we try with *2 site DMRG* and *AD*.

# %% [markdown]
# **CLUSTER ISING MODEL**

# %%
# cluster Ising model
L = 11
g = -0.75
terms_CIM = [] * (3 * L - 3)
for x in range(L):
    terms_CIM.append((-((1 + g) ** 2), "x", [x]))
for x in range(L - 1):
    terms_CIM.append((-2 * (1 - g**2), "zz", [x, x + 1]))
for x in range(L - 2):
    terms_CIM.append(((g - 1) ** 2, "zxz", [x, x + 1, x + 2]))

ham_CIM = Hamiltonian(terms_CIM, L)


cutoff = 1e-12
max_bond = 128
T = 0.5
n_steps = 10

trotter_mpo_ham_CIM_step = trotter_approx_as_MPO(
    ham_CIM,
    order=4,
    dt=T / 10,
    cutoff=cutoff,
    max_bond=max_bond,
)

trotter_mpo_ham_CIM = trotter_mpo_ham_CIM_step.copy(deep=True)
for _ in range(n_steps - 1):
    trotter_mpo_ham_CIM = trotter_mpo_ham_CIM.apply(
        trotter_mpo_ham_CIM_step, compress=True, cutoff=cutoff, max_bond=max_bond
    )
    print(trotter_mpo_ham_CIM.max_bond())

# %%
n_sweeps = 20
depth = 3
list_seeds = [1, 2, 3]
for seed in list_seeds:
    cost_tn_closed = init_cost_tn(
        unitary_mpo=trotter_mpo_ham_CIM,
        depth=depth,
        param_scaling=1e-1,
        factorize=False,
        closed=True,
        seed=seed,
    )

    dict_transf = find_transfer_structure(n_qubits=L, cost_tn=cost_tn_closed)

    dict_contr_envs = build_first_sweep(
        n_qubits=L, cost_tn=cost_tn_closed, dict_transf=dict_transf, drop_tags=True
    )

    opt_cost_tn, opt_dict_contr_envs = optimize_single_gate_update(
        n_qubits=L,
        cost_tn=cost_tn_closed,
        n_sweeps=n_sweeps,
        dict_transf=dict_transf,
        dict_contr_envs=dict_contr_envs,
    )

# %% [markdown]
# In this model we also detect dependence on the initial Ansatz. We can either change the seed or the amplitude of the random initialization of the angles:

# %%
n_sweeps = 20
depth = 3
list_seeds = [1, 2, 3]
for seed in list_seeds:
    cost_tn_closed = init_cost_tn(
        unitary_mpo=trotter_mpo_ham_CIM,
        depth=depth,
        param_scaling=1.0,
        factorize=False,
        closed=True,
        seed=seed,
    )

    dict_transf = find_transfer_structure(n_qubits=L, cost_tn=cost_tn_closed)

    dict_contr_envs = build_first_sweep(
        n_qubits=L, cost_tn=cost_tn_closed, dict_transf=dict_transf, drop_tags=True
    )

    opt_cost_tn, opt_dict_contr_envs = optimize_single_gate_update(
        n_qubits=L,
        cost_tn=cost_tn_closed,
        n_sweeps=n_sweeps,
        dict_transf=dict_transf,
        dict_contr_envs=dict_contr_envs,
    )
