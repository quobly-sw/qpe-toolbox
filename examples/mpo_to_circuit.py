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
    sgu_optimize_cost_tn,
    trotter_approx_as_MPO,
)
from qpe_toolbox.hamiltonian import Hamiltonian

list_paulis = ["I", "X", "Y", "Z"]

# %% [markdown]
# <img src="./figures/transpil_Uref.svg" align="center">

# %% [markdown]
# <img src="./figures/transpil_Ubw.svg" align="center">

# %% [markdown]
# <img src="./figures/transpil_cost.svg" align="center">

# %% [markdown]
# <img src="./figures/transpil_state_prep.svg" align="center">

# %% [markdown]
# next-nearest-neighbor Ising model

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
    ham_NNIM.terms,
    n_qubits=ham_NNIM.n_qubits,
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
    tol=1e-1,
    factorize=False,
    closed=False,
)
cost_tn_open.draw([f"ROUND_{i}" for i in range(depth)], show_inds=True, show_tags=False)

cost_tn_closed = init_cost_tn(
    unitary_mpo=trotter_mpo_ham_NNIM,
    depth=depth,
    tol=1e-1,
    factorize=False,
    closed=True,
)
cost_tn_closed.draw(
    [f"ROUND_{i}" for i in range(depth)], show_inds=False, show_tags=False
)

# %%
dict_transf = find_transfer_structure(n_qubits=L, cost_tn=cost_tn_closed)


"""
For key, items in dict_transf["L"].items():

print(key)
cost_tn_closed.draw(items, show_inds=False, show_tags=False)
"""
"""
For key, items in dict_transf["R"].items():

print(key)
cost_tn_closed.draw(items, show_inds=False, show_tags=False)
"""

# %%
dict_contr_envs = build_first_sweep(
    n_qubits=L, cost_tn=cost_tn_closed, dict_transf=dict_transf, drop_tags=True
)

# %%
# optimize for "depth" ("2*depth" in Causer et al. paper)

list_n_sweeps = [10, 20, 50, 100]

for n_sweeps in list_n_sweeps:
    cp_cost_tn_closed = cost_tn_closed.copy(deep=True)
    cp_dict_transf = copy.deepcopy(dict_transf)
    cp_dict_contr_envs = copy.deepcopy(dict_contr_envs)

    opt_cost_tn, opt_dict_contr_envs = sgu_optimize_cost_tn(
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
        tol=1e-1,
        factorize=False,
        closed=True,
        seed=seed,
    )

    dict_transf = find_transfer_structure(n_qubits=L, cost_tn=cost_tn_closed)

    dict_contr_envs = build_first_sweep(
        n_qubits=L, cost_tn=cost_tn_closed, dict_transf=dict_transf, drop_tags=True
    )

    opt_cost_tn, opt_dict_contr_envs = sgu_optimize_cost_tn(
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
# cluster Ising model

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
    ham_CIM.terms,
    n_qubits=ham_CIM.n_qubits,
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
        tol=1e-1,
        factorize=False,
        closed=True,
        seed=seed,
    )

    dict_transf = find_transfer_structure(n_qubits=L, cost_tn=cost_tn_closed)

    dict_contr_envs = build_first_sweep(
        n_qubits=L, cost_tn=cost_tn_closed, dict_transf=dict_transf, drop_tags=True
    )

    opt_cost_tn, opt_dict_contr_envs = sgu_optimize_cost_tn(
        n_qubits=L,
        cost_tn=cost_tn_closed,
        n_sweeps=n_sweeps,
        dict_transf=dict_transf,
        dict_contr_envs=dict_contr_envs,
    )

# %% [markdown]
# In this model we also detect dependence on the initial Ansatz. We can either change the seed or the tolerance:

# %%
n_sweeps = 20
depth = 3
list_seeds = [1, 2, 3]
for seed in list_seeds:
    cost_tn_closed = init_cost_tn(
        unitary_mpo=trotter_mpo_ham_CIM,
        depth=depth,
        tol=1.0,
        factorize=False,
        closed=True,
        seed=seed,
    )

    dict_transf = find_transfer_structure(n_qubits=L, cost_tn=cost_tn_closed)

    dict_contr_envs = build_first_sweep(
        n_qubits=L, cost_tn=cost_tn_closed, dict_transf=dict_transf, drop_tags=True
    )

    opt_cost_tn, opt_dict_contr_envs = sgu_optimize_cost_tn(
        n_qubits=L,
        cost_tn=cost_tn_closed,
        n_sweeps=n_sweeps,
        dict_transf=dict_transf,
        dict_contr_envs=dict_contr_envs,
    )
