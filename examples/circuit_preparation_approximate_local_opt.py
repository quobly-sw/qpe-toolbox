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
# # Circuit Preparation via Local Optimization with Approximate Tensor Contraction
#
# This script implements a local optimization to prepare a quantum circuit that approximates the ground state of a given Hamiltonian. Unlike the straightforward global fitting of the entire circuit (as in `circuit_preparation_global_opt.py`), this version performs a **sweeping algorithm** that optimizes the circuit layer by layer, sweeping from the bottom (first layer) to the top (last layer) and back, much like a DMRG sweep.
# The method is useful when the circuit is deep. By sequentially updating each layer while approximating the rest of the circuit with finite bond dimensions MPSs, we can efficiently achieve high fidelity with the target MPS.
#
# We demonstrate the technique on the **1D transverse-field Ising model**, using a brick-wall SU(4) circuit. The target state is obtained via DMRG. The algorithm constructs the circuit's MPS layer by layer, and at each step it uses the tensor-network fitting routine to update the tensors of a given layer so that the overlap with the target state is maximized.
#
# The main steps are:
# 1. Define the Hamiltonian and compute its ground state `GS` via DMRG.
# 2. Build an initial random SU(4) brick-wall circuit of a given depth.
# 3. Perform an **initial sweep** (down then up) that optimizes the parameters layer by layer, reusing the partially optimized MPS from previous layers.
# 4. After each full sweep, compute the energy and fidelity of the current circuit against the DMRG target.
# 5. Repeat for a number of sweeps (500 in this example) to converge.
#
# This approach is inspired by the **local optimization** idea but applies it in a sequential, sweep-based fashion, which often leads to faster convergence for deep circuits.

# %%
import os

# Limit threading for reproducibility and performance.
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import autoray
import numpy as np
import quimb.tensor as qtn
import tqdm
from quimb.tensor import DMRG2
from quimb.tensor.contraction import contract_strategy

from qpe_toolbox.circuit import ansatz_circuit_su4
from qpe_toolbox.hamiltonian import Hamiltonian

# %% [markdown]
# ## Tensor Network Fitting (ALS Core)
#
# The function `_tn_fit_core` is the workhorse of the fitting procedure. It implements the **alternating least-squares** update:
#
# - Each variable tensor (the one we want to optimize) is identified by a unique tag `"__VARi__"` and also carries the tag `"__KET__"` to separate it from the target part of the network.
# - We form the overlap network `tnAB = tn_fit * conj(tn_target)`. Its contraction gives the overlap `<tn_fit | tn_target>`.
# - For each variable tensor, we compute its **environment** `b` by contracting all other tensors in `tnAB`. This environment is the partial derivative of the overlap with respect to that tensor.
# - The optimal update (given the rest) is the **polar decomposition** of `b`: we reshape `b` into a matrix (combining left and right indices), perform an SVD `b = U S V^dagger`, and set the new tensor to `U V^dagger` (the unitary part). This maximizes the overlap with the target.
# - We iterate over all variable tensors (one sweep) and repeat for a number of steps or until convergence.
#
# The wrapper `tn_fit` handles tagging, combining with the target, and copying the optimized data back to the original tensor network.

# %%
def _tn_fit_core(
    var_tags,
    tnAB,
    tol,
    steps,
    *,
    progbar=False,
):
    """
    Core ALS optimization loop for tensor network fitting.

    Parameters
    ----------
    var_tags : list of str
        Unique tags identifying the variable tensors to optimize.
    tnAB : TensorNetwork
        The overlap network (ket part + conjugated target).
    tol : float
        Convergence tolerance on the change of the overlap.
    steps : int
        Maximum number of sweeps.
    progbar : bool
        Whether to show a progress bar.
    """

    xp = tnAB.get_namespace()

    # Pre-compute environment contractions for each variable tensor.
    env_contractions = []
    for tg in var_tags:
        # The variable tensor is identified by the "__KET__" tag and its unique var tag.
        tb = tnAB["__KET__", tg]
        lix = tb.left_inds
        rix = tuple(x for x in tb.inds if x not in tb.left_inds)
        # The environment is everything except this tensor.
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
            # Optimal update: U @ V^dagger (the unitary part).
            x = u @ v
            x_r = xp.reshape(x, tb.shape)
            tb.modify(data=xp.conj(x_r))

        # Check convergence based on the change in the overlap.
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
    Fit a tensor network `tn` to a target tensor network `tn_target`.

    This is the external interface that tags the tensors to optimize,
    builds the overlap network, calls the ALS core, and copies the results back.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to be optimized (modified in-place).
    tn_target : TensorNetwork
        The target network (usually a state we want to approximate).
    tags : str or list of str, optional
        Tags selecting which tensors of `tn` to optimize. If None, all are optimized.
    steps : int
        Number of ALS sweeps.
    tol : float
        Convergence tolerance.
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

    # Build the overlap network: <tn_fit | tn_target>.
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
# ## Script Execution: Layer-by-Layer Sweep
#
# We now apply the tensor-network fitting in a **sweeping** fashion:
#
# - We start from an initial product state `psi0 = |0...0>`.
# - We build the circuit layer by layer. The ansatz consists of SU(4) gates arranged in a brick-wall pattern. Each layer is a collection of two-qubit gates that act on disjoint pairs.
# - We maintain two lists of MPS: `mpsK` (the "forward" state, built from the bottom) and `mpsB` (the "backward" state, built from the top). The goal is to make the overlap `<mpsB|circuit|mpsK>` as close to 1 as possible, where `circuit` is the full unitary.
# - In the **sweep-down** phase, we start from the top layer and work downwards. For each layer, we:
#     - Form a trial circuit consisting of the current `mpsK` (which already contains all layers below this one) plus the gates of the current layer.
#     - Fit the tensors of that layer so that the resulting MPS matches the current `mpsB` (which contains all layers above this one, already optimized).
#     - After updating the layer, we update `mpsB` by applying the conjugated gates (using MPO) to move one layer down.
# - In the **sweep-up** phase, we reverse the direction: we start from the bottom and work upwards, updating each layer similarly.
# - After each full sweep (down+up), we evaluate the energy and fidelity of the full circuit against the DMRG target.
#
# This approach is very similar to the DMRG sweep algorithm, but with the roles of the Hamiltonian and the unitary gates interchanged. It allows us to efficiently optimize deep circuits while keeping bond dimensions manageable.
#
# We run 500 sweeps; the algorithm should converge to a state that closely approximates the ground state.

# %% [markdown]
# ### 1. Hamiltonian and DMRG Reference
# We set up the 1D transverse-field Ising model:
# \( H = \sum_i g_x X_i + \sum_i g_{zz} Z_i Z_{i+1} \)
# with \( g_x = -1.1 \), \( g_{zz} = -1.0 \). We take 16 qubits.
# The ground state is computed with DMRG (bond dimension 64, 16 sweeps) and used as the target.

# %%
## hamiltonian
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

# Run DMRG to get the target state and energy.
dmrg = DMRG2(mpo)
dmrg.solve(max_sweeps=16, tol=1e-8, bond_dims=64, verbosity=0)
GS = dmrg.state
dmrg_energy = np.real(dmrg.energy)
print("*** DMRG reference energy:", dmrg_energy)
print()

# %% [markdown]
# ### 2. Circuit Initialization and Layer Preparation
#
# We build an ansatz circuit of a given `depth` (here 6). The circuit is a brick-wall of SU(4) gates.
# We also build an initial product state `psi0 = |0...0>`.
#
# Then we create two lists:
# - `mpsK`: starts with `psi0`. For each layer (from bottom to top), we apply the corresponding gates as MPOs to build the forward MPS.
# - `mpsB`: starts with the conjugate of the target state `GS.H`. This will be updated as we sweep from the top downwards.
#
# These lists are used in the sweeping algorithm to keep track of the partially contracted states.

# %%
## initialise
depth = 6
psi0 = qtn.MPS_computational_state("0" * n_qubits)
circ = ansatz_circuit_su4(
    n_qubits=n_qubits, depth=depth, param_scaling=0.1, parametrize=False, psi0=psi0
)
circ_P = circ.psi
circ_G = circ.gates

# Build the forward MPS for each layer (from bottom to top).
mpsB = []
mpsB.append(GS.H)
mpsK = []
mpsK.append(psi0)
for ii in range(depth - 1):
    # Get gates belonging to this round (layer).
    mpos = [gate.build_mpo(L=n_qubits) for gate in circ_G if gate.round == ii]
    tmp = mpsK[-1]
    for jj in mpos:
        tmp = tmp.gate_with_submpo(jj, transpose=False, max_bond=32, cutoff=1e-10)
    mpsK.append(tmp)

# %% [markdown]
# ### 3. Sweeping Optimization
#
# We perform a number of full sweeps (`sweep` from 0 to 499). Each sweep consists of two phases:
#
# - **Sweep down**: from the top layer (`depth-1`) down to layer 0.
#     - For a given layer `ii` (top to bottom), we construct a trial circuit `trial` from `mpsK[-1]` (which contains all layers below) and the gates of that layer.
#     - We call `tn_fit` to optimize the tensors of that layer (tagged `'SU4'`) so that the trial MPS matches the current `mpsB[-1]` (which contains all layers above, already optimized).
#     - We then copy the optimized tensor data back into the main circuit representation (`circ_P` and `circ_G`).
#     - We update `mpsB` by applying the **conjugated** gates (as MPOs) to move the boundary one layer down.
#
# - **Sweep up**: from the bottom layer (0) up to `depth-1`.
#     - Similar to sweep down, but we start from `mpsK[-1]` and update layers from bottom to top.
#     - We also update `mpsK` by applying the (non-conjugated) gates to move the forward boundary up.
#
# After each full sweep, we contract the full circuit with the MPO to compute the energy and the overlap with the DMRG state. This gives a measure of convergence.

# %%
# Print a header for the convergence output
print("Sweep    Energy         Error       1-Fidelity")
print("------------------------------------------------")

ene_old = float("nan")
ene = float("nan")

for sweep in range(1000):
    ## Sweep down: optimize layers from top to bottom.
    for ii in range(depth - 1):
        # Build trial circuit from current mpsK (layers below) + gates of this layer.
        trial = qtn.Circuit(psi0=mpsK.pop())
        gates = [gate for gate in circ_G if gate.round == depth - 1 - ii]
        for gate in gates:
            trial.apply_gate(gate)

        tn = trial.psi
        # Fit the SU4 tensors of this layer to match the current mpsB (layers above).
        tn_fit(
            tn,
            mpsB[-1].H,  # note: mpsB stores the conjugate, so we use its adjoint.
            tags="SU4",
            steps=10,  # inner ALS sweeps per layer update
            tol=1e-12,
            contract_optimize="auto-hq",
            progbar=False,
        )

        # Copy the optimized tensor data back into the master circuit tensors.
        g0 = tn.select_tensors("ROUND_" + str(depth - 1 - ii), "any")
        g1 = circ_P.select_tensors("ROUND_" + str(depth - 1 - ii), "any")
        g2 = [gate for gate in circ_G if gate.round == depth - 1 - ii]
        for t0, t1, t2 in zip(g0, g1, g2, strict=True):
            t1.modify(data=t0.data)
            t2._array = t0.data  # noqa: SLF001

        # Update mpsB: apply the conjugated gates (as MPOs) to move down one layer.
        mpos = [gate.build_mpo(L=n_qubits) for gate in g2]
        mpos.reverse()  # reverse order because we are conjugating and moving down

        tmp = mpsB[-1]
        for jj in mpos:
            tmp = tmp.gate_with_submpo(jj, transpose=True, max_bond=32, cutoff=1e-12)
        mpsB.append(tmp)

    ## Sweep up: optimize layers from bottom to top.
    for ii in range(depth - 1):
        # Build trial circuit from current mpsK[-1] (which has all layers below) + gates of this layer.
        trial = qtn.Circuit(psi0=mpsK[-1])
        gates = [gate for gate in circ_G if gate.round == ii]
        for gate in gates:
            trial.apply_gate(gate)

        tn = trial.psi
        # Fit the SU4 tensors of this layer to match the current mpsB (top part, already optimized).
        tn_fit(
            tn,
            mpsB.pop().H,  # use the top part; pop removes the last element (which corresponds to the layer just below)
            tags="SU4",
            steps=10,
            tol=1e-12,
            contract_optimize="auto-hq",
            progbar=False,
        )

        # Copy optimized data back to master circuit.
        g0 = tn.select_tensors("ROUND_" + str(ii), "any")
        g1 = circ_P.select_tensors("ROUND_" + str(ii), "any")
        g2 = [gate for gate in circ_G if gate.round == ii]
        for t0, t1, t2 in zip(g0, g1, g2, strict=True):
            t1.modify(data=t0.data)
            t2._array = t0.data  # noqa: SLF001

        # Update mpsK: apply the (non-conjugated) gates to move the forward boundary up.
        mpos = [gate.build_mpo(L=n_qubits) for gate in g2]

        tmp = mpsK[-1]
        for jj in mpos:
            tmp = tmp.gate_with_submpo(jj, transpose=False, max_bond=32, cutoff=1e-12)
        mpsK.append(tmp)

    # ## 4. Evaluation After Each Full Sweep
    #
    # We contract the full circuit MPS `tn = circ_P` with the Hamiltonian MPO to compute the energy,
    # and also compute the overlap with the DMRG target state to get the fidelity.
    # These numbers indicate how well the circuit approximates the ground state.

    tn = circ_P
    ovlp = (dmrg.state.H & tn).contract()
    tnH = tn.H
    tn.align_(mpo, tnH)
    energy_tn = tnH & mpo & tn
    ene = autoray.do("real", energy_tn.contract(all))
    error = np.abs(1 - ene / dmrg_energy)
    infidelity = 1 - np.abs(ovlp) ** 2
    print(f"{sweep:5d}   {ene:12.8f}   {error:10.3e}   {infidelity:10.3e}")

    if abs(1 - ene / ene_old) < 1e-8:
        break
    else:
        ene_old = ene
