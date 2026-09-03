# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

import numpy as np
import quimb.tensor as qtn
import tqdm


def svd_optimal_gate_update(tensor, left_inds):
    r"""
    Solve for the SVD-optimal isometric update of a variational tensor.

    ``tensor`` is the environment of a variational tensor: the rest of a
    tensor network, fully contracted, with that tensor removed. This is the
    closed-form solution of the associated *unconstrained linear* problem:
    the isometry $X$ maximizing $\mathrm{Re}\,\mathrm{Tr}(X^\dagger B)$ is
    $X = U V^\dagger$, obtained by discarding the singular values of the SVD
    $B = U S V^\dagger$.

    Parameters
    ----------
    tensor : :quimb-api:`Tensor`
        Contracted environment tensor.
    left_inds : sequence of str
        Subset of ``tensor``'s indices forming the left side of the SVD
        bipartition; the remaining indices form the right side.

    Returns
    -------
    new_isometry : :quimb-api:`Tensor`
        Optimal isometric tensor, carrying the same indices as ``tensor``.
    objective : float
        Sum of the discarded singular values, i.e. the achieved value of
        $\mathrm{Re}\,\mathrm{Tr}(X^\dagger B)$.
    """
    svd_factors = qtn.tensor_split(
        T=tensor, left_inds=left_inds, method="svd", absorb=None
    )
    objective = np.sum(svd_factors.tensors[1].data)
    new_isometry = (svd_factors.tensors[0].conj() & svd_factors.tensors[2].conj()) ^ ...
    return new_isometry, objective


def _tn_fit_core(
    var_tags,
    tnAB,
    tol,
    steps,
):
    """
    Core optimization loop for fitting a tensor network using alternating least squares (ALS).

    This function updates a set of variable tensors (identified by `var_tags`) within a
    tensor network `tnAB` to maximize the overlap (or minimize the energy) with the
    rest of the network. It performs a sweep over the variable tensors, updating each
    one by contracting its environment, performing a singular value decomposition (SVD),
    and replacing the tensor with the optimal unitary (or general) transformation.

    Parameters
    ----------
    var_tags : iterable of hashable
        Tags identifying the variable tensors in the network (each must also carry the
        tag "__KET__"). Consumed by a single pass, so any iterable works.
    tnAB : TensorNetwork-like object
        The full tensor network containing both the variable tensors and the fixed
        environment. It must support selection by tags and contraction to dense arrays.
    tol : float
        Convergence tolerance on the change of the objective function (Frobenius inner
        product between the updated tensor and its environment). If 0, no convergence
        check is performed (runs for exactly `steps` iterations).
    steps : int
        Maximum number of full sweeps (each sweep updates all variable tensors once).

    Returns
    -------
    None
        The variable tensors in `tnAB` are updated in-place.

    Notes
    -----
    The algorithm assumes that each variable tensor is associated with the tag "__KET__"
    and one of the `var_tags`. The environment for a given variable is the rest of the
    network after contracting all other tensors. Each update is the closed-form SVD
    solution computed by :func:`svd_optimal_gate_update`. Progress is shown via a
    :mod:`tqdm` bar; set the ``TQDM_DISABLE`` environment variable to silence it.
    """
    # --------------------------------------------------------------------------
    # Precompute the environment sub-network and left/right bipartition for each
    # variable tensor.
    # --------------------------------------------------------------------------
    env_contractions = []
    for tg in var_tags:
        # The variable tensor is identified by the tag "__KET__" and the variable tag.
        tb = tnAB["__KET__", tg]
        # The rest of the network (all tensors except this one) forms the environment.
        b_tn = tnAB.select((tg,), "!all")  # select all except those with tag tg
        env_contractions.append((tb, b_tn, tb.left_inds))

    # Initialize objective value for convergence tracking if needed.
    if tol != 0.0:
        old_objective = float("inf")

    pbar = tqdm.trange(steps)

    # --------------------------------------------------------------------------
    # Main sweep loop: each iteration updates all variable tensors once.
    # --------------------------------------------------------------------------
    for _ in pbar:
        # Sweep over all variable tensors.
        for tb, b_tn, left_inds in env_contractions:
            # Contract the environment (all other tensors) into a single tensor,
            # then replace the variable tensor with its SVD-optimal update.
            b_tensor = b_tn ^ ...
            new_isometry, objective = svd_optimal_gate_update(b_tensor, left_inds)
            tb.modify(data=new_isometry.transpose_like(tb).data)

        # ----------------------------------------------------------------------
        # Convergence check: `objective` is the value reached by the last
        # updated tensor of the sweep; if the network is consistent, this
        # gives a measure of the total energy/overlap.
        # ----------------------------------------------------------------------
        if tol != 0.0:
            # Check if the change in objective is below tolerance.
            if abs(objective - old_objective) < tol:
                break
            old_objective = objective

        pbar.set_description(f"{objective:.4g}")


def tn_fit(
    tn,
    tn_target,
    tags="SU4",
    steps=100,
    tol=1e-8,
    contract_optimize="auto-hq",
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

    Notes
    -----
    Each tensor selected by `tags` must be a single, whole gate tensor. If
    `tn` comes from a :quimb-api:`Circuit` built with the default
    ``gate_contract='auto-split-gate'``, a gate can be silently split into
    two fragments when the neighboring bond dimension is small (e.g. the
    circuit is converging toward a low-entanglement state) -- pass
    ``gate_contract=False`` when constructing that :quimb-api:`Circuit` to
    avoid it.

    Progress is shown via a :mod:`tqdm` bar; set the ``TQDM_DISABLE``
    environment variable to silence it.
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

    with qtn.contract_strategy(contract_optimize):
        _tn_fit_core(
            var_tags=var_tags,
            tnAB=tnAB,
            tol=tol,
            steps=steps,
        )

    # Copy optimized data back to the original tensor network.
    for t1, t2 in zip(tn, tn_fit, strict=True):
        t2.transpose_like_(t1)
        t1.modify(data=t2.data, left_inds=t1.left_inds)
