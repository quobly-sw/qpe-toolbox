# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

import autoray
import quimb.tensor as qtn
import tqdm


def _tn_fit_core(
    var_tags,
    tnAB,
    tol,
    steps,
    *,
    progbar=False,
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
    var_tags : list of hashable
        Tags identifying the variable tensors in the network (each must also carry the
        tag "__KET__").
    tnAB : TensorNetwork-like object
        The full tensor network containing both the variable tensors and the fixed
        environment. It must support selection by tags and contraction to dense arrays.
    tol : float
        Convergence tolerance on the change of the objective function (Frobenius inner
        product between the updated tensor and its environment). If 0, no convergence
        check is performed (runs for exactly `steps` iterations).
    steps : int
        Maximum number of full sweeps (each sweep updates all variable tensors once).
    progbar : bool, optional
        If True, display a progress bar showing the current objective value.

    Returns
    -------
    None
        The variable tensors in `tnAB` are updated in-place.

    Notes
    -----
    The algorithm assumes that each variable tensor is associated with the tag "__KET__"
    and one of the `var_tags`. The environment for a given variable is the rest of the
    network after contracting all other tensors. The optimal update is derived from the
    SVD of the environment reshaped as a matrix: if the environment is B, and we seek
    a tensor X (same shape as the variable) that maximizes the real part of the inner
    product ⟨X|B⟩ (or equivalent), the solution is given by X = U V^†, where B = U S V^†.
    The implementation uses the conjugate of this update to conform to the network's
    contraction conventions.
    """
    xp = tnAB.get_namespace()  # Get the array module (e.g., numpy, cupy)

    # --------------------------------------------------------------------------
    # Precompute environment contractions for each variable tensor.
    # For each variable, we store:
    #   - the tensor object itself
    #   - the environment network (all tensors except this one)
    #   - the left and right indices that define the matrix reshaping
    # --------------------------------------------------------------------------
    env_contractions = []
    for tg in var_tags:
        # The variable tensor is identified by the tag "__KET__" and the variable tag.
        tb = tnAB["__KET__", tg]
        # Determine left (outer) and right (inner) indices for matrix reshaping.
        # Typically, left_inds are the indices that connect to the environment on one side.
        lix = tb.left_inds
        rix = tuple(x for x in tb.inds if x not in tb.left_inds)
        # The rest of the network (all tensors except this one) forms the environment.
        b_tn = tnAB.select((tg,), "!all")  # select all except those with tag tg
        env_contractions.append((tb, b_tn, lix, rix))

    # Initialize objective value for convergence tracking if needed.
    if tol != 0.0 or progbar:
        old_d = float("inf")

    # Set up progress bar if requested.
    if progbar:
        pbar = tqdm.trange(steps)
    else:
        pbar = range(steps)

    # --------------------------------------------------------------------------
    # Main sweep loop: each iteration updates all variable tensors once.
    # --------------------------------------------------------------------------
    for _ in pbar:
        # Sweep over all variable tensors.
        for tb, b_tn, lix, rix in env_contractions:
            # Contract the environment (all other tensors) to a dense matrix.
            # The contraction is performed with the indices ordered as (rix, lix)
            # so that the resulting matrix B has shape compatible with the tensor.
            b = b_tn.to_dense(rix, lix)

            # Perform SVD of the environment matrix: B = U S V^†.
            # In most libraries, svd returns U, S, Vh (V conjugate transpose).
            u, _, v = xp.linalg.svd(b)  # v is Vh

            # Optimal unitary update: X = U V^† (i.e., U @ Vh).
            # This maximizes the overlap with the environment under a unitary constraint.
            x = u @ v

            # Reshape the matrix back to the original tensor shape.
            x_r = xp.reshape(x, tb.shape)

            # Update the tensor in-place with the conjugate of the optimal matrix.
            # The conjugation accounts for the fact that the network contraction
            # may involve complex conjugation of the variable (depending on the
            # bra/ket convention).
            tb.modify(data=xp.conj(x_r))

        # ----------------------------------------------------------------------
        # Convergence check: compute the objective value for the last updated tensor.
        # The objective is the real part of the Frobenius inner product between the
        # updated tensor and its environment (or trace for a matrix).
        # We use the last `x` and `b` from the loop; if the network is consistent,
        # this gives a measure of the total energy/overlap.
        # ----------------------------------------------------------------------
        if (tol != 0.0) or progbar:
            dagx = autoray.dag(x)  # conjugate transpose of x (matrix form)
            d = float(0)
            if x.ndim == 2:
                # For matrix-shaped tensors, the objective is the trace of dag(x) @ b.
                d = xp.trace(xp.real(dagx @ b))
            else:
                # For higher-order tensors, the objective is the full contraction
                # (real part of the inner product).
                d = xp.real(dagx @ b)

            # Check if the change in objective is below tolerance.
            if abs(d - old_d) < tol:
                break
            old_d = d

        # Update progress bar description with current objective value.
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

    with qtn.contract_strategy(contract_optimize):
        _tn_fit_core(
            var_tags=var_tags,
            tnAB=tnAB,
            tol=tol,
            steps=steps,
            progbar=progbar,
        )

    # Copy optimized data back to the original tensor network.
    for t1, t2 in zip(tn, tn_fit, strict=True):
        t2.transpose_like_(t1)
        t1.modify(data=t2.data)
