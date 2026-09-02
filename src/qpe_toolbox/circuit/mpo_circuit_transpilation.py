import re

import quimb.tensor as qtn
from tqdm import tqdm

from qpe_toolbox.circuit.circuits_opt import svd_optimal_gate_update
from qpe_toolbox.circuit.parametrized_circuits import (
    generate_brickwall_circuit,
)


def init_cost_tn(ref_mpo, depth, *, param_scaling=1e-1, closed=False, rng=None):
    r"""
    Initialize the tensor network used for optimization.

    It represents a cost function obtained from contracting:

    1. A brickwall unitary circuit ansatz,
    2. A reference target unitary (or state x zero) register represented as an MPO.

    The ansatz is generated as a layered brickwall circuit of two-qubit
    ``SU4`` gates initialized close to the identity. The resulting unitary
    tensor network is then combined with the target MPO into a single tensor
    network suitable for overlap evaluation.

    Parameters
    ----------
    ref_mpo : :quimb-api:`MatrixProductOperator`
        Target MPO to be reproduced by the Ansatz.
    depth : int
        Depth of the brickwall circuit ansatz (even and odd count as 1).
    param_scaling : float, optional
        Scale of the random initialization parameters for the gates.
        Smaller values initialize the circuit closer to the identity.
        Default is ``1e-1``.
    closed : bool, optional
        If ``False`` (default), the bra indices of the MPO remain open.
        If ``True``, ket indices are contracted (trace contraction).
    rng : :numpy-random:`numpy.random.Generator <generator>`, optional
        Random number generator to generate gate parameters.
        If ``None``, a default generator is created.

    Returns
    -------
    :quimb-api:`TensorNetwork`
        Tensor network containing both the variational circuit ansatz and
        the target MPO.
    """

    n_qubits = ref_mpo.num_tensors

    # -----------------------------------------
    # Define the Ansatz as a brickwall circuit

    bw_circ = generate_brickwall_circuit(
        n_qubits=n_qubits,
        depth=depth,
        one_qubit_gate_label="U1",  # it is irrelevant (purely_ent cancels its effect)
        two_qubit_gate_label="SU4",
        start_ent=True,
        include_1qubit_gates=False,  # whether or not to do 1-spin rotations
        param_scaling=param_scaling,  # initialize close to identity
        rng=rng,
    )
    cost_tn = qtn.TensorNetwork() & bw_circ.get_uni()

    # -----------------------------------
    # Add the MPO in the network

    new_ket_ind = "B"  # by defect leave the (B)ra open
    if closed:
        new_ket_ind = "k"  # execute the trace by contracting the (k)et

    for x, tensor in enumerate(ref_mpo):
        reind_tens = tensor.copy(deep=True)
        old_inds = reind_tens.inds

        if x == 0 or x == n_qubits - 1:
            inds = (next(iter(old_inds)), new_ket_ind + str(x), f"b{x}")
        else:
            inds = (
                next(iter(old_inds)),
                list(old_inds)[1],
                new_ket_ind + str(x),
                f"b{x}",
            )

        reind_tens.modify(inds=inds, tags=(f"I{x}", f"Uref{x}", "MPO"))
        cost_tn &= reind_tens

    return cost_tn


def get_envs_tns(n_qubits, site_index, cost_tn):
    r"""
    Extract the left and right environment TNs associated to a given site.

    This function removes the tensors tagged with ``I{site_index}`` from the
    full cost tensor network and partitions the remaining network into
    left and/or right environments relative to ``site_index``.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (sites) in the tensor network.
    site_index : int
        Site index for which the environments are constructed.
    cost_tn : :quimb-api:`TensorNetwork`
        Full cost tensor network.

    Returns
    -------
    list of :quimb-api:`TensorNetwork`
        List containing the environment tensor networks.

        - If ``site_index == 0``, only the right environment is returned.
        - If ``site_index == n_qubits - 1``, only the left environment is
          returned.
        - Otherwise, the list is ordered as ``[left_env, right_env]``.
    """

    env_tn = cost_tn.select(tags=[f"I{site_index}"], which="!any")

    left_tags = [f"I{i}" for i in range(site_index)]
    right_tags = [f"I{i}" for i in range(site_index + 1, n_qubits)]

    env_tns = []
    if site_index > 0:
        env_tns.append(env_tn.select(tags=left_tags, which="any"))
    if site_index < n_qubits - 1:
        env_tns.append(env_tn.select(tags=right_tags, which="any"))

    return env_tns


def find_transfer_structure(n_qubits, cost_tn):
    r"""
    Determine the transfer structures connecting neighboring environments in a cost TN.

    This function analyzes the left and right environment tensor networks
    associated with each site and identifies the tensor tags involved in
    transferring contractions between neighboring environments.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (sites) in the tensor network.
    cost_tn : TensorNetwork
        Full cost tensor network.

    Returns
    -------
    dict
        Nested dictionary containing the transfer structures.

        The returned dictionary has the form:

        .. code-block:: python

            {
                "L": {
                    "L1_to_L2": [...],
                    ...
                    "L{n_qubits-1}_to_L{n_qubits}": [...]
                },
                "R": {
                    "R{n_qubits-2}_to_R{n_qubits-1}": [...],
                    ...
                    "R1_to_R0": [...]
                }
            }

        where each value is a list of tensor tags participating in the
        corresponding transfer operation.
    """

    left_uncontracted = {}
    right_uncontracted = {}
    for x in range(n_qubits):
        env_tns = get_envs_tns(n_qubits, x, cost_tn)

        if x == 0:
            right_uncontracted[f"I{x}"] = env_tns[0]
        elif x == n_qubits - 1:
            left_uncontracted[f"I{x}"] = env_tns[0]
        else:
            left_uncontracted[f"I{x}"] = env_tns[0]
            right_uncontracted[f"I{x}"] = env_tns[1]

    transfer_structure = {"L": {}, "R": {}}
    for x in range(1, n_qubits - 1):
        # left transfer
        transf_tn = left_uncontracted[f"I{x + 1}"].select(tags=(f"I{x}"), which="any")
        filtered_tags = [tag for tag in transf_tn.tags if tag[0] in ("G", "U")]
        transfer_structure["L"][f"L{x}_to_L{x + 1}"] = filtered_tags

        # right transfer
        transf_tn = right_uncontracted[f"I{n_qubits - 2 - x}"].select(
            tags=(f"I{n_qubits - 1 - x}"), which="any"
        )
        filtered_tags = [tag for tag in transf_tn.tags if tag[0] in ("G", "U")]
        transfer_structure["R"][f"R{n_qubits - 1 - x}_to_R{n_qubits - 2 - x}"] = (
            filtered_tags
        )

    return transfer_structure


def build_first_sweep(n_qubits, cost_tn, transfer_structure, *, drop_tags=True):
    r"""
    Construct the initial set of contracted left and right environments for a sweeping TN optimization.

    This function iteratively builds partially contracted environments by
    propagating contractions from the edges of the tensor network toward
    the center using the transfer structures generated by
    :func:`find_transfer_structure`.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (sites) in the tensor network.
    cost_tn : :quimb-api:`TensorNetwork`
        Full cost tensor network.
    transfer_structure : dict
        Transfer structure dictionary generated by
        :func:`find_transfer_structure`.
    drop_tags : bool, optional
        Whether to drop tensor tags during contractions.
        Passed to :func:`tensor_contract`.
        Default is ``True``.

    Returns
    -------
    dict
        Nested dictionary containing contracted environments:

        .. code-block:: python

            {
                "L": {
                    "L1": Tensor,
                    ...
                },
                "R": {
                    "R{n_qubits-2}": Tensor,
                    ...
                }
            }

        Each entry corresponds to an effective contracted environment tensor.
    """

    contracted_envs = {"L": {}, "R": {}}

    # the first environments are L{1} and R{n_qubits-1}, which are the edge tensors on the MPO
    left_env = cost_tn.select(tags="Uref0", which="all").tensors[0]
    left_env.add_tag(tag="L1")
    right_env = cost_tn.select(tags=f"Uref{n_qubits - 1}", which="all").tensors[0]
    right_env.add_tag(tag=f"R{n_qubits - 2}")

    contracted_envs["L"]["L1"] = left_env
    contracted_envs["R"][f"R{n_qubits - 2}"] = right_env

    for counter, transf_tags in enumerate(
        zip(
            transfer_structure["L"].values(),
            transfer_structure["R"].values(),
            strict=True,
        )
    ):
        left_env &= cost_tn.select(tags=transf_tags[0], which="any")
        left_env = qtn.tensor_contract(*left_env.tensors, drop_tags=drop_tags)
        left_env.add_tag(tag=f"L{counter + 2}")

        contracted_envs["L"][f"L{counter + 2}"] = left_env

        right_env &= cost_tn.select(tags=transf_tags[1], which="any")
        right_env = qtn.tensor_contract(*right_env.tensors, drop_tags=drop_tags)
        right_env.add_tag(tag=f"R{n_qubits - 3 - counter}")

        contracted_envs["R"][f"R{n_qubits - 3 - counter}"] = right_env

    return contracted_envs


def build_loc_cost_tn(n_qubits, site_index, contracted_envs, cost_tn):
    r"""
    Construct the local cost tensor network associated with a given site.

    This function combines the local gate tensors acting on site
    ``site_index`` with the corresponding contracted left and/or right
    environments to form an effective local tensor network suitable for
    optimization.

    It also returns the ordered list of gate tags corresponding to the
    variational tensors to optimize.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (sites) in the tensor network.
    site_index : int
        Site index for which the local cost tensor network is constructed.
    contracted_envs : dict
        Dictionary of contracted environments generated by
        :func:`build_first_sweep`.
    cost_tn : :quimb-api:`TensorNetwork`
        Full cost tensor network.

    Returns
    -------
    loc_cost_tn : :quimb-api:`TensorNetwork`
        Local effective tensor network containing the relevant environments
        and gate tensors for site ``site_index``.
    gate_to_opt_tags : list of str
        Ordered list of gate tensor tags to optimize.

    Notes
    -----
    The local tensor network has the structure

    .. math::

        \mathcal{S}_x = L_x \; \text{- gates -} \; R_x,

    where:

    - ``L_x`` is the contracted left environment,
    - ``R_x`` is the contracted right environment,
    - ``gates`` are the tensors tagged with ``I{site_index}``.

    Only tags beginning with ``"G"`` are considered optimization gate tags.

    Boundary sites are treated separately:

    - for ``site_index = 0``, only the right environment is included,
    - for ``site_index = n_qubits - 1``, only the left environment is included.
    """
    gates_to_opt = cost_tn.select(tags=f"I{site_index}", which="any")

    if site_index == 0:
        loc_cost_tn = gates_to_opt & contracted_envs["R"][f"R{site_index}"]
    elif site_index == n_qubits - 1:
        loc_cost_tn = contracted_envs["L"][f"L{site_index}"] & gates_to_opt
    else:
        loc_cost_tn = (
            contracted_envs["L"][f"L{site_index}"]
            & gates_to_opt
            & contracted_envs["R"][f"R{site_index}"]
        )

    filtered_tags = [tag for tag in gates_to_opt.tags if tag[0] == "G"]
    gate_to_opt_tags = sorted(filtered_tags, key=lambda s: int(s.split("_")[1]))
    return loc_cost_tn, gate_to_opt_tags


def PRC_loc_cost_tn(loc_cost_tn, tags, optimize):
    r"""
    Perform a Pop-Rehearse-Contract step on a local cost tensor network.

    This function removes a set of tensors from a local cost tensor network
    and contracts the remaining network using a specified contraction strategy.

    The procedure is intended to support efficient repeated contractions during
    sweeping optimization algorithms, where contraction paths may be rehearsed
    and reused across iterations.

    Parameters
    ----------
    loc_cost_tn : :quimb-api:`TensorNetwork`
        Local cost tensor network.
    tags : sequence of str
        Tags identifying the tensors to remove before contraction.
        Typically corresponds to the variational gates currently being optimized.
    optimize : str or :cotengra-api:`HyperOptimizer`
        Contraction optimization strategy passed to
        :meth:`TensorNetwork.contract`, e.g. ``"auto-hq"`` or a
        :cotengra-api:`ReusableHyperOptimizer` instance.

    Returns
    -------
    :quimb-api:`Tensor`
        Contracted tensor obtained after removing the specified tensors.
    """
    p_loc_cost_tn = loc_cost_tn.copy(deep=True)
    p_loc_cost_tn.delete(tags=tags)
    return p_loc_cost_tn.contract(optimize=optimize)


def update_cost_tn(cost_tn, gate_tens):
    r"""
    Update a cost tensor network with a newly optimized gate tensor.

    This function replaces the existing gate tensor in the cost tensor
    network with the updated optimized tensor, keeping the same gate tag.

    Parameters
    ----------
    cost_tn : :quimb-api:`TensorNetwork`
        Full cost tensor network.
    gate_tens : :quimb-api:`Tensor`
        Optimized gate tensor to insert into the network.

    Returns
    -------
    :quimb-api:`TensorNetwork`
        Updated cost tensor network containing the optimized gate tensor.
    """

    # the first tag is "GATE_{n}" by construction
    tag_tens = next(iter(gate_tens.tags))
    cost_tn.delete(tags=tag_tens)  # delete the old tensor with same tags
    cost_tn &= gate_tens  # add the new tensor

    return cost_tn


def update_contracted_envs(
    mode, gate_tens, cost_tn, transfer_structure, contracted_envs
):
    r"""
    Update contracted environments after a local gate optimization.

    This function updates the cached contracted left or right environments
    affected by a newly optimized gate tensor during a sweeping optimization
    procedure.

    Depending on the sweep direction, only the environments influenced by
    the updated gate are recomputed.

    Parameters
    ----------
    mode : {"LR", "RL"}
        Sweep direction.

        - ``"LR"`` : left-to-right sweep,
        - ``"RL"`` : right-to-left sweep.
    gate_tens : :quimb-api:`Tensor`
        Optimized gate tensor.
    cost_tn : :quimb-api:`TensorNetwork`
        Full updated cost tensor network.
    transfer_structure : dict
        Transfer structure dictionary generated by
        :func:`find_transfer_structure`.
    contracted_envs : dict
        Updated dictionary of contracted environments initialized by
        :func:`build_first_sweep`.

    Returns
    -------
    dict
        Updated dictionary of contracted environments.

    Notes
    -----
    The optimized gate acts on neighboring sites ``I{n}`` and ``I{n+1}``.
    Consequently, only nearby environments need to be updated.

    For a left-to-right sweep (``"LR"``):

    .. math::

        L_{n+2} \leftarrow L_{n+1} \cdot T_{n+1 \to n+2},

    while for a right-to-left sweep (``"RL"``):

    .. math::

        R_{n-1} \leftarrow R_n \cdot T_{n \to n-1}.

    This local update strategy avoids rebuilding all environments from
    scratch after each optimization step.
    """
    if mode == "LR":  # sweeping L to R only requires updating L's
        side, shift1, shift2 = "L", 1, 2
    elif mode == "RL":  # sweeping R to L only requires updating R's
        side, shift1, shift2 = "R", 0, -1
    else:
        raise ValueError(f"Unknown sweep mode {mode!r}")

    gate_tags = list(gate_tens.tags)
    n = int(re.search(r"\d+", gate_tags[3]).group())

    from_key = f"{side}{n + shift1}"
    to_key = f"{side}{n + shift2}"

    transf_tens = cost_tn.select(
        tags=transfer_structure[side][f"{from_key}_to_{to_key}"], which="any"
    )
    new_env = contracted_envs[side][from_key] & transf_tens
    contracted_envs[side][to_key] = new_env.contract()

    return contracted_envs


def optimize_one_gate(
    tag,
    loc_cost_tn,
    cost_tn,
    mode,
    transfer_structure,
    contracted_envs,
    *,
    optimize="auto-hq",
):
    r"""
    Update a single gate tensor via its SVD-optimal unitary approximation.

    Parameters
    ----------
    tag : str
        Tag identifying the gate tensor to update.
    loc_cost_tn : :quimb-api:`TensorNetwork`
        Local cost tensor network for the site the gate belongs to.
    cost_tn : :quimb-api:`TensorNetwork`
        Full cost tensor network.
    mode : {"LR", "RL"}
        Sweep direction, forwarded to :func:`update_contracted_envs`.
    transfer_structure : dict
        Transfer structure dictionary generated by
        :func:`find_transfer_structure`.
    contracted_envs : dict
        Dictionary of contracted environments generated by
        :func:`build_first_sweep`.
    optimize : str or :cotengra-api:`HyperOptimizer`, optional
        Contraction optimization strategy passed to
        :func:`PRC_loc_cost_tn`. Default is ``"auto-hq"``.

    Returns
    -------
    cost_tn : :quimb-api:`TensorNetwork`
        Updated full cost tensor network.
    loc_cost_tn : :quimb-api:`TensorNetwork`
        Updated local cost tensor network.
    contracted_envs : dict
        Updated dictionary of contracted environments.
    overlap : float
        Cost-function value reached after this gate update.
    """

    original_gate_tens = cost_tn.select(tags=tag).tensors[0]
    inds = original_gate_tens.inds

    # contract the local cost to a 4-legged non-unitary tensor
    # and rehearse the contraction for later sweeps
    prc_loc_cost_tens = PRC_loc_cost_tn(loc_cost_tn, tag, optimize)

    # do the SVD, retaining isometries
    new_gate_tens, overlap = svd_optimal_gate_update(
        prc_loc_cost_tens,
        # recall index ordering in Gate class:
        # (OUT_LEFT, OUT_RIGHT, IN_LEFT, IN_RIGHT)
        left_inds=(inds[0], inds[1]),
    )

    # ensure index order
    new_gate_tens.transpose(inds[2], inds[3], inds[0], inds[1])
    new_gate_tens.modify(tags=original_gate_tens.tags)

    # update the (local) cost tensor network
    cost_tn = update_cost_tn(cost_tn, new_gate_tens)
    loc_cost_tn = update_cost_tn(loc_cost_tn, new_gate_tens)

    # update transfer tensors and environments
    contracted_envs = update_contracted_envs(
        mode, original_gate_tens, cost_tn, transfer_structure, contracted_envs
    )

    return cost_tn, loc_cost_tn, contracted_envs, overlap


def sweep_direction(
    mode,
    site_indices,
    cost_tn,
    transfer_structure,
    contracted_envs,
    *,
    optimize="auto-hq",
):
    r"""
    Run one left-to-right or right-to-left sweep over all sites.

    Parameters
    ----------
    mode : {"LR", "RL"}
        Sweep direction, forwarded to :func:`optimize_one_gate`.
    site_indices : sequence of int
        Site indices to sweep over, in order. Its length must be
        ``n_qubits - 2`` (true of both the "LR" and "RL" ranges built by
        :func:`optimize_single_gate_update`), since ``n_qubits`` is
        recovered from it as ``len(site_indices) + 2``.
    cost_tn : :quimb-api:`TensorNetwork`
        Full cost tensor network.
    transfer_structure : dict
        Transfer structure dictionary generated by
        :func:`find_transfer_structure`.
    contracted_envs : dict
        Dictionary of contracted environments generated by
        :func:`build_first_sweep`.
    optimize : str or :cotengra-api:`HyperOptimizer`, optional
        Contraction optimization strategy forwarded to
        :func:`optimize_one_gate`. Default is ``"auto-hq"``.

    Returns
    -------
    cost_tn : :quimb-api:`TensorNetwork`
        Updated full cost tensor network.
    contracted_envs : dict
        Updated dictionary of contracted environments.
    overlap : float
        Cost-function value reached after the last gate update of the sweep.
    """

    n_qubits = len(site_indices) + 2
    overlap = None
    for site_index in site_indices:
        loc_cost_tn, gate_to_opt_tags = build_loc_cost_tn(
            n_qubits, site_index, contracted_envs, cost_tn
        )

        # sweep through gate list (along column, from lower to higher depths)
        for tag in gate_to_opt_tags:
            cost_tn, loc_cost_tn, contracted_envs, overlap = optimize_one_gate(
                tag,
                loc_cost_tn,
                cost_tn,
                mode,
                transfer_structure,
                contracted_envs,
                optimize=optimize,
            )

    return cost_tn, contracted_envs, overlap


def optimize_single_gate_update(
    n_qubits,
    cost_tn,
    rtol,
    n_sweeps_max,
    transfer_structure,
    contracted_envs,
    *,
    optimize="auto-hq",
):
    r"""
    Optimize a TN ansatz circuit using sequential single-gate updates.

    This function performs alternating left-to-right and right-to-left
    sweeps over the variational gates of the cost tensor network. Each gate
    is optimized individually by contracting its effective environment,
    performing a singular value decomposition (SVD), and projecting the
    result back onto the unitary manifold.

    The optimization iteratively updates both the tensor network and the
    cached contracted environments.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (sites) in the tensor network.
    cost_tn : :quimb-api:`TensorNetwork`
        Full cost tensor network containing the variational circuit and
        target MPO.
    rtol : float
        Relative tolerance condition for stopping.
    n_sweeps_max : int
        Maximum number of optimization sweeps.
    transfer_structure : dict
        Transfer structure dictionary generated by
        :func:`find_transfer_structure`.
    contracted_envs : dict
        Dictionary of contracted environments generated by
        :func:`build_first_sweep`.
    optimize : str or :cotengra-api:`HyperOptimizer`, optional
        Contraction optimization strategy passed to
        :meth:`TensorNetwork.contract` for every local environment
        contraction. Default is ``"auto-hq"``; pass e.g. ``"greedy"`` for a
        cheaper, deterministic strategy, or a
        :cotengra-api:`ReusableHyperOptimizer` instance for finer control.

    Returns
    -------
    cost_tn : :quimb-api:`TensorNetwork`
        Optimized cost tensor network.
    contracted_envs : dict
        Updated dictionary of contracted environments.

    Notes
    -----
    The optimization proceeds as follows:

    1. Construct a local effective tensor network around a gate,
    2. Remove the gate tensor and contract the surrounding environment,
    3. Perform an SVD of the resulting effective tensor,
    4. Reconstruct the optimal unitary gate from the isometric factors,
    5. Update the tensor network and cached environments.

    The effective local contraction is computed using
    :func:`PRC_loc_cost_tn`.

    The gate update is obtained from an SVD decomposition.

    The overlap displayed in the progress bar corresponds to the
    cost-function value and can be verified independently from the singular
    values of the effective environment tensor.

    The sweeping schedule alternates between:

    - left-to-right (``"LR"``),
    - right-to-left (``"RL"``),

    in order to iteratively improve all variational gates.
    """

    overlap = None
    trange_counter = tqdm(range(n_sweeps_max))
    for _ in trange_counter:
        cost_tn, contracted_envs, _ = sweep_direction(
            "LR",
            range(n_qubits - 2),
            cost_tn,
            transfer_structure,
            contracted_envs,
            optimize=optimize,
        )
        cost_tn, contracted_envs, new_overlap = sweep_direction(
            "RL",
            range(n_qubits - 1, 1, -1),
            cost_tn,
            transfer_structure,
            contracted_envs,
            optimize=optimize,
        )
        trange_counter.set_description(f"overlap: {(new_overlap / 2**n_qubits):.8f}")

        if overlap is not None:
            rel_change = abs(new_overlap - overlap) / abs(overlap)

            if rel_change < rtol:
                break

        overlap = new_overlap

    return cost_tn, contracted_envs, overlap / 2**n_qubits


def transpile_mpo_to_circuit(
    ref_mpo,
    depth,
    rtol,
    n_sweeps_max,
    *,
    param_scaling=1e-1,
    closed=False,
    rng=None,
    optimize="auto-hq",
):
    r"""
    Fit a brickwall circuit ansatz to a reference MPO by single-gate sweeps.

    This is a convenience wrapper chaining :func:`init_cost_tn`,
    :func:`find_transfer_structure`, :func:`build_first_sweep` and
    :func:`optimize_single_gate_update`.

    Parameters
    ----------
    ref_mpo : :quimb-api:`MatrixProductOperator`
        Target MPO to be reproduced by the Ansatz.
    depth : int
        Depth of the brickwall circuit ansatz (even and odd count as 1).
    rtol : float
        Relative tolerance condition for stopping.
    n_sweeps_max : int
        Maximum number of optimization sweeps.
    param_scaling : float, optional
        Scale of the random initialization parameters for the gates.
        Smaller values initialize the circuit closer to the identity.
        Default is ``1e-1``.
    closed : bool, optional
        If ``False`` (default), the bra indices of the MPO remain open.
        If ``True``, ket indices are contracted (trace contraction).
    rng : :numpy-random:`numpy.random.Generator <generator>`, optional
        Random number generator to generate gate parameters.
        If ``None``, a default generator is created.
    optimize : str or :cotengra-api:`HyperOptimizer`, optional
        Contraction optimization strategy forwarded to
        :func:`optimize_single_gate_update`. Default is ``"auto-hq"``; pass
        e.g. ``"greedy"`` for a cheaper, deterministic strategy.

    Returns
    -------
    cost_tn : :quimb-api:`TensorNetwork`
        Optimized cost tensor network.
    contracted_envs : dict
        Updated dictionary of contracted environments.
    """

    n_qubits = ref_mpo.num_tensors
    cost_tn = init_cost_tn(
        ref_mpo, depth, param_scaling=param_scaling, closed=closed, rng=rng
    )
    transfer_structure = find_transfer_structure(n_qubits, cost_tn)
    contracted_envs = build_first_sweep(
        n_qubits, cost_tn, transfer_structure, drop_tags=True
    )

    return optimize_single_gate_update(
        n_qubits,
        cost_tn,
        rtol,
        n_sweeps_max,
        transfer_structure,
        contracted_envs,
        optimize=optimize,
    )
