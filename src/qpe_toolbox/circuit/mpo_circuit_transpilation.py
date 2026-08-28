import re

import numpy as np
import quimb as qu
import quimb.tensor as qtn
from tqdm import tqdm

from qpe_toolbox.circuit.parametrized_circuits import (
    generate_brickwall_circuit,
)


# --------------------------------------------------------------------------
def exp_Pauli_string_as_MPO(term, dt, n_qubits):
    r"""
    Construct the MPO representation of the unitary exponential of a Pauli string.

    Given a Hamiltonian term of the form:

        H = c * P

    where ``c`` is a scalar coefficient and ``P`` is a tensor product of Pauli
    operators acting on a subset of qubits, this function builds the Matrix
    Product Operator (MPO) corresponding to:

        exp(i * dt * c * P)

    using the identity:

        exp(i α P) = cos(α) I + i sin(α) P

    where ``P^2 = I``.

    The Pauli string is expanded to the full system size by inserting identity
    operators on inactive qubits.

    Parameters
    ----------
    term : tuple
        A tuple ``(coeff, pauli_string, active_qubits)`` where:

        - ``coeff`` (float): Scalar coefficient multiplying the Pauli string.
        - ``pauli_string`` (str): String of Pauli operators (e.g. ``"ZYXXZ"``).
        - ``active_qubits`` (list[int]): Indices of qubits where the Pauli
          operators act. The length must match ``pauli_string``.

    dt : float
        Evolution parameter (e.g. time or rotation angle).

    n_qubits : int
        Total number of qubits in the system.

    Returns
    -------
    qtn.MatrixProductOperator in lrud format
        MPO representing the operator:

            exp(i * dt * coeff * P)

        where ``P`` is the full Pauli string embedded in the ``n_qubits`` system.

    Raises
    ------
    ValueError
        If the length of ``pauli_string`` does not match the number of
        ``active_qubits``.
    """

    string_coeff, pauli_string, active_qubits = term
    id4 = qu.identity(2).reshape(1, 1, 2, 2)

    pauli_string_tensors = []
    pauli_weight = 0
    for qubit in range(n_qubits):
        if qubit in active_qubits:
            pauli_string_tensors.append(
                qu.pauli(pauli_string[pauli_weight]).reshape(1, 1, 2, 2)
            )
            pauli_weight += 1
        else:
            pauli_string_tensors.append(id4)

    # Fix the legs on the edges
    pauli_string_tensors[0] = pauli_string_tensors[0].reshape(1, 2, 2)
    pauli_string_tensors[-1] = pauli_string_tensors[-1].reshape(1, 2, 2)
    string_mpo = qtn.MatrixProductOperator(arrays=pauli_string_tensors)
    id_mpo = string_mpo.identity()

    id_mpo[0] *= np.cos(dt * string_coeff)
    string_mpo[0] *= 1j * np.sin(dt * string_coeff)

    exp_pauli_string_mpo = id_mpo.add_MPO(string_mpo)
    exp_pauli_string_mpo.compress(cutoff=1e-6, max_bond=2)

    return exp_pauli_string_mpo


def trotter1_approx_as_MPO(
    hamiltonian, dt, *, cutoff=1e-10, max_bond=None, reverse_order=False
):
    r"""
    Construct the first-order Trotter-Suzuki approximation as a Matrix Product Operator (MPO).

    This function builds an MPO representation of the first-order product
    formula for the time-evolution operator

    .. math::

        U(dt) \approx \prod_j e^{-i dt \, H_j},

    where ``hamiltonian.terms = [H_0, H_1, ..., H_{m-1}]`` is a decomposition
    of the Hamiltonian into terms that can each be exponentiated individually
    as MPOs.

    Each exponential factor is generated with
    :func:`exp_Pauli_string_as_MPO`, and factors are successively multiplied
    together using MPO compression.

    Parameters
    ----------
    hamiltonian : :class:`~src.hamiltonian.hamiltonian.Hamiltonian`
        Includes Pauli strings, positions and couplings.
    dt : float
        Time step used in the Trotter approximation.
    cutoff : float, optional
        Singular value truncation threshold used during MPO compression.
        Default is ``1e-10``.
    max_bond : int, optional
        Maximum allowed bond dimension during MPO compression.
        Default is ``None`` (no limit).
    reverse_order : bool, optional
        If ``False`` (default), terms are applied in forward order.
        after the first term. If ``True``, terms are applied in reverse index order.

    Returns
    -------
    MPO
        MPO representation of the first-order Trotter approximation.
    """

    ham_terms = hamiltonian.terms
    n_qubits = hamiltonian.n_qubits

    if reverse_order:
        init_term = len(ham_terms) - 1
        trange_counter = tqdm(reversed(range(len(ham_terms) - 1)))
    else:
        init_term = 0
        trange_counter = tqdm(range(1, len(ham_terms)))

    U_trotter1_mpo = exp_Pauli_string_as_MPO(ham_terms[init_term], -dt, n_qubits)
    for i in trange_counter:
        new_factor_mpo = exp_Pauli_string_as_MPO(ham_terms[i], -dt, n_qubits)
        U_trotter1_mpo = U_trotter1_mpo.apply(
            new_factor_mpo, compress=True, cutoff=cutoff, max_bond=max_bond
        )
        trange_counter.set_description(
            f"{'': <4}Bond dimension (Trotter 1): {U_trotter1_mpo.max_bond()}"
        )

    return U_trotter1_mpo


def trotter2_approx_as_MPO(
    hamiltonian, dt, *, cutoff=1e-10, max_bond=None, verbosity=0
):
    r"""
    Construct the second-order symmetric Trotter-Suzuki approximation as an MPO.

    This function builds the second-order product formula

    .. math::

        U(dt) \approx U_1(dt/2)\,U_1^{\mathrm{rev}}(dt/2),

    where :math:`U_1` is the first-order Trotter approximation and
    :math:`U_1^{\mathrm{rev}}` uses the reverse operator ordering.

    The approximation is accurate to second order in ``dt``.

    Parameters
    ----------
    hamiltonian : :class:`~src.hamiltonian.hamiltonian.Hamiltonian`
        Includes Pauli strings, positions and couplings.
    dt : float or complex
        Time step used in the Trotter approximation.
    cutoff : float, optional
        Singular value truncation threshold used during MPO compression.
        Default is ``1e-10``.
    max_bond : int, optional
        Maximum allowed bond dimension during MPO compression.
        Default is ``None`` (no limit).
    verbosity : int, default ``0``
        Verbosity level. If >= 1, print progress.

    Returns
    -------
    MPO
        MPO representation of the second-order Trotter approximation.
    """

    if verbosity >= 1:
        print(f"{'': <2}Building 2nd order Trotter")
        print(f"{'': <4}Building 1st order Trotter (1st half)")

    layer1_mpo = trotter1_approx_as_MPO(
        hamiltonian, dt / 2, cutoff=cutoff, max_bond=max_bond
    )

    if verbosity >= 1:
        print(rf"{'': <4}Building 1st order Trotter (2nd half)")

    layer2_mpo = trotter1_approx_as_MPO(
        hamiltonian, dt / 2, cutoff=cutoff, max_bond=max_bond, reverse_order=True
    )
    return layer1_mpo.apply(layer2_mpo, compress=True, cutoff=cutoff, max_bond=max_bond)


def trotter4_approx_as_MPO(
    hamiltonian, dt, *, cutoff=1e-10, max_bond=None, verbosity=0
):
    r"""
    Construct the fourth-order Trotter-Suzuki approximation as an MPO.

    This function implements the standard symmetric fourth-order composition

    .. math::

        U_4(dt) =
        U_2(s\,dt)\,
        U_2((1 - 2s)\,dt)\,
        U_2(s\,dt),

    where :math:`U_2` is the second-order Trotter approximation and

    .. math::

        s = \frac{1}{2 - 2^{1/3}}

    is the symmetry factor.

    This approximation is accurate to fourth order in ``dt``.

    Parameters
    ----------
    hamiltonian : :class:`~src.hamiltonian.hamiltonian.Hamiltonian`
        Includes Pauli strings, positions and couplings.
    dt : float
        Time step used in the Trotter approximation.
    cutoff : float, optional
        Singular value truncation threshold used during MPO compression.
        Default is ``1e-10``.
    max_bond : int, optional
        Maximum allowed bond dimension during MPO compression.
        Default is ``None`` (no limit).
    verbosity : int, default ``0``
        Verbosity level. If >= 1, print progress.

    Returns
    -------
    MPO
        MPO representation of the fourth-order Trotter approximation.
    """

    sym_factor = 1.0 / (2.0 - 2 ** (1.0 / 3.0))

    if verbosity >= 1:
        print("Building 4th order Trotter")
        print(rf"{'': <2}Building 2nd order Trotter (1st and 3rd layers)")

    layer1_3_mpo = trotter2_approx_as_MPO(
        hamiltonian,
        dt * sym_factor,
        cutoff=cutoff,
        max_bond=max_bond,
        verbosity=verbosity,
    )

    if verbosity >= 1:
        print(rf"{'': <2}Building 2nd order Trotter (2nd layer)")

    layer2_mpo = trotter2_approx_as_MPO(
        hamiltonian,
        dt * (1 - 2 * sym_factor),
        cutoff=cutoff,
        max_bond=max_bond,
        verbosity=verbosity,
    )

    if verbosity >= 1:
        print(f"{'': <2}Multiplying the 3 MPO layers")

    U_trotter4_mpo = layer1_3_mpo.apply(
        layer2_mpo, compress=True, cutoff=cutoff, max_bond=max_bond
    )
    U_trotter4_mpo = U_trotter4_mpo.apply(
        layer1_3_mpo, compress=True, cutoff=cutoff, max_bond=max_bond
    )
    if verbosity == 1:
        print(f"{'': <4}Final bond dimension:", U_trotter4_mpo.max_bond())

    return U_trotter4_mpo


def trotter_approx_as_MPO(
    hamiltonian, dt, *, trotter_order=1, cutoff=1e-10, max_bond=None, verbosity=0
):
    r"""
    Construct a Trotter-Suzuki approximation of a Hamiltonian evolution operator as an MPO.

    This function dispatches to a specific Trotter-Suzuki decomposition
    according to the requested approximation order. The Hamiltonian is assumed
    to provide a decomposition into elementary terms through ``hamiltonian.terms``.

    Depending on ``trotter_order``, the approximation is built using:

    .. math::

        U(dt) \approx
        \begin{cases}
            \text{1st-order product formula}, & \text{if trotter\_order} = 1, \\
            \text{2nd-order symmetric formula}, & \text{if trotter\_order} = 2, \\
            \text{4th-order Suzuki formula}, & \text{if trotter\_order} = 4.
        \end{cases}

    The resulting operator is returned as a compressed MPO.

    Parameters
    ----------
    hamiltonian : :class:`~src.hamiltonian.hamiltonian.Hamiltonian`
        Includes Pauli strings, positions and couplings.
    dt : float
        Time step used in the Trotter approximation.
    trotter_order : {1, 2, 4}, optional
        Order of the Trotter-Suzuki decomposition. Default is ``1``.
    cutoff : float, optional
        Singular value truncation threshold used during MPO compression.
        Default is ``1e-10``.
    max_bond : int, optional
        Maximum allowed MPO bond dimension during compression.
        Default is ``None`` (no limit).
    verbosity : int, default ``0``
        Verbosity level. If >= 1, print progress.

    Returns
    -------
    MPO
        MPO representation of the Trotterized time-evolution operator.

    Raises
    ------
    ValueError
        If the requested ``trotter_order`` is not implemented.
    """
    if trotter_order == 1:
        return trotter1_approx_as_MPO(hamiltonian, dt, cutoff=cutoff, max_bond=max_bond)

    if trotter_order == 2:
        return trotter2_approx_as_MPO(
            hamiltonian, dt, cutoff=cutoff, max_bond=max_bond, verbosity=verbosity
        )

    if trotter_order == 4:
        return trotter4_approx_as_MPO(
            hamiltonian, dt, cutoff=cutoff, max_bond=max_bond, verbosity=verbosity
        )
    raise ValueError(f"Order {trotter_order} not implemented")


def state_preparation_mpo(state_mps):
    r"""
    Perform outer product between an MPS and the state 0.

    Parameters
    ----------
    state_mps : :quimb-api:`MatrixProductState`
        Target MPS to be reproduced by some circuit Ansatz.

    Returns
    -------
    :quimb-api:`TensorNetwork`
        Tensor network containing both the reference MPO
        for some variational procedure.
    """

    n_qubits = state_mps.num_tensors
    ket0 = np.array([2.0, 0])  # normalization of the cost by 2**n_qubits

    # DMRG2's output orders boundary-tensor legs as (phys, bond) at the first
    # site and (bond, phys) at the last site
    first_array = state_mps.tensors[0].data
    arrays = [
        np.outer(first_array, ket0)
        .reshape(2, first_array.shape[1], 2)
        .transpose(1, 0, 2)  # transpose to lpp'
    ]

    for i in range(1, n_qubits - 1):
        array = state_mps.tensors[i].data
        dims = np.shape(array)
        # assume array had order lpr then get lprp' so transpose to lrpp'
        arrays.append(
            np.outer(array, ket0).reshape(dims[0], 2, dims[2], 2).transpose(0, 2, 1, 3)
        )

    last_array = state_mps.tensors[n_qubits - 1].data
    # assume array had order rp then get rpp' so no transpose
    arrays.append(np.outer(last_array, ket0).reshape(last_array.shape[0], 2, 2))

    return qtn.MatrixProductOperator(arrays=arrays)


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
    TN = qtn.TensorNetwork()

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

    bw_unitary_tn = bw_circ.get_uni()

    TN = TN & bw_unitary_tn

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

        TN = TN & reind_tens

    return TN


def get_envs_tns(n_qubits, x, cost_tn):
    r"""
    Extract the left and right environment TNs associated to a given site.

    This function removes the tensors tagged with ``I{x}`` from the full
    cost tensor network and partitions the remaining network into left and/or
    right environments relative to site ``x``.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (sites) in the tensor network.
    x : int
        Site index for which the environments are constructed.
    cost_tn : :quimb-api:`TensorNetwork`
        Full cost tensor network.

    Returns
    -------
    list of :quimb-api:`TensorNetwork`
        List containing the environment tensor networks.

        - If ``x == 0``, only the right environment is returned.
        - If ``x == n_qubits - 1``, only the left environment is returned.
        - Otherwise, the list is ordered as ``[left_env, right_env]``.

    Notes
    -----
    The environments are obtained by selecting tensors according to their
    ``I{i}`` tags:

    - left environment: tensors tagged with ``I0`` through ``I{x-1}``,
    - right environment: tensors tagged with ``I{x+1}`` through
      ``I{n_qubits-1}``.
    """

    env_tn = cost_tn.select(tags=[f"I{x}"], which="!any")

    left_tags = [f"I{x}" for x in range(x)]
    right_tags = [f"I{x}" for x in range(x + 1, n_qubits)]

    env_tns = []
    if x > 0:
        env_tns.append(env_tn.select(tags=left_tags, which="any"))
    if x < n_qubits - 1:
        env_tns.append(env_tn.select(tags=right_tags, which="any"))

    return env_tns


def find_transfer_structure(n_qubits, cost_tn):
    r"""
    Determine the transfer structures connecting neighboring environments in a cost TN.

    This function analyzes the left and right environment tensor networks
    associated with each site and identifies the tensor tags involved in
    transferring contractions between neighboring environments.

    The resulting transfer structure is organized into dictionaries for
    left-to-right and right-to-left propagation.

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

    Notes
    -----
    For each neighboring pair of environments, tensors are selected according
    to their ``I{i}`` tags, and only tags beginning with ``"G"`` or ``"U"``
    are retained in the transfer description.

    The transfer structures can be used to optimize sequential contraction
    strategies or environment updates in tensor-network algorithms.
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
        transf_tags = transf_tn.tags
        filtered_tags = []
        for tag in transf_tags:
            if tag[0] == "G" or tag[0] == "U":
                filtered_tags.append(tag)
        transfer_structure["L"][f"L{x}_to_L{x + 1}"] = filtered_tags

        # right transfer
        transf_tn = right_uncontracted[f"I{n_qubits - 2 - x}"].select(
            tags=(f"I{n_qubits - 1 - x}"), which="any"
        )
        transf_tags = transf_tn.tags
        filtered_tags = []
        for tag in transf_tags:
            if tag[0] == "G" or tag[0] == "U":
                filtered_tags.append(tag)
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

    The resulting environments are stored as contracted tensors labeled
    ``L{i}`` and ``R{i}``, corresponding to left and right effective
    environments at each site.

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

    Notes
    -----
    The contraction procedure starts from the MPO edge tensors tagged
    ``"Uref0"`` and ``"Uref{n_qubits-1}"`` and recursively absorbs the
    transfer tensor networks specified in ``transfer_structure``.

    These environments are typically reused during local optimization sweeps
    to avoid repeated large tensor contractions.
    """

    contracted_envs = {"L": {}, "R": {}}

    # the first environments are L{1} and R{n_qubits-1}, which are the edge tensors on the MPO
    L_init = cost_tn.select(tags="Uref0", which="all").tensors[0]
    L_init.add_tag(tag=("L1"))
    R_init = cost_tn.select(tags=f"Uref{n_qubits - 1}", which="all").tensors[0]
    R_init.add_tag(tag=(f"R{n_qubits - 2}"))

    contracted_envs["L"]["L1"] = L_init
    contracted_envs["R"][f"R{n_qubits - 2}"] = R_init
    L_next = L_init.copy(deep=True)
    R_next = R_init.copy(deep=True)

    for counter, transf_tags in enumerate(
        zip(
            transfer_structure["L"].values(),
            transfer_structure["R"].values(),
            strict=True,
        )
    ):
        L_next = L_next.copy(deep=True)
        L_transf_tn = cost_tn.select(tags=transf_tags[0], which="any")
        L_next = L_next & L_transf_tn
        L_next = qtn.tensor_contract(*L_next.tensors, drop_tags=drop_tags)
        L_next.add_tag(tag=(f"L{counter + 2}"))

        contracted_envs["L"][f"L{counter + 2}"] = L_next

        R_next = R_next.copy(deep=True)
        R_transf_tn = cost_tn.select(tags=transf_tags[1], which="any")
        R_next = R_next & R_transf_tn
        R_next = qtn.tensor_contract(*R_next.tensors, drop_tags=drop_tags)
        R_next.add_tag(tag=(f"R{n_qubits - 3 - counter}"))

        contracted_envs["R"][f"R{n_qubits - 3 - counter}"] = R_next

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
    tags_to_opt = gates_to_opt.tags

    filtered_tags = []
    for tag in tags_to_opt:
        if tag[0] == "G":
            filtered_tags.append(tag)

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


def update_cost_tn(cost_tn, gate_tensors):
    r"""
    Update a cost tensor network with newly optimized gate tensors.

    This function replaces existing gate tensors in the cost tensor network
    with updated optimized tensors having the same gate tags.

    Parameters
    ----------
    cost_tn : :quimb-api:`TensorNetwork`
        Full cost tensor network.
    gate_tensors : sequence of :quimb-api:`Tensor`
        Sequence of optimized gate tensors to insert into the network.

    Returns
    -------
    :quimb-api:`TensorNetwork`
        Updated cost tensor network containing the optimized gate tensors.
    """

    for tens in gate_tensors:
        tag_tens = next(
            iter(tens.tags)
        )  # list(tens.tags)[0]  # the first tag is "GATE_{n}" by construction
        cost_tn.delete(tags=tag_tens)  # delete the old tensor with same tags
        cost_tn = cost_tn & tens  # add the new tensor

    return cost_tn


def update_contracted_envs(
    mode, gate_tensors, cost_tn, transfer_structure, contracted_envs
):
    r"""
    Update contracted environments after local gate optimization.

    This function updates the cached contracted left or right environments
    affected by newly optimized gate tensors during a sweeping optimization
    procedure.

    Depending on the sweep direction, only the environments influenced by
    the updated gates are recomputed.

    Parameters
    ----------
    mode : {"LR", "RL"}
        Sweep direction.

        - ``"LR"`` : left-to-right sweep,
        - ``"RL"`` : right-to-left sweep.
    gate_tensors : sequence of :quimb-api:`Tensor`
        Sequence of optimized gate tensors.
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
    Each optimized gate acts on neighboring sites ``I{n}`` and ``I{n+1}``.
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

    for tens in gate_tensors:
        gate_tags = list(tens.tags)
        n = int(re.search(r"\d+", gate_tags[3]).group())

        from_key = f"{side}{n + shift1}"
        to_key = f"{side}{n + shift2}"

        transf_tens = cost_tn.select(
            tags=transfer_structure[side][f"{from_key}_to_{to_key}"], which="any"
        )
        new_env = contracted_envs[side][from_key].copy(deep=True) & transf_tens
        contracted_envs[side][to_key] = new_env.contract()

    return contracted_envs


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

    instr = [
        ("LR", list(range(n_qubits - 2))),
        ("RL", list(reversed(range(2, n_qubits)))),
    ]

    prev_overlap = None

    trange_counter = tqdm(range(n_sweeps_max))
    for _ in trange_counter:
        for mode, site_indices in instr:
            for site_index in site_indices:
                # build local cost function for all gates at "site_index"
                loc_cost_tn, gate_to_opt_tags = build_loc_cost_tn(
                    n_qubits, site_index, contracted_envs, cost_tn
                )

                # sweep through gate list (along column, from lower to higher depths)
                for tag in gate_to_opt_tags:
                    original_gate_tens = cost_tn.select(tags=tag).tensors[0]
                    inds = original_gate_tens.inds

                    # contract the local cost to a 4-legged non-unitary tensor
                    # and rehearse the contraction for later sweeps
                    prc_loc_cost_tens = PRC_loc_cost_tn(
                        loc_cost_tn=loc_cost_tn,
                        tags=tag,
                        optimize=optimize,
                    )

                    # do the SVD
                    prc_loc_cost_UsVh = qtn.tensor_split(
                        T=prc_loc_cost_tens,
                        # recall index ordering in Gate class:
                        # (OUT_LEFT, OUT_RIGHT, IN_LEFT, IN_RIGHT)
                        left_inds=(inds[0], inds[1]),
                        method="svd",
                        absorb=None,
                    )

                    # retain isometries
                    overlap = np.sum(prc_loc_cost_UsVh.tensors[1].data)
                    new_overlap = overlap
                    trange_counter.set_description(
                        f"overlap: {(overlap / pow(2, n_qubits)):.8f}"
                    )
                    new_gate_tens = (
                        prc_loc_cost_UsVh.tensors[0].conj()
                        & prc_loc_cost_UsVh.tensors[2].conj()
                    ) ^ ...

                    # ensure index order
                    new_gate_tens.transpose(inds[2], inds[3], inds[0], inds[1])

                    new_gate_tens.modify(tags=original_gate_tens.tags)

                    # update the (local) cost tensor network
                    cost_tn = update_cost_tn(cost_tn, gate_tensors=[new_gate_tens])
                    loc_cost_tn = update_cost_tn(
                        loc_cost_tn, gate_tensors=[new_gate_tens]
                    )

                    # update transfer tensors and environments
                    contracted_envs = update_contracted_envs(
                        mode,
                        [original_gate_tens],
                        cost_tn,
                        transfer_structure,
                        contracted_envs,
                    )
        if prev_overlap is not None:
            rel_change = abs(new_overlap - prev_overlap) / abs(prev_overlap)

            if rel_change < rtol:
                break

        prev_overlap = new_overlap

    return cost_tn, contracted_envs


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
