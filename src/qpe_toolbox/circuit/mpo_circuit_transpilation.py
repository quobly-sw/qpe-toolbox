import re

import cotengra as ctg
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_core import (
    TensorNetwork,
    get_tags,
    tensor_contract,
    tensor_split,
)
from tqdm import tqdm

from qpe_toolbox.circuit.parametrized_circuits import (
    one_qubit_layer,
    two_qubit_nn_layer,
)  # do PR to generalize nn_layer so just import that function

list_paulis = ["I", "X", "Y", "Z"]


# --------------------------------------------------------------------------
def exp_Pauli_string_as_MPO(ham_term, n_qubits, *, theta):
    r"""
    Construct the MPO representation of the unitary exponential of a Pauli string.

    Given a Hamiltonian term of the form:

        H = c * P

    where ``c`` is a scalar coefficient and ``P`` is a tensor product of Pauli
    operators acting on a subset of qubits, this function builds the Matrix
    Product Operator (MPO) corresponding to:

        exp(i * theta * c * P)

    using the identity:

        exp(i α P) = cos(α) I + i sin(α) P

    where ``P^2 = I``.

    The Pauli string is expanded to the full system size by inserting identity
    operators on inactive qubits.

    Parameters
    ----------
    ham_term : tuple
        A tuple ``(coeff, pauli_string, active_qubits)`` where:

        - ``coeff`` (float): Scalar coefficient multiplying the Pauli string.
        - ``pauli_string`` (str): String of Pauli operators (e.g. ``"ZYXXZ"``).
        - ``active_qubits`` (list[int]): Indices of qubits where the Pauli
          operators act. The length must match ``pauli_string``.

    n_qubits : int
        Total number of qubits in the system.

    theta : float
        Evolution parameter (e.g. time or rotation angle).

    Returns
    -------
    qtn.MatrixProductOperator
        MPO representing the operator:

            exp(i * theta * coeff * P)

        where ``P`` is the full Pauli string embedded in the ``n_qubits`` system.

    Notes
    -----
    - The MPO is constructed in left-right-up-down index ordering.
    - The Pauli string MPO has bond dimension 1 before summation.
    - After combining the identity and Pauli MPOs, the result should not
      exceed a maximum bond dimension of 2.

    Raises
    ------
    ValueError
        If the length of ``pauli_string`` does not match the number of
        ``active_qubits``.

    Examples
    --------
    >>> ham_term = (-0.345, "ZYXXZ", [0, 2, 3, 6, 9])
    >>> mpo = exp_Pauli_string_as_MPO(ham_term, n_qubits=10, theta=0.1)
    >>> mpo
    <MatrixProductOperator ...>
    """

    string_coeff = ham_term[0]
    pauli_string = ham_term[1]
    active_qubits = ham_term[2]
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

    id_mpo[0] *= np.cos(theta * string_coeff)
    string_mpo[0] *= 1j * np.sin(theta * string_coeff)

    exp_pauli_string_mpo = id_mpo.add_MPO(string_mpo)
    exp_pauli_string_mpo.compress(cutoff=1e-6, max_bond=2)

    return exp_pauli_string_mpo


def trotter1_approx_as_MPO(
    ham_terms, n_qubits, *, dt, cutoff, max_bond, reverse_order=False
):
    r"""
    Construct the first-order Trotter-Suzuki approximation as a Matrix
    Product Operator (MPO).

    This function builds an MPO representation of the first-order product
    formula for the time-evolution operator

    .. math::

        U(dt) \approx \prod_j e^{-i dt \, H_j},

    where ``ham_terms = [H_0, H_1, ..., H_{m-1}]`` is a decomposition of the
    Hamiltonian into terms that can each be exponentiated individually as MPOs.

    Each exponential factor is generated with
    :func:`exp_Pauli_string_as_MPO`, and factors are successively multiplied
    together using MPO compression.

    Parameters
    ----------
    ham_terms : sequence
        Sequence of Hamiltonian terms. Each element must be compatible with
        :func:`exp_Pauli_string_as_MPO`.
    n_qubits : int
        Number of qubits (sites) in the system.
    dt : float
        Time step used in the Trotter approximation.
    cutoff : float
        Singular value truncation threshold used during MPO compression.
    max_bond : int
        Maximum allowed bond dimension during MPO compression.
    reverse_order : bool, optional
        If ``False`` (default), terms are applied in forward order.
        after the first term. If ``True``, terms are applied in reverse index order.

    Returns
    -------
    MPO
        MPO representation of the first-order Trotter approximation.

    Notes
    -----
    The resulting MPO is built iteratively using compressed MPO products via
    ``apply(..., compress=True)``. Truncation errors may accumulate depending
    on ``cutoff`` and ``max_bond``.

    See Also
    --------
    trotter2_approx_as_MPO
    trotter4_approx_as_MPO
    exp_Pauli_string_as_MPO
    """

    if reverse_order:
        init_term = len(ham_terms) - 1
        trange_counter = tqdm(list(reversed(range(len(ham_terms) - 1))))
    else:
        init_term = 0
        trange_counter = tqdm(list(range(1, len(ham_terms))))

    U_trotter1_mpo = exp_Pauli_string_as_MPO(ham_terms[init_term], n_qubits, theta=-dt)
    for i in trange_counter:
        new_factor_mpo = exp_Pauli_string_as_MPO(ham_terms[i], n_qubits, theta=-dt)
        U_trotter1_mpo = U_trotter1_mpo.apply(
            new_factor_mpo, compress=True, cutoff=cutoff, max_bond=max_bond
        )
        trange_counter.set_description(
            f"{'': <8}bond dimension: {U_trotter1_mpo.max_bond()}"
        )

    return U_trotter1_mpo


def trotter2_approx_as_MPO(ham_terms, n_qubits, *, dt, cutoff, max_bond, verbosity=0):
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
    ham_terms : sequence
        Sequence of Hamiltonian terms. Each term must be compatible with
        :func:`exp_Pauli_string_as_MPO`.
    n_qubits : int
        Number of qubits (sites) in the system.
    dt : float or complex
        Time step used in the Trotter approximation.
    cutoff : float
        Singular value truncation threshold used during MPO compression.
    max_bond : int
        Maximum allowed bond dimension during MPO compression.
    verbosity : int, optional
        If set to ``1``, prints progress information. Default is ``0``.

    Returns
    -------
    MPO
        MPO representation of the second-order Trotter approximation.

    Notes
    -----
    Two first-order MPO approximants with half time step are constructed and
    then multiplied together using compression.

    See Also
    --------
    trotter1_approx_as_MPO
    trotter4_approx_as_MPO
    """

    if verbosity == 1:
        print(
            f"{'': <4}The 2nd order Trotter approximant consists on two subsequent 1st order approximants:",
            "\n",
        )
        print(
            f"{'': <8}Computing the first 1st order approximant for a time step {{0.5}}dt"
        )

    layer1_mpo = trotter1_approx_as_MPO(
        ham_terms,
        n_qubits,
        dt=dt / 2,
        cutoff=cutoff,
        max_bond=max_bond,
    )

    if verbosity == 1:
        print("\n")
        print(
            rf"{'': <8}Computing the second 1st order approximant for a time step {{0.5}}dt"
        )

    layer2_mpo = trotter1_approx_as_MPO(
        ham_terms,
        n_qubits,
        dt=dt / 2,
        cutoff=cutoff,
        max_bond=max_bond,
        reverse_order=True,
    )

    if verbosity == 1:
        print("\n")

    return layer1_mpo.apply(layer2_mpo, compress=True, cutoff=cutoff, max_bond=max_bond)


def trotter4_approx_as_MPO(ham_terms, n_qubits, *, dt, cutoff, max_bond, verbosity=0):
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
    ham_terms : sequence
        Sequence of Hamiltonian terms. Each term must be compatible with
        :func:`exp_Pauli_string_as_MPO`.
    n_qubits : int
        Number of qubits (sites) in the system.
    dt : float
        Time step used in the Trotter approximation.
    cutoff : float
        Singular value truncation threshold used during MPO compression.
    max_bond : int
        Maximum allowed bond dimension during MPO compression.
    verbosity : int, optional
        If set to ``1``, prints progress information. Default is ``0``.

    Returns
    -------
    MPO
        MPO representation of the fourth-order Trotter approximation.

    Notes
    -----
    The method constructs three second-order MPO approximants and combines
    them through compressed MPO multiplication. Intermediate bond dimensions
    may grow significantly depending on the system and truncation parameters.

    See Also
    --------
    trotter1_approx_as_MPO
    trotter2_approx_as_MPO
    """

    sym_factor = 1.0 / (2.0 - 2 ** (1.0 / 3.0))

    if verbosity == 1:
        print(
            "The 4th order Trotter approximant consists on three subsequent 2nd order approximants:",
            "\n",
        )
        print(
            rf"{'': <4}Computing the first 2nd order approximant for a time step s$_{{sym}}$dt"
        )
        # display(Math(r'\quad \text{Computing the second 2nd order approximant for a time step } (1-2s_{\mathrm{sym}})\,dt'))

    layer1_3_mpo = trotter2_approx_as_MPO(
        ham_terms,
        n_qubits,
        dt=dt * sym_factor,
        cutoff=cutoff,
        max_bond=max_bond,
        verbosity=verbosity,
    )

    if verbosity == 1:
        print(rf"{'': <4}Computing U_2(s\,dt)\$")

    layer2_mpo = trotter2_approx_as_MPO(
        ham_terms,
        n_qubits,
        dt=dt * (1 - 2 * sym_factor),
        cutoff=cutoff,
        max_bond=max_bond,
        verbosity=verbosity,
    )

    if verbosity == 1:
        print(f"{'': <4}Multiplying first and second MPOs", "\n")

    U_trotter4_mpo = layer1_3_mpo.apply(
        layer2_mpo, compress=True, cutoff=cutoff, max_bond=max_bond
    )
    if verbosity == 1:
        print(f"{'': <4}Intermediate bond dimension:", U_trotter4_mpo.max_bond(), "\n")
        print(f"{'': <4}Multiplying intermediate and third MPOs", "\n")
    U_trotter4_mpo = U_trotter4_mpo.apply(
        layer2_mpo, compress=True, cutoff=cutoff, max_bond=max_bond
    )
    if verbosity == 1:
        print(f"{'': <4}Final bond dimension:", U_trotter4_mpo.max_bond())

    return U_trotter4_mpo


# --------------------------------------------------------------------------


def trotter_approx_as_MPO(
    hamiltonian, *, dt, order, cutoff, max_bond, verbosity=0
):

    ham_terms = hamiltonian.terms
    n_qubits = hamiltonian.n_qubits

    # list_bondims = []
    if order == 1:
        U_trotter_mpo = trotter1_approx_as_MPO(
            ham_terms,
            n_qubits,
            dt=dt,
            cutoff=cutoff,
            max_bond=max_bond,
        )

    elif order == 2:
        U_trotter_mpo = trotter2_approx_as_MPO(
            ham_terms,
            n_qubits,
            dt=dt,
            cutoff=cutoff,
            max_bond=max_bond,
            verbosity=verbosity,
        )

    elif order == 4:
        U_trotter_mpo = trotter4_approx_as_MPO(
            ham_terms,
            n_qubits,
            dt=dt,
            cutoff=cutoff,
            max_bond=max_bond,
            verbosity=verbosity,
        )
    else:
        raise ValueError(f"order {order} not implemented")

    # print(*(U_trotter_mpo[i].shape for i in range(n_qubits)), sep="\n")
    return U_trotter_mpo


# THIS IS TO BE MERGED IN A SMALL PR WITH THE EXISTING FUNCTION
def generate_brickwall_circuit_modified(
    n_qubits,
    depth,
    one_qubit_gate_label,
    two_qubit_gate_label,
    *,
    start_ent=False,
    purely_ent=False,  # whether or not to do 1-spin rotations
    param_scaling=1.0,
    rng=None,
):
    """
    Generate a brickwall-structured ``quimb`` :quimb-api:`Circuit`.

    This function constructs a quantum circuit composed of alternating
    single-body layers and nearest-neighbor two-body entangling layers
    arranged in a brickwall pattern. Each circuit layer is assigned a
    distinct circuit round.

    Circuit structure (one layer, ``start_ent=False``)::

        q0 ──[]───●───────
                  │
        q1 ──[]───●───●───
                      │
        q2 ──[]───●───●───
                  |
        q3 ──[]───●───●───
                      |
        q4 ──[]───●───●───
                  |
        q5 ──[]───●───────
          <─────────────>
            first layer

    where::

        []     = single-body gate
        ●─●    = nearest-neighbor entangling gate

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.

    depth : int
        Number of circuit layers (identified here with the gate rounds).

    one_qubit_gate_label : str
        Label identifying the single-body gate.

    two_qubit_gate_label : str
        Label identifying the two-body entangling gate.

    start_ent : bool, optional
        If ``True``, each layer starts with the brickwall entangling layer.
        Otherwise (default ``False``), the single-body layer is applied first.

    param_scaling : float, default ``1.0``
        Scaling factor for randomly initialized parameters.

    rng : :numpy-random:`numpy.random.Generator <generator>`, optional
        Random number generator used to generate gate parameters.
        If ``None``, a default generator is created.

    Returns
    -------
    circ : :quimb-api:`Circuit`
        The generated brickwall quantum circuit.

    Raises
    ------
    ValueError
        If ``one_qubit_gate_label`` does not correspond to a valid single-body gate,
        or if ``two_qubit_gate_label`` is not a valid two-body gate.

    Notes
    -----
    - Separate random number generators are used for single-body and
      two-body gate parameters to ensure reproducibility and decoupled
      randomness.
    - The same gate parameters are reused across all gates within a
      given layer.
    """
    if one_qubit_gate_label.upper() not in qtn.circuit.ONE_QUBIT_GATES:
        raise ValueError(f"Expected a single-body gate: {one_qubit_gate_label}")
    if two_qubit_gate_label.upper() not in qtn.circuit.TWO_QUBIT_GATES:
        raise ValueError(f"Expected a two-body gate: {two_qubit_gate_label}")
    if rng is None:
        rng = np.random.default_rng()

    circ = qtn.Circuit(n_qubits)
    for k in range(depth):
        if start_ent:
            for start in range(2):
                two_qubit_nn_layer(
                    circ,
                    start,
                    two_qubit_gate_label,
                    param_scaling=param_scaling,
                    gate_round=k,
                    rng=rng,
                )
            if not purely_ent:
                one_qubit_layer(circ, one_qubit_gate_label, gate_round=k)
        else:
            if not purely_ent:
                one_qubit_layer(circ, one_qubit_gate_label, gate_round=k)
            for start in range(2):
                two_qubit_nn_layer(
                    circ,
                    start,
                    two_qubit_gate_label,
                    param_scaling=param_scaling,
                    gate_round=k,
                    rng=rng,
                )

    return circ


def init_cost_tn(
    unitary_mpo, depth, *, param_scaling=1e-1, factorize=False, closed=False, seed=42
):
    """
    By defect, "SU4" are fed unfactorized in the circuits,
    while others like "RZZ" are always split by defect.

    Since we choose "SU4" as a starting point,
    we give the option of splitting
    """

    n_qubits = unitary_mpo.num_tensors
    rng = np.random.default_rng()
    TN = TensorNetwork()

    # -----------------------------------------
    # Define the Ansatz as a brickwall circuit

    bw_circ = generate_brickwall_circuit_modified(
        n_qubits=n_qubits,
        depth=depth,
        one_qubit_gate_label="U1",  # it is irrelevant (purely_ent cancels its effect)
        two_qubit_gate_label="SU4",
        start_ent=True,
        purely_ent=True,  # whether or not to do 1-spin rotations
        param_scaling=param_scaling,  # initialize close to identity
        rng=rng,
    )

    bw_unitary_tn = bw_circ.get_uni()
    r"""
    if factorize:
        n_gates = int(
            re.search(r"\d+", next(iter(bw_unitary_tn.tensors[-1].tags))).group()
        )

        #for n in range(1, n_gates + 1):
        #    gate_tens = bw_unitary_tn.select(tags=(f"GATE_{n}"), which="any").tensors
            # bw_unitary_tn.contract_between(tags1=gate_tens[0].tags, tags2=gate_tens[1].tags) # contract (if RZZ)
            # # do the same but splitting!
    """

    TN = TN & bw_unitary_tn

    # -----------------------------------
    # Add the MPO unitary in the network

    new_ket_ind = "B"  # by defect leave the (B)ra open
    if closed:
        new_ket_ind = "k"  # execute the trace by contracting the (k)et

    for x, tensor in enumerate(unitary_mpo):
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

    env_tn = cost_tn.select(tags=[f"I{x}"], which="!any")

    left_tags = [f"I{x}" for x in range(x)]
    right_tags = [f"I{x}" for x in range(x + 1, n_qubits)]

    list_envs_tns = []
    if x > 0:
        list_envs_tns.append(env_tn.select(tags=left_tags, which="any"))
    if x < n_qubits - 1:
        list_envs_tns.append(env_tn.select(tags=right_tags, which="any"))

    return list_envs_tns


def find_transfer_structure(n_qubits, cost_tn):

    dict_uncontr_envs = {"L": {}, "R": {}}
    for x in range(n_qubits):
        list_envs_tns = get_envs_tns(n_qubits, x, cost_tn)

        if x == 0:
            dict_uncontr_envs["R"][f"I{x}"] = list_envs_tns[0]
        elif x == n_qubits - 1:
            dict_uncontr_envs["L"][f"I{x}"] = list_envs_tns[0]
        else:
            dict_uncontr_envs["L"][f"I{x}"] = list_envs_tns[0]
            dict_uncontr_envs["R"][f"I{x}"] = list_envs_tns[1]

    dict_transf = {"L": {}, "R": {}}
    for x in range(1, n_qubits - 1):
        # left transfer
        transf_tn = dict_uncontr_envs["L"][f"I{x + 1}"].select(
            tags=(f"I{x}"), which="any"
        )
        transf_tags = get_tags(transf_tn)
        filtered_tags = []
        for tag in transf_tags:
            if tag[0] == "G" or tag[0] == "U":
                filtered_tags.append(tag)
        dict_transf["L"][f"L{x}_to_L{x + 1}"] = filtered_tags

        # right transfer
        transf_tn = dict_uncontr_envs["R"][f"I{n_qubits - 2 - x}"].select(
            tags=(f"I{n_qubits - 1 - x}"), which="any"
        )
        transf_tags = get_tags(transf_tn)
        filtered_tags = []
        for tag in transf_tags:
            if tag[0] == "G" or tag[0] == "U":
                filtered_tags.append(tag)
        dict_transf["R"][f"R{n_qubits - 1 - x}_to_R{n_qubits - 2 - x}"] = filtered_tags

    return dict_transf


def build_first_sweep(n_qubits, cost_tn, dict_transf, *, drop_tags=True):
    """Generate the first set of left and right environments (recycling the former)
    and save them in a list of tensor networks.
    """

    dict_contr_envs = {"L": {}, "R": {}}

    # the first environments are L{1} and R{n_qubits-1}, which are the edge tensors on the MPO
    L_init = cost_tn.select(tags="Uref0", which="all").tensors[0]
    L_init.add_tag(tag=("L1"))
    R_init = cost_tn.select(tags=f"Uref{n_qubits - 1}", which="all").tensors[0]
    R_init.add_tag(tag=(f"R{n_qubits - 2}"))

    dict_contr_envs["L"]["L1"] = L_init
    dict_contr_envs["R"][f"R{n_qubits - 2}"] = R_init
    L_next = L_init.copy(deep=True)
    R_next = R_init.copy(deep=True)

    for counter, transf_tags in enumerate(
        zip(dict_transf["L"].values(), dict_transf["R"].values(), strict=True)
    ):
        L_next = L_next.copy(deep=True)
        L_transf_tn = cost_tn.select(tags=transf_tags[0], which="any")
        L_next = L_next & L_transf_tn
        L_next = tensor_contract(*L_next.tensors, drop_tags=drop_tags)
        L_next.add_tag(tag=(f"L{counter + 2}"))

        dict_contr_envs["L"][f"L{counter + 2}"] = L_next

        R_next = R_next.copy(deep=True)
        R_transf_tn = cost_tn.select(tags=transf_tags[1], which="any")
        R_next = R_next & R_transf_tn
        R_next = tensor_contract(*R_next.tensors, drop_tags=drop_tags)
        R_next.add_tag(tag=(f"R{n_qubits - 3 - counter}"))

        dict_contr_envs["R"][f"R{n_qubits - 3 - counter}"] = R_next

    return dict_contr_envs


def build_loc_cost_tn(n_qubits, x, dict_contr_envs, cost_tn):
    """Build intermediate contractions S=L{x}-gates-R{x}
    Return ordered list of tags of gates to optimize.
    """

    gates_to_opt = cost_tn.select(tags=f"I{x}", which="any")
    tags_to_opt = get_tags(gates_to_opt)

    filtered_tags = []
    for tag in tags_to_opt:
        if tag[0] == "G":
            filtered_tags.append(tag)

    if x == 0:
        loc_cost_tn = gates_to_opt & dict_contr_envs["R"][f"R{x}"]
    elif x == n_qubits - 1:
        loc_cost_tn = dict_contr_envs["L"][f"L{x}"] & gates_to_opt
    else:
        loc_cost_tn = (
            dict_contr_envs["L"][f"L{x}"] & gates_to_opt & dict_contr_envs["R"][f"R{x}"]
        )

    gate_to_opt_tags = sorted(filtered_tags, key=lambda s: int(s.split("_")[1]))

    return loc_cost_tn, gate_to_opt_tags


def PRC_loc_cost_tn(loc_cost_tn, tags, hyperopt):
    """
    Pop-Rehearse-Contract.

    this will be called from sweep, and will only rehearse on the first sweep;
    can pop more than one gate!
    """

    p_loc_cost_tn = loc_cost_tn.copy(deep=True)
    p_loc_cost_tn.delete(tags=tags)

    return p_loc_cost_tn.contract(optimize=hyperopt)


"""
list_methods = ["sgu", "vgcu"]
    dict_methods = {
        "sgu": "single gate update",
        "vgcu": "variational gate column update"
        }
    if method not in ["sgu", "vgcu"]:
        raise ValueError(f"Available methods: ", list_methods)
        """


def update_cost_tn(cost_tn, list_opt_gate_tens):

    for tens in list_opt_gate_tens:
        tag_tens = next(
            iter(tens.tags)
        )  # list(tens.tags)[0]  # the first tag is "GATE_{n}" by construction
        cost_tn.delete(tags=tag_tens)  # delete the old tensor with same tags
        cost_tn = cost_tn & tens  # add the new tensor

    return cost_tn


def update_dict_contr_envs(
    mode, list_opt_gate_tens, cost_tn, dict_transf, dict_contr_envs
):
    """Getting the x externally simplifies everything bc automatically know the column we are talking about."""
    for tens in list_opt_gate_tens:
        list_tags = list(tens.tags)
        # tag_tens = list_tags[0]  # the first tag is "GATE_{n}" by construction
        n = int(
            re.search(r"\d+", list_tags[3]).group()
        )  # number of the left site from tag

        if mode == "LR":  # sweeping L to R only requires updating L's
            # the gate couples I{n} and I{n+1}
            # affects transition L{n+1}_to_L{n+2}
            # so update L{n+2}
            # print(f"L{n+2}")
            transf_tens = cost_tn.select(
                tags=dict_transf["L"][f"L{n + 1}_to_L{n + 2}"], which="any"
            )
            new_env = dict_contr_envs["L"][f"L{n + 1}"].copy(deep=True) & transf_tens
            dict_contr_envs["L"][f"L{n + 2}"] = new_env.contract()

        if mode == "RL":  # sweeping R to L only requires updating R's
            # the gate couples I{n} and I{n+1}
            # affects transition R{n}_to_R{n-1}
            # so update R{n-1}
            # print(f"R{n-1}")
            transf_tens = cost_tn.select(
                tags=dict_transf["R"][f"R{n}_to_R{n - 1}"], which="any"
            )
            new_env = dict_contr_envs["R"][f"R{n}"].copy(deep=True) & transf_tens
            dict_contr_envs["R"][f"R{n - 1}"] = new_env.contract()

    return dict_contr_envs


def optimize_single_gate_update(n_qubits, cost_tn, n_sweeps, dict_transf, dict_contr_envs):
    """
    Receives the cost function and optimizes for n_sweeps.

    notes: the sum of singular values after doing the SVD of the environment contraction around a gate
    coincides with the value of the cost function -> sanity check (already did it): contraction of cost_tn
    and contraction of local_cost_tn yields the same value as the sum of singular values
    """

    # instantiate hypercontractor
    hyperopt = ctg.ReusableHyperOptimizer(
        # just do a few runs
        max_repeats=32,
        # only use the basic greedy optimizer ...
        methods=["greedy"],
        # ... but pair it with reconfiguration
        reconf_opts={},
        # just uniformly sample the space
        optlib="random",
        # terminate search if contraction is cheap
        max_time="rate:1e9",
        # account for both flops and write - usually wise for practical performance
        minimize="combo",
        parallel=False,
    )  # different from hyperopt_env

    instruct = [
        ("LR", list(range(n_qubits - 2))),
        ("RL", list(reversed(range(2, n_qubits)))),
    ]
    trange_counter = tqdm(list(range(n_sweeps)))
    for _ in trange_counter:  # set a progressbar and measure error
        for sweep in instruct:
            for x in sweep[1]:
                # build local cost function for all gates at position "x"
                loc_cost_tn, gate_to_opt_tags = build_loc_cost_tn(
                    n_qubits,
                    x=x,
                    dict_contr_envs=dict_contr_envs,
                    cost_tn=cost_tn,
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
                        hyperopt=hyperopt,
                    )

                    # do the SVD
                    prc_loc_cost_UsVh = tensor_split(
                        T=prc_loc_cost_tens,
                        # recall index ordering in Gate class:
                        # (OUT_LEFT, OUT_RIGHT, IN_LEFT, IN_RIGHT)
                        left_inds=(inds[0], inds[1]),
                        method="svd",
                        absorb=None,
                    )

                    # retain isometries
                    overlap = np.sum(prc_loc_cost_UsVh.tensors[1].data)
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
                    cost_tn = update_cost_tn(
                        cost_tn, list_opt_gate_tens=[new_gate_tens]
                    )
                    loc_cost_tn = update_cost_tn(
                        loc_cost_tn, list_opt_gate_tens=[new_gate_tens]
                    )

                    # update transfer tensors and environments
                    dict_contr_envs = update_dict_contr_envs(
                        mode=sweep[0],
                        list_opt_gate_tens=[original_gate_tens],
                        cost_tn=cost_tn,
                        dict_transf=dict_transf,
                        dict_contr_envs=dict_contr_envs,
                    )

    return cost_tn, dict_contr_envs
