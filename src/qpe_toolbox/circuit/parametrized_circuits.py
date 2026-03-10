# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

import autoray
import numpy as np
import quimb as qu
import quimb.tensor as qtn

list_single_body_labels = list(qtn.circuit.ONE_QUBIT_GATES)
list_two_body_labels = list(qtn.circuit.TWO_QUBIT_GATES)
list_all_labels = (
    list_single_body_labels + list_two_body_labels
)  # "all": 1- and 2-qubit gates only
list_all_param_labels = list(qtn.circuit.ALL_PARAM_GATES)
dict_quimb_all_param_single_body_numbers = {
    "RX": 1,
    "RY": 1,
    "RZ": 1,
    "U1": 1,
    "U2": 2,
    "U3": 3,
    "PHASE": 1,
}
dict_quimb_all_param_two_body_numbers = {
    "CRX": 1,
    "CRY": 1,
    "CRZ": 1,
    "CU1": 1,
    "CU2": 2,
    "CU3": 3,
    "CPHASE": 1,
    "RXX": 1,
    "RYY": 1,
    "RZZ": 1,
    "XXMINUSYY": 2,
    "XXPLUSYY": 2,
    "GIVENS": 1,
    "GIVENS2": 2,
    "FS": 2,
    "FSIM": 2,
    "FSIMG": 5,
    "SU4": 15,
}


def one_qubit_layer(
    circ, gate_label, *, random_coeff=1.0, gate_round=None, parametrize=False
):
    """Apply a single-body gate layer to all qubits of a ``quimb`` :class:`quimb.tensor.circuit.Circuit`.

    This function applies the same single-qubit gate to every qubit in the
    circuit at a specified circuit round. If the gate is parametrized,
    random parameters are generated using the provided random number
    generator and shared across all qubits in the layer.

    Parameters
    ----------
    circ : :class:`quimb.tensor.circuit.Circuit`
        The quimb circuit to which the layer is applied.

    gate_label : str
        Label identifying the single-body gate to apply (e.g. ``"RX"``).
        The label must be compatible with :meth:`quimb.tensor.circuit.Circuit.apply_gate`.

    random_coeff : float, default ``1.0``
        Scaling factor for randomly initialized parameters.

    gate_round : int or None, default ``None``
        Circuit layer at which the gates are applied.

    parametrize : bool, default ``False``
        Activate the possibility of using the layer as a parametrized Ansatz
        on some variational scheme.

    Notes
    -----
    - If ``gate_label`` corresponds to a parameterized gate, the number of
      parameters is inferred from ``dict_quimb_all_param_single_body_numbers``.
    - The same set of parameters is used for all qubits in the layer.

    """

    if gate_label.upper() in list_all_param_labels:
        list_params = random_coeff * qu.randn(
            shape=(dict_quimb_all_param_single_body_numbers[gate_label.upper()]),
            dist="uniform",
        )
    elif gate_label.upper() in list_all_labels:
        list_params = []
    else:
        raise ValueError(f"Expected a gate from: {list_all_labels}")

    extra_kwargs = {}
    if len(list_params) > 0:
        # quimb automatically raises an error for constant gates
        # if any parametrize is passed, even if False
        extra_kwargs["parametrize"] = bool(parametrize)

    for i in range(circ.N):
        circ.apply_gate(
            gate_id=gate_label,
            params=list_params,
            qubits=[i],
            gate_round=gate_round,
            **extra_kwargs,
        )


####################################################


def two_qubit_nn_layer(
    circ,
    start,
    gate_label,
    *,
    random_coeff=1.0,
    gate_round=None,
    parametrize=False,
    reverse=False,
):
    """Apply a nearest-neighbor two-body entangling layer to a ``quimb`` :class:`quimb.tensor.circuit.Circuit`.

    This function applies a two-qubit entangling gate between nearest
    neighbors in a brickwork pattern. The starting qubit index determines
    the parity of the layer.

    Parameters
    ----------
    circ : :class:`quimb.tensor.circuit.Circuit`
        The ``quimb`` circuit to which the layer is applied.

    start : int
        Starting qubit index for the nearest-neighbor pattern (typically
        ``0`` or ``1``). Allows for defining even (0) and odd (1) layers.

    gate_label : str
        Label identifying the two-body entangling gate (e.g. ``"CNOT"``)

    random_coeff : float, default ``1.0``
        Scaling factor for randomly initialized parameters.

    gate_round : int or None, default ``None``
        Gate round index used to tag gates.

    parametrize : bool, default ``False``
        Activate the possibility of using the layer as a parametrized Ansatz
        on some variational scheme.

    reverse : bool, default ``False``
        Possibility to invert direction of the layer.
        Relevant whe using controlled gates.

    Returns
    -------
    circ : :class:`quimb.tensor.circuit.Circuit`
        The circuit with the entangling layer applied.

    Notes
    -----
    - Parameterized gates draw their parameter count from
      ``dict_quimb_all_param_two_body_numbers``.
    - The same parameters are reused for all entangling gates in the layer.
    - Gates are applied between qubits ``(i, i+1)`` for
      ``i = start, start+2,`` ...

    """
    if gate_label.upper() in list_all_param_labels:
        list_params = random_coeff * qu.randn(
            shape=(dict_quimb_all_param_two_body_numbers[gate_label.upper()]),
            dist="uniform",
        )
    elif gate_label.upper() in list_all_labels:
        list_params = []
    else:
        raise ValueError(f"Expected a gate from: {list_all_labels}")

    extra_kwargs = {}
    if len(list_params) > 0:
        extra_kwargs["parametrize"] = bool(parametrize)

    if reverse:
        order = reversed(range(start, circ.N - 1, 2))
    else:
        order = range(start, circ.N - 1, 2)
    for i in order:
        circ.apply_gate(
            gate_id=gate_label,
            params=list_params,
            qubits=[i, i + 1],
            gate_round=gate_round,
            **extra_kwargs,
        )

    return circ


def two_qubit_rand_layer(
    circ,
    gate_label,
    gate_range,
    gate_prob,
    *,
    rng=None,
    random_coeff=1.0,
    gate_round=None,
    parametrize=True,
    reverse=False,
):
    """Apply a random two-body entangling layer to a ``quimb`` :class:`quimb.tensor.circuit.Circuit`.
    This function applies two-qubit entangling gates between randomly chosen
    qubit pairs. For each qubit, a partner qubit is selected within a given
    range, and the entangling gate is applied with a specified probability.

    Parameters
    ----------
    circ : :class:`quimb.tensor.circuit.Circuit`
        The quimb circuit to which the layer is applied.

    gate_label : str
        Label identifying the two-body entangling gate.

    gate_range : int
        Sets a maximum interaction range ``(gate_range+1)`` for two-body entangling gates,
        measured in qubit index separation.

    gate_prob : float
        Probability threshold controlling whether an entangling gate is
        applied. A gate is applied if ``rng_prob.random() <= gate_prob``.

    rng : :class:`numpy.random.Generator`, default ``None``
        Random number generator for ``gate_range`` and ``gate_prob``.

    random_coeff : float, default ``1.0``
        Scaling factor for randomly initialized parameters.

    gate_round : int or None, default ``None``
        Gate round index used to tag gates.

    parametrize : bool, default ``False``
        Activate the possibility of using the layer as a parametrized Ansatz
        on some variational scheme.

    reverse : bool, default ``False``
        Possibility to invert direction of the layer.
        Relevant whe using controlled gates.

    Returns
    -------
    circ : :class:`quimb.tensor.circuit.Circuit`
        The circuit with the random entangling layer applied.

    Notes
    -----
    - Parameterized gates draw their parameter count from
      ``dict_quimb_all_param_two_body_numbers``.
    - The same parameters are reused for all entangling gates in the layer.
    - Entangling partners are chosen as ``j = i + 1 + Δ``, where
      ``Δ`` is sampled uniformly from ``[0, gate_range)``.

    """
    if gate_label.upper() in list_all_param_labels:
        list_params = random_coeff * qu.randn(
            dict_quimb_all_param_two_body_numbers[gate_label.upper()], dist="uniform"
        )
    elif gate_label.upper() in list_all_labels:
        list_params = []
    else:
        raise ValueError(f"Expected a gate from: {list_all_labels}")

    if rng is None:
        rng = np.random.default_rng()

    extra_kwargs = {}
    if len(list_params) > 0:
        # quimb automatically raises an error for constant gates
        # if any parametrize is passed, even if False
        extra_kwargs["parametrize"] = bool(parametrize)

    n_qubits = circ.N
    order = reversed(range(n_qubits)) if reverse else range(n_qubits)
    for i in order:
        if rng.random() < gate_prob:
            j = i + 1 + rng.integers(gate_range)
            if j < n_qubits:
                circ.apply_gate(
                    gate_id=gate_label,
                    qubits=[i, j],
                    params=list_params,
                    gate_round=gate_round,
                    **extra_kwargs,
                )

    return circ


def generate_brickwall_quimb(
    n_qubits,
    depth,
    sb_gate_label,
    ent_gate_label,
    *,
    start_ent=False,
    random_coeff=1.0,
    rng=None,
):
    """Generate a brickwall-structured ``quimb`` :class:`quimb.tensor.circuit.Circuit`.

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

    sb_gate_label : str
        Label identifying the single-body gate.

    ent_gate_label : str
        Label identifying the two-body entangling gate.

    start_ent : bool, optional
        If ``True``, each layer starts with the brickwall entangling layer.
        Otherwise (default ``False``), the single-body layer is applied first.

    random_coeff : float, default ``1.0``
        Scaling factor for randomly initialized parameters.

    rng : :class:`numpy.random.Generator`, optional
        Random number generator used to generate gate parameters.
        If ``None``, a default generator is created.

    Returns
    -------
    circ : :class:`quimb.tensor.circuit.Circuit`
        The generated brickwall quantum circuit.

    Raises
    ------
    ValueError
        If ``sb_gate_label`` does not correspond to a valid single-body gate,
        or if ``ent_gate_label`` is not a valid two-body gate.

    Notes
    -----
    - Separate random number generators are used for single-body and
      two-body gate parameters to ensure reproducibility and decoupled
      randomness.
    - The same gate parameters are reused across all gates within a
      given layer.

    """
    if sb_gate_label.lower() not in [lab.lower() for lab in list_single_body_labels]:
        raise ValueError(f"Expected a single-body gate: {sb_gate_label}")
    if ent_gate_label.lower() not in [lab.lower() for lab in list_two_body_labels]:
        raise ValueError(f"Expected a two-body gate: {ent_gate_label}")

    circ = qtn.Circuit(n_qubits)
    for k in range(depth):
        if start_ent:
            for start in range(2):
                two_qubit_nn_layer(
                    circ,
                    start,
                    ent_gate_label,
                    random_coeff=random_coeff,
                    gate_round=k,
                )
            one_qubit_layer(circ, sb_gate_label, gate_round=k)
        else:
            one_qubit_layer(circ, sb_gate_label, gate_round=k)
            for start in range(2):
                two_qubit_nn_layer(
                    circ,
                    start,
                    ent_gate_label,
                    random_coeff=random_coeff,
                    gate_round=k,
                )

    return circ


def generate_rand_quimb(
    n_qubits,
    depth,
    sb_gate_label,
    ent_gate_label,
    ent_gate_range,
    ent_gate_prob,
    *,
    start_ent=False,
    random_coeff=1.0,
    rng=None,
):
    """Generate a random entangling ``quimb`` :class:`quimb.tensor.circuit.Circuit`.

    This function constructs a quantum circuit consisting of alternating
    single-body layers and randomly generated two-body entangling layers.
    Entangling gates are applied probabilistically between qubits within
    a finite interaction range.

    Circuit structure (one layer, ``start_ent=False``)::

        q0 ──[]───●───────
                  │
        q1 ──[]───●───●───
                      │
        q2 ──[]───●───|───  ^
                  |   |     |
        q3 ──[]───|───●───  |
                  |         | ent_gate_range=2
        q4 ──[]───|───────  |
                  |         |
        q5 ──[]───●───────  v
          <─────────────>
            first layer


    where::

        []     = single-body gate
        ●─●    = entangling gate

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.

    depth : int
        Number of circuit layers (rounds).

    sb_gate_label : str
        Label identifying the single-body gate applied at each layer.

    ent_gate_label : str
        Label identifying the two-body entangling gate.

    ent_gate_range : int
        Sets a maximum interaction range ``(ent_gate_range+1)`` for two-body entangling gates,
        measured in qubit index separation.

    ent_gate_prob : float
        Probability threshold controlling the application of an entangling
        gate for a given qubit.

    start_ent : bool, optional
        If ``True``, each layer starts with the random entangling layer.
        Otherwise (default ``False``), the single-body layer is applied first.

    random_coeff : float, default ``1.0``
        Scaling factor for randomly initialized parameters.

    rng : :class:`numpy.random.Generator`, optional
        Random number generator used to sample community sizes.
        If ``None``, a default generator is created.

    Returns
    -------
    circ : :class:`quimb.tensor.circuit.Circuit`
        The generated random entangling quantum circuit.

    Raises
    ------
    ValueError
        If ``sb_gate_label`` does not correspond to a valid single-body gate,
        or if ``ent_gate_label`` is not a valid two-body gate.

    Notes
    -----
    - Separate random number generators are used for:

        - single-body gate parameters,
        - two-body gate parameters,
        - entanglement range selection,
        - entanglement occurrence probability.

    - This separation ensures reproducibility while avoiding unintended
      correlations between different sources of randomness.
    - Gate parameters are shared across all gates within the same layer.

    """
    if sb_gate_label.lower() not in [lab.lower() for lab in list_single_body_labels]:
        raise ValueError(f"Expected a single-body gate: {sb_gate_label}")
    if ent_gate_label.lower() not in [lab.lower() for lab in list_two_body_labels]:
        raise ValueError(f"Expected a two-body gate: {ent_gate_label}")
    if rng is None:
        rng = np.random.default_rng()

    circ = qtn.Circuit(n_qubits)

    for k in range(depth):
        if start_ent:
            two_qubit_rand_layer(
                circ,
                ent_gate_label,
                gate_range=ent_gate_range,
                gate_prob=ent_gate_prob,
                random_coeff=random_coeff,
                gate_round=k,
                rng=rng,
            )
            one_qubit_layer(circ, sb_gate_label, gate_round=k)
        else:
            one_qubit_layer(circ, sb_gate_label, gate_round=k)
            two_qubit_rand_layer(
                circ,
                ent_gate_label,
                gate_range=ent_gate_range,
                gate_prob=ent_gate_prob,
                random_coeff=random_coeff,
                gate_round=k,
                rng=rng,
            )

    return circ


def ansatz_circuit(n, depth, *, gate_round=0, random_coeff=1.0):
    """Construct an ansatz circuit of single qubit and entangling layers.

    Parameters
    ----------
    n : int
        Number of qubits.
    depth : int
        Number of repeated ansatz layers.
    gate_round : int, default ``0``
        Starting gate round index.
    random_coeff : float, default ``1.0``
        Scaling factor for random parameter initialization.

    Returns
    -------
    quimb.tensor.Circuit
        Parametrized ansatz circuit.

    """
    circ = qtn.Circuit(n)
    for r in range(gate_round, gate_round + depth):
        # single qubit gate layer
        one_qubit_layer(
            circ,
            "U3",
            random_coeff=random_coeff,
            gate_round=r,
            parametrize=True,
        )
        # even-odd two qubit gate layer
        for start in range(2):
            two_qubit_nn_layer(
                circ,
                start=start,
                gate_label="RZZ",
                random_coeff=random_coeff,
                gate_round=r,
                parametrize=True,
            )
    return circ


def ansatz_circuit_su4(n, depth, *, gate_round=0, random_coeff=1.0):
    """
    Construct an ansatz circuit using SU(4) two-qubit gates.

    Parameters
    ----------
    n : int
        Number of qubits.
    depth : int
        Number of circuit layers.
    gate_round : int, default ``0``
        Starting gate round index.
    random_coeff : float, default ``1.0``
        Scaling factor for random parameter initialization.

    Returns
    -------
    quimb.tensor.Circuit
        Parametrized SU(4) ansatz circuit.

    """
    circ = qtn.Circuit(n)
    for r in range(gate_round, gate_round + depth):
        for start in range(2):
            two_qubit_nn_layer(
                circ,
                start=start,
                gate_label="SU4",
                random_coeff=random_coeff,
                gate_round=r,
                parametrize=True,
            )
    return circ


def recursive_stack(x):
    """
    Recursively stack nested sequences into a tensor.

    Parameters
    ----------
    x : array-like or nested sequence
        Input structure to stack.

    Returns
    -------
    array-like
        Stacked array compatible with the active autoray backend.

    """
    if not isinstance(x, (list, tuple)):
        return x
    return autoray.do("stack", tuple(map(recursive_stack, x)))


def ansatz_circuit_sym(n, depth, *, gate_round=0, random_coeff=1.0):
    """
    Construct a symmetry-preserving ansatz circuit.

    This ansatz uses custom XX+YY gates, RZZ entanglers,
    and single-qubit Z rotations.

    Parameters
    ----------
    n : int
        Number of qubits.
    depth : int
        Number of ansatz layers.
    gate_round : int, default ``0``
        Starting gate round index.
    random_coeff : float, default ``1.0``
        Scaling factor for random parameter initialization.

    Returns
    -------
    quimb.tensor.Circuit
        Parametrized symmetric ansatz circuit.

    """

    circ = qtn.Circuit(n)
    if gate_round == 0:
        for i in range(circ.N // 2):
            circ.apply_gate("X", 2 * i)
    for r in range(gate_round, gate_round + depth):
        for start in range(2):
            two_qubit_nn_layer(
                circ,
                start,
                "XXPLUSYY",
                random_coeff=random_coeff,
                gate_round=r,
                parametrize=True,
            )
        for start in range(2):
            two_qubit_nn_layer(
                circ,
                start,
                "RZZ",
                random_coeff=random_coeff,
                gate_round=r,
                parametrize=True,
            )
        one_qubit_layer(
            circ, "RZ", random_coeff=random_coeff, gate_round=r, parametrize=True
        )
    return circ
