# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import quimb.tensor as qtn

_text_kwargs = {"color": "k", "ha": "center", "va": "center"}


def assign_sublayers_per_round(circ, gate_round):
    """Assign non-overlapping two-qubit gates of a given circuit round
    to sublayers for plotting purposes.

    This function groups all two-qubit gates acting in a given entangling layer (``gate_round``)
    into sublayers, such that no two gates within the same sublayer act on overlapping
    qubit intervals when the circuit is plotted in a 1D layout. For example, in QAOA:

    :math:`U_x U_{zz} \\langle\\Psi\\rangle \\rightarrow U_{zz}` may include arbitrarily long-range two-qubit gates
    spanning on overlapping register lines

    The sublayers are constructed greedily by sorting gates by their qubit
    intervals and placing each gate in the earliest compatible sublayer.

    Parameters
    ----------
    circ : :quimb-api:`Circuit`
        Quantum circuit.

    gate_round : int
        Circuit round for which the sublayers are constructed.

    Returns
    -------
    dict_gates_to_sublayers : dict
        Mapping from the two-qubit gates in a depth/round/layer
        (represented as an ordered qubit pair ``(i, j)`` with ``i < j``)
        to the index of the sublayer it is assigned to.
        The overall structure is ``dict[tuple[int, int], int]``
        corresponding to ``dict_mappings[tuple[qubit1, qubit2], sublayer_ind]``.

    list_sublayers : list
        List of sublayers for the given depth/round/layer.
        Each element of the list corresponds to one circuit layer,
        and contains a list of sublayers, where each sublayer
        is represented as ``(sublayer_index, gates)``;
        ``gates`` is a list of two-qubit edges ``(i, j)``.
        The overall structure is ``list[tuple[int, list[tuple[int, int]]]]``,
        corresponding to ``list_sublayers[tuple[sublayer_end, list[tuple[qubit1,qubit2]]]]``.
        ``sublayer_end`` refers to the largest qubit index occupied in the sublayer.

    Notes
    -----
    - Only two-qubit gates (``len(gate.qubits) == 2``) are considered.
    - Gates are treated as closed intervals ``[i, j]`` on the qubit line.
    - The algorithm is greedy and does not guarantee a minimum number of
      sublayers, but is sufficient for circuit plotting.
    - This function assumes that the full circuit is being plotted.

    """
    # Get the list of gates as intervals - (start, end) - of the two-body gates within the layer 'round'
    list_gates = []
    for gate in circ.gates:
        # Select the subset of gates acting on the target depth given by 'round'
        if len(gate.qubits) == 2 and gate.round == gate_round:
            if gate.qubits[0] < gate.qubits[1]:
                list_gates.append((gate.qubits[0], gate.qubits[1]))
            else:
                list_gates.append((gate.qubits[1], gate.qubits[0]))

    list_gates.sort()
    list_sublayers = []
    dict_gates_to_sublayers = {}

    # Loop over detected intervals
    for gate in list_gates:
        i, j = gate
        placed = False
        # Check the list of sublayers (grouping one or more gates)
        for idx, (end_edge, gates_in_layer) in enumerate(list_sublayers):
            if end_edge < i:  # current gate starts after the interval is finished
                # add new gate and update interval edge
                list_sublayers[idx] = (j, [*gates_in_layer, gate])
                dict_gates_to_sublayers[gate] = idx
                placed = True
                break
        # If not assigned yet, assign a new sublayer
        if not placed:
            list_sublayers.append((j, [gate]))
            dict_gates_to_sublayers[gate] = len(list_sublayers) - 1

    return dict_gates_to_sublayers, list_sublayers


def assign_sublayers(circ):
    """Assign non-overlapping two-qubit gates of a given circuit
    to sublayers at all depths for plotting purposes. See :func:`assign_sublayers_per_round`.

    Parameters
    ----------
    circ : :quimb-api:`Circuit`
        Quantum circuit.

    Returns
    -------
    list_dict_gates_to_sublayers : list
        List of mappings from the two-qubit gates in a depth/round/layer
        (represented as an ordered qubit pair ``(i, j)`` with ``i < j``)
        to the index of the sublayer it is assigned to.
        The overall structure is ``list[dict[tuple[int, int], int]]``
        corresponding to ``list_layers[dict_mappings[tuple[qubit1, qubit2], sublayer_ind]]``.

    list_sublayers : list
        Nested list of sublayers organized by circuit depth.
        When we refer to depth, we also refer to round or layer indistinctly.
        Each element of the list corresponds to one circuit layer,
        and contains a list of sublayers, where each sublayer
        is represented as ``(sublayer_index, gates)``;
        ``gates`` is a list of two-qubit edges ``(i, j)``.
        The overall structure is ``list[list[tuple[int, list[tuple[int, int]]]]]``,
        corresponding to ``list_layers[list_sublayers[tuple[sublayer_edge, list[tuple[qubit1,qubit2]]]]]``.

    """
    depth = max(gate.round for gate in circ.gates) + 1
    list_sublayers = []
    list_dict_gates_to_sublayers = []
    for k in range(depth):
        dict_gates_to_sublayers_round_k, list_sublayers_round_k = (
            assign_sublayers_per_round(circ=circ, gate_round=k)
        )

        list_sublayers.append(list_sublayers_round_k)
        list_dict_gates_to_sublayers.append(dict_gates_to_sublayers_round_k)

    return list_dict_gates_to_sublayers, list_sublayers


def _add_square(ax, x, y, col_face):
    """Add a filled square patch to a Matplotlib Axes.

    The square is drawn as a rotated ``RegularPolygon`` with four vertices,
    centered at the specified coordinates.

    Parameters
    ----------
    ax : :matplotlib-api:`axes.Axes`
        Axes object to which the square patch is added.

    x : float
        x-coordinate of the square center.

    y : float
        y-coordinate of the square center.

    col_face : color
        Face color of the square.

    Notes
    -----
    - The patch is drawn in ``zorder=2`` (above ``zorder=1``).

    """
    ax.add_patch(
        mpl.patches.RegularPolygon(
            (x, y),
            numVertices=4,
            radius=0.4,
            orientation=np.pi / 4,
            fc=col_face,
            ec="k",
            linewidth=4,
            zorder=2,
        )
    )


def _add_circle(ax, x, y, col_face):
    """Add a filled circle patch to a Matplotlib Axes,
    centered at the specified coordinates.

    Parameters
    ----------
    ax : :matplotlib-api:`axes.Axes`
        Axes object to which the circle patch is added.

    x : float
        x-coordinate of the circle center.

    y : float
        y-coordinate of the circle center.

    col_face : color
        Face color of the circle.

    Notes
    -----
    - The patch is drawn in ``zorder=2`` (above ``zorder=1``).

    """
    ax.add_patch(
        mpl.patches.Circle(
            (x, y),
            radius=0.4,
            fc=col_face,
            ec="k",
            linewidth=3,
            zorder=2,
        )
    )


def draw_2_qubit_layer(
    ax,
    n_qubits,
    X,
    sublayers,
    dict_sublayer,
    gate_label,
    fontsize,
    col_face,
    active_qubits,
    *,
    reverse=False,
):
    """Draw a two-qubit gate layer with sublayer structure on a Matplotlib Axes.

    This function visualizes a layer of two-qubit gates, which are arranged horizontally
    according to their assigned sublayer to avoid overlaps.

    Parameters
    ----------
    ax : :matplotlib-api:`axes.Axes`
        Axes object on which the layer is drawn.

    n_qubits : int
        Total number of qubits in the circuit. Used to position the layer label.

    X : float
        Horizontal offset for the start of this layer in ``ax``.

    sublayers : list
        List of sublayers. Each entry is expected to be a tuple of the form
        ``(last_end, list_of_gate_pairs)``, where ``list_of_gate_pairs`` contains
        tuples ``(i, j)`` of qubit indices acted on by each gate.

    dict_sublayer : dict
        Mapping from gate qubit pairs ``(i, j)`` to integer sublayer indices.
        These indices determine the horizontal placement of the gates.

    gate_label : str
        Label for the entire layer (e.g. ``r"$U_{zz}$"``, ``r"CNOT"``), drawn above
        the qubit register.

    fontsize : int or float
        Font size used for the layer label.

    col_face : color
        Face color of the square gate markers (Matplotlib-compatible).

    active_qubits : iterable of int
        Indices of qubits on which the gate is applied.
        Useful when drawing expectation values with unitary cancellation.

    reverse : bool, optional
        If ``True``, reverse the horizontal ordering of the sublayers. This is
        useful when plotting circuits from right to left, like in an expectation value
        (see :func:`draw_expval` function).  Default is ``False``.

    Notes
    -----
    - The horizontal position of each gate is ``X + sublayer_index``.

    """
    if reverse:
        max_num = max(dict_sublayer.values())
        mod_dict_sublayer = {gate: max_num - idx for gate, idx in dict_sublayer.items()}
    else:
        mod_dict_sublayer = dict_sublayer

    for sublayer in sublayers:
        for g0, g1 in sublayer[1]:
            if g0 in active_qubits or g1 in active_qubits:
                x = X + mod_dict_sublayer[g0, g1]
                _add_square(ax, x=x, y=g0, col_face=col_face)
                _add_square(ax, x=x, y=g1, col_face=col_face)
                ax.vlines(x, g0, g1, lw=4, color="k", zorder=1)

    ax.text(
        X + (len(sublayers) - 1) / 2,
        n_qubits,
        gate_label,
        size=fontsize,
        **_text_kwargs,
    )


def draw_1_qubit_layer(
    ax,
    n_qubits,
    X,
    gate_label,
    fontsize,
    col_face,
    active_qubits,
):
    """Draw a single-qubit gate layer on selected qubits.

    Parameters
    ----------
    ax : :matplotlib-api:`axes.Axes`
        Axes object on which the layer is drawn.

    n_qubits : int
        Total number of qubits in the circuit.

    X : float
        Horizontal position of the gate layer.

    gate_label : str
        Label for the gate layer (e.g. ``"$R_y$"``, ``"$U_x$"``), drawn above the
        qubit register.

    fontsize : int or float
        Font size used for the layer label.

    col_face : color
        Face color of the square gate markers (Matplotlib-compatible).

    active_qubits : iterable of int
        Indices of qubits on which the gate is applied.
        Useful when drawing expectation values with unitary cancellation.

    Notes
    -----
    - Only qubits listed in ``active_qubits`` are drawn.

    """
    for i in range(n_qubits):
        if i in active_qubits:
            _add_square(ax, x=X, y=i, col_face=col_face)
    ax.text(X, n_qubits, gate_label, size=fontsize, **_text_kwargs)


def draw_init_product_state(
    ax,
    n_qubits,
    X,
    state_label,
    fontsize,
    col_face,
    *,
    is_right_side=False,
):
    """Draw an initial product state on a circuit diagram.

    This function visualizes the initial state of each qubit as a labeled
    circle, optionally placing qubit indices to the left or right.

    Parameters
    ----------
    ax : :matplotlib-api:`axes.Axes`
        Axes object on which the initial state is drawn.

    n_qubits : int
        Total number of qubits.

    X : float
        Horizontal offset for the initial state drawing.

    state_label : str
        Label for the qubit state (e.g. ``"$0$"``, ``"$+$"``).

    fontsize : int or float
        Font size used for the labels.

    col_face : color
        Face color of the state circles.

    is_right_side : bool, optional
        Side on which to draw the qubit index labels. Default is left.

    """
    for i in range(n_qubits):
        ax.text(X + 2 * is_right_side, i, f"{i + 1}", size=fontsize, **_text_kwargs)
        _add_circle(ax, X + 1, i, col_face=col_face)
        ax.text(X + 1, i, state_label, size=fontsize, **_text_kwargs)


def _determine_layout_depth(circ):
    """Determine the horizontal layout depth required to draw a quantum circuit.

    This function computes the total horizontal space needed to plot a circuit
    diagram, accounting for all gate layers and their internal sublayer
    structure.

    Parameters
    ----------
    circ : :quimb-api:`Circuit`
        Quantum circuit whose layout depth is to be determined.

    Returns
    -------
    layout_depth : int
        Total horizontal span required to draw the circuit.

    Notes
    -----
    - The circuit depth is inferred from the maximum gate round.
    - Two-qubit layers are expanded according to their number of sublayers
      (as determined by ``assign_sublayers_per_round``).
    - Fixed offsets are added between layers for readability.

    """
    depth = max(gate.round for gate in circ.gates) + 1

    X_init, X_end = -3, -1

    for layer in range(depth):
        _, sublayers = assign_sublayers_per_round(circ, layer)
        X_end += len(sublayers) + 3

    return X_end - X_init


def draw_layered_circuit(circ, *, max_depth=np.inf, list_names=None):
    """
    Draw a layered quantum circuit using Matplotlib.

    This function visualizes a quantum circuit, ASSUMING it is composed of alternating
    single-qubit and two-qubit layers. The circuit is drawn left-to-right,
    and gates placed according to their layer (round) structure.

    Parameters
    ----------
    circ : :quimb-api:`Circuit`
        Quantum circuit.

    max_depth : int or inf, optional
        Maximum number of circuit layers to draw. Default is inf, the full circuit
        is drawn.

    list_names : list, optional
        Labels used for annotating different parts of the circuit.
        Expected structure:

        - ``list_names[0]`` : str
            Label for the initial product state.
        - ``list_names[1]`` : list of str
            Labels for single-qubit layers, one per circuit layer.
            e.g. for QAOA: ``[f"$\\mathrm{{R_x^{{({i})}} }}$" for i in range(1, p + 1)]``
            where ``p`` is the depth of the Ansatz and ``i`` indicates the layer.
        - ``list_names[2]`` : list of str
            Labels for two-qubit layers, one per circuit layer.

    Returns
    -------
    fig : :matplotlib-api:`figure.Figure`

    """
    n_qubits = circ.N
    gate_rounds = [gate.round for gate in circ.gates]
    if any(x is None for x in gate_rounds):
        # without this information, no packing of gates within a layer
        raise ValueError("Gate round information required.")
    true_max_depth = max(gate_rounds) + 1
    depth = min(max_depth, true_max_depth)
    if list_names is None:
        list_names = [[""], [""] * depth, [""] * depth]  # empty labels

    width = _determine_layout_depth(circ)

    fig, ax = plt.subplots(figsize=(width, n_qubits))
    ax.set_facecolor("white")

    col_psi = "#e8e8e8"
    col_U1 = "#00bc32"
    col_U2 = "#d30000"
    fontsize = 30

    ax.hlines(np.arange(n_qubits), 0, width, lw=4, color="#A9A9A9", zorder=1)

    draw_init_product_state(
        ax=ax,
        n_qubits=n_qubits,
        X=-1,
        state_label=list_names[0],
        fontsize=fontsize,
        col_face=col_psi,
    )

    X = 2

    # Unitary circuit
    for layer in range(depth):
        dict_sublayer, sublayers = assign_sublayers_per_round(circ, layer)
        layer_order = ("2q", "1q") if len(circ.gates[-1].qubits) == 1 else ("1q", "2q")

        for kind in layer_order:
            if kind == "1q":
                draw_1_qubit_layer(
                    ax=ax,
                    n_qubits=n_qubits,
                    X=X,
                    gate_label=list_names[1][layer],
                    fontsize=fontsize,
                    col_face=col_U1,
                    active_qubits=list(range(n_qubits)),
                )
                X += 2

            else:  # "2q"
                draw_2_qubit_layer(
                    ax=ax,
                    n_qubits=n_qubits,
                    X=X,
                    sublayers=sublayers,
                    dict_sublayer=dict_sublayer,
                    gate_label=list_names[2][layer],
                    fontsize=fontsize,
                    col_face=col_U2,
                    active_qubits=list(range(n_qubits)),
                    reverse=False,
                )
                X += len(sublayers) + 1

    ax.set_aspect("equal")
    ax.set_xlim(-3, width + 1)
    ax.set_ylim(-1, n_qubits + 0.5)
    ax.axis("off")

    return fig


def build_reverse_light_cone_circuit(selected_edge, circ):
    """Construct the reverse light-cone circuit skelleton for a given interaction edge
    on a circuit constituted by single-qubit rotations and entangling layers
    for visualization purposes:

    .. math::

       U_1 U_{\\mathrm{ent}} U_1 U_{\\mathrm{ent}} \\cdots |\\text{initial product state}\\rangle

    e.g. for QAOA: :math:`U_x U_{zz} U_x U_{zz} |+\\rangle`.

    This function extracts the light cone of a selected two-qubit
    interaction term (Pauli string with weight 2) from a full circuit and
    reconstructs it as an explicit :quimb-api:`Circuit` instance.
    The resulting circuit contains only the gates that
    causally influence the selected edge, ordered by their round.

    The reconstruction is performed by parsing tensor tags from the reverse
    light-cone TN representation and re-applying the corresponding
    single- and two-qubit gates to a new circuit.

    Parameters
    ----------
    selected_edge : tuple[int, int]
        The edge (pair of qubit indices) for which the light cone is constructed.

    circ : :quimb-api:`Circuit`
        Quantum circuit.

    Returns
    -------
    circ_revlc : :quimb-api:`Circuit`
        A new circuit containing only the gates in the reverse light cone of
        ``selected_edge``, acting on the same number of qubits as ``circ``.

    Notes
    -----
    - Gate information is recovered from tensor tags.
    - Gate parameters are set to zero when reconstructing the circuit, as the
      function is intended for structural and visualization purposes rather
      than numerical simulation.

    """
    n_qubits = circ.N

    # Get the reverse light cone of the particular edge
    psi_edge = circ.get_psi_reverse_lightcone(where=selected_edge)

    # Build a reverse light cone Circuit instance
    circ_revlc = qtn.Circuit(N=n_qubits)

    for tensor in psi_edge.tensors:
        tags = list(tensor.tags)

        if tensor.shape == (2,):  # edge tensor
            pass

        elif tensor.shape == (2, 2):  # it is a single-qubit gate
            circ_revlc.apply_gate(
                gate_id=next(
                    tag
                    for tag in tags
                    if any(tag.startswith(g) for g in qtn.circuit.ONE_QUBIT_GATES)
                ),
                qubits=[
                    int(re.fullmatch(r"I(\d+)", tag).group(1))
                    for tag in tags
                    if re.fullmatch(r"I(\d+)", tag)
                ],
                params=[0.0] * 3,
                gate_round=next(
                    int(re.fullmatch(r"ROUND_(\d+)", tag).group(1))
                    for tag in tags
                    if re.fullmatch(r"ROUND_(\d+)", tag)
                ),
            )

        elif tensor.shape == (2, 2, 2, 2):  # it is a two-qubit gate
            circ_revlc.apply_gate(
                gate_id=next(
                    tag
                    for tag in tags
                    if any(tag.startswith(g) for g in qtn.circuit.TWO_QUBIT_GATES)
                ),
                qubits=[
                    int(m.group(1))
                    for tag in tags
                    if (m := re.fullmatch(r"I(\d+)", tag))
                ],
                params=[0.0] * 3,
                gate_round=next(
                    int(re.fullmatch(r"ROUND_(\d+)", tag).group(1))
                    for tag in tags
                    if re.fullmatch(r"ROUND_(\d+)", tag)
                ),
            )

        else:
            msg = f"Invalid gate shape: {tensor.shape}"
            raise ValueError(msg)

    return circ_revlc


def draw_layered_expval(selected_edge, circ, *, list_names=None, commutation=True):
    """Draw the tensor-network representation of an expectation value

    .. math::

        \\langle \\Psi | U^\\dagger O_{\\text{edge}} U | \\Psi \\rangle

    This function visualizes the light-cone structure of a quantum circuit,
    ASSUMING that it consists on layers of single- and two-spin rotations, around a two-site observable.
    The circuit is split symmetrically around the observable and drawn from the inside out,
    showing how the light cone grows layer by layer.

    Parameters
    ----------
    selected_edge : iterable of int with length 2
        Pair of qubit indices on which the observable acts.

    circ : :quimb-api:`Circuit`
        Quantum circuit.

    list_names : list, optional
        Labels used for annotating the diagram. Expected structure:

        - ``list_names[0]`` : str
            Label for the product state.
        - ``list_names[1]`` : list of str
            Labels for single-qubit layers (indexed by layer).
        - ``list_names[2]`` : list of str
            Labels for two-qubit layers (indexed by layer).

    commutation : bool, optional
        If the entangling gates commute with themselves when overlapping
        on one qubit (e.g. the RZZ gate), then further simplification
        is carried out at the level of the active qubits of each layer.

    Returns
    -------
    fig : :matplotlib-api:`figure.Figure`

    """
    n_qubits = circ.N
    if len(selected_edge) != 2:
        raise ValueError("Invalid selected_edge")
    circ_revlc = build_reverse_light_cone_circuit(selected_edge, circ)

    gate_rounds = [gate.round for gate in circ_revlc.gates]
    if any(x is None for x in gate_rounds):
        # without this information, no packing of gates within a layer
        raise ValueError("Gate round information required.")
    depth = max(gate_rounds) + 1
    if list_names is None:
        list_names = [[""], [""] * depth, [""] * depth]  # empty labels
    width = 2 * _determine_layout_depth(circ_revlc) - 5
    list_dict_gates_to_sublayers, list_sublayers = assign_sublayers(circ_revlc)

    fig, ax = plt.subplots(figsize=(width, n_qubits))
    ax.set_facecolor("white")

    col_psi = "#e8e8e8"
    col_U1 = "#00bc32"
    col_U2 = "#d30000"
    col_obs = "#2c7ded"
    fontsize = 30

    X_ledge = -4
    X_redge = width + 3

    # Draw the quantum register lines
    ax.hlines(
        np.arange(n_qubits), X_ledge + 1, X_redge, lw=4, color="#A9A9A9", zorder=1
    )

    # Draw the product states on the edges of the network
    for X, is_right_side in [(X_ledge, False), (X_redge - 1, True)]:
        draw_init_product_state(
            ax=ax,
            n_qubits=n_qubits,
            X=X,
            state_label=list_names[0],
            fontsize=fontsize,
            col_face=col_psi,
            is_right_side=is_right_side,
        )

    # Draw the observable (usually a Pauli string, visually equal to a single-spin rotation)
    draw_1_qubit_layer(
        ax=ax,
        n_qubits=n_qubits,
        X=0.5 * width,
        gate_label=r"$\mathcal{O}$",
        fontsize=fontsize,
        col_face=col_obs,
        active_qubits=selected_edge,
    )

    X_l = 0.5 * width
    X_r = 0.5 * width

    if len(circ.gates[-1].qubits) == 1:  # innermost layer is single spin rotations
        # Loop over layers
        active_qubits = set(selected_edge)
        for layer in range(depth):
            # start from the last layer (the one coupling to the observable)
            rev_layer = depth - layer - 1

            X_l -= 2
            X_r += 2

            # Single-spin rotations
            for X in [X_l, X_r]:
                draw_1_qubit_layer(
                    ax=ax,
                    n_qubits=n_qubits,
                    X=X,
                    gate_label=list_names[1][rev_layer],
                    fontsize=fontsize,
                    col_face=col_U1,
                    active_qubits=active_qubits,
                )

            X_l -= len(list_sublayers[rev_layer]) + 1
            X_r += 2
            # Two-spin rotations
            for X, reverse in [(X_l, False), (X_r, True)]:
                draw_2_qubit_layer(
                    ax=ax,
                    n_qubits=n_qubits,
                    X=X,
                    sublayers=list_sublayers[rev_layer],
                    dict_sublayer=list_dict_gates_to_sublayers[rev_layer],
                    gate_label=list_names[2][layer],
                    fontsize=fontsize,
                    col_face=col_U2,
                    active_qubits=active_qubits,
                    reverse=reverse,
                )

            X_r += len(list_sublayers[rev_layer]) - 1

            # List of qubits featuring in the light cone;
            # `draw_1_qubit_layer` and `draw_2_qubit_layer` require
            # to "blind" the rotations on unactive qubits
            if commutation:
                new_active_qubits = active_qubits.copy()
                for i, j in list_dict_gates_to_sublayers[rev_layer]:
                    if i in active_qubits or j in active_qubits:
                        new_active_qubits.update((i, j))
                active_qubits.update(new_active_qubits)
            else:
                for i, j in list_dict_gates_to_sublayers[rev_layer]:
                    active_qubits.update((i, j))
    else:  # outtermost layer is single spin rotations
        X_l -= 2
        # Loop over layers
        active_qubits = set()
        for layer in range(depth):
            # start from the last layer (the one coupling to the observable)
            rev_layer = depth - layer - 1
            # this ordering assumes commutation!
            for i, j in list_dict_gates_to_sublayers[rev_layer]:
                active_qubits.update((i, j))

            X_l -= len(list_sublayers[rev_layer]) - 1
            X_r += 2

            # Two-spin rotations
            for X, reverse in [(X_l, False), (X_r, True)]:
                draw_2_qubit_layer(
                    ax=ax,
                    n_qubits=n_qubits,
                    X=X,
                    sublayers=list_sublayers[rev_layer],
                    dict_sublayer=list_dict_gates_to_sublayers[rev_layer],
                    gate_label=list_names[2][layer],
                    fontsize=fontsize,
                    col_face=col_U2,
                    active_qubits=active_qubits,
                    reverse=reverse,
                )

            X_l -= 2
            X_r += len(list_sublayers[rev_layer]) + 1

            # Single-spin rotations
            for X in [X_l, X_r]:
                draw_1_qubit_layer(
                    ax=ax,
                    n_qubits=n_qubits,
                    X=X,
                    gate_label=list_names[1][rev_layer],
                    fontsize=fontsize,
                    col_face=col_U1,
                    active_qubits=active_qubits,
                )

            X_l -= 2

    ax.set_aspect("equal")
    ax.set_xlim(X_ledge, X_redge + 1)
    ax.set_ylim(-1, n_qubits + 0.5)
    ax.axis("off")

    return fig
