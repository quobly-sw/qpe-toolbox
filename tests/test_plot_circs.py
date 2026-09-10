#!/usr/bin/env python3

import os

os.environ["MPLBACKEND"] = "Agg"


import matplotlib.pyplot as plt
import numpy as np
import pytest
import quimb.tensor as qtn

from qpe_toolbox.circuit import (
    draw_layered_circuit,
    draw_layered_expval,
    generate_brickwall_circuit,
)
from qpe_toolbox.circuit.plot_circuits import build_reverse_light_cone_circuit

tol = 1e-2


def test_drawings():
    rng = np.random.default_rng(37)
    circ = generate_brickwall_circuit(5, 8, "rz", "cx", rng=rng)
    depth = max(gate.round for gate in circ.gates) + 1

    fig_full = draw_layered_circuit(circ, max_depth=depth)
    fig_trunc = draw_layered_circuit(circ, max_depth=depth // 2)
    assert isinstance(fig_full, plt.Figure)
    assert isinstance(fig_trunc, plt.Figure)
    # truncation actually drops layers from the drawing
    assert len(fig_trunc.axes[0].patches) < len(fig_full.axes[0].patches)

    assert isinstance(draw_layered_expval((1, 2), circ), plt.Figure)
    plt.close("all")


def _build_ent_then_rot_circuit(n_qubits, depth, rng):
    # per round: a two-qubit CX brickwall then a single-qubit RZ layer, so the
    # innermost (last) layer is single-qubit rotations
    circ = qtn.Circuit(n_qubits)
    for r in range(depth):
        for start in range(2):
            for q in range(start, n_qubits - 1, 2):
                circ.apply_gate("CX", q, q + 1, gate_round=r)
        for q in range(n_qubits):
            circ.apply_gate("RZ", rng.random(), q, gate_round=r)
    return circ


def test_draw_expval_single_qubit_innermost():
    rng = np.random.default_rng(37)
    circ = _build_ent_then_rot_circuit(5, 4, rng)
    assert len(circ.gates[-1].qubits) == 1  # innermost layer is single-qubit
    for commutation in (True, False):
        fig = draw_layered_expval((1, 2), circ, commutation=commutation)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def _label_order(fig, n_qubits, prefix):
    # layer labels are the only texts drawn above the register, one per layer
    labels = [
        (text.get_position()[0], int(text.get_text().removeprefix(prefix)))
        for text in fig.axes[0].texts
        if text.get_position()[1] == n_qubits and text.get_text().startswith(prefix)
    ]
    return [index for _, index in sorted(labels)]


def test_draw_layered_circuit_labels():
    rng = np.random.default_rng(37)
    circ = generate_brickwall_circuit(5, 4, "rz", "cx", rng=rng)
    depth = max(gate.round for gate in circ.gates) + 1

    fig = draw_layered_circuit(
        circ,
        state_label="$+$",
        labels_1qubit=[f"A{i}" for i in range(depth)],
        labels_2qubit=[f"B{i}" for i in range(depth)],
    )
    # both layer kinds are labelled left to right, in layer order
    assert _label_order(fig, circ.N, "A") == list(range(depth))
    assert _label_order(fig, circ.N, "B") == list(range(depth))
    assert sum(text.get_text() == "$+$" for text in fig.axes[0].texts) == circ.N
    plt.close(fig)

    # unlabelled diagrams draw empty strings, not the repr of a container
    fig = draw_layered_circuit(circ)
    assert all(text.get_text() != "['']" for text in fig.axes[0].texts)
    plt.close(fig)

    with pytest.raises(ValueError, match="labels_2qubit"):
        draw_layered_circuit(circ, labels_2qubit=["B0"])


def test_draw_expval_layer_labels():
    rng = np.random.default_rng(37)
    circ = generate_brickwall_circuit(5, 4, "rz", "cx", rng=rng)
    edge = (1, 2)
    circ_revlc = build_reverse_light_cone_circuit(edge, circ)
    depth = max(gate.round for gate in circ_revlc.gates) + 1
    assert depth > 1  # otherwise the ordering below is vacuous

    fig = draw_layered_expval(
        edge,
        circ,
        labels_1qubit=[f"A{i}" for i in range(depth)],
        labels_2qubit=[f"B{i}" for i in range(depth)],
    )
    # the diagram is drawn outwards from the observable, so both label lists must
    # run in the same direction: down to the innermost layer, then back up
    expected = list(range(depth)) + list(reversed(range(depth)))
    assert _label_order(fig, circ.N, "A") == expected
    assert _label_order(fig, circ.N, "B") == expected
    plt.close(fig)


if __name__ == "__main__":
    test_drawings()
    test_draw_expval_single_qubit_innermost()
    test_draw_layered_circuit_labels()
    test_draw_expval_layer_labels()
