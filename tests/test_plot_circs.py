#!/usr/bin/env python3

import os

os.environ["MPLBACKEND"] = "Agg"


import matplotlib.pyplot as plt
import numpy as np
import quimb.tensor as qtn

from qpe_toolbox.circuit import (
    draw_layered_circuit,
    draw_layered_expval,
    generate_brickwall_circuit,
)

tol = 1e-2
rng = np.random.default_rng(37)


def test_drawings():
    depth = 8
    circ = generate_brickwall_circuit(5, depth, "rz", "cx", rng=rng)
    depth = max(gate.round for gate in circ.gates) + 1
    assert isinstance(draw_layered_circuit(circ, max_depth=depth), plt.Figure)
    assert isinstance(draw_layered_expval((1, 2), circ), plt.Figure)

    circ = generate_brickwall_circuit(5, depth, "rz", "cx", rng=rng)
    depth = max(gate.round for gate in circ.gates) + 1
    assert isinstance(draw_layered_circuit(circ, max_depth=depth // 2), plt.Figure)
    assert isinstance(draw_layered_expval((1, 2), circ), plt.Figure)


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


if __name__ == "__main__":
    test_drawings()
    test_draw_expval_single_qubit_innermost()
