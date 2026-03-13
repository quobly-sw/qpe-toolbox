#!/usr/bin/env python3

import os

os.environ["MPLBACKEND"] = "Agg"


import matplotlib.pyplot as plt
import numpy as np

from qpe_toolbox.circuit import (
    deserialize_to_quimb_Circuit,
    draw_layered_circuit,
    draw_layered_expval,
    generate_brickwall_circuit,
    serialize_from_quimb_Circuit,
)

tol = 1e-2
rng = np.random.default_rng(37)


def test_drawings():
    depth = 8
    circ_quimb = generate_brickwall_circuit(5, depth, "rz", "cx", rng=rng)
    circ_dict = serialize_from_quimb_Circuit(circ_quimb)
    circ = deserialize_to_quimb_Circuit(circ_dict)
    depth = max(gate.round for gate in circ.gates) + 1
    assert isinstance(draw_layered_circuit(circ, max_depth=depth), plt.Figure)
    assert isinstance(draw_layered_expval((1, 2), circ), plt.Figure)

    circ_quimb = generate_brickwall_circuit(
        5, depth, "rz", "cx", start_ent=True, rng=rng
    )
    circ_dict = serialize_from_quimb_Circuit(circ_quimb)
    circ = deserialize_to_quimb_Circuit(circ_dict)
    depth = max(gate.round for gate in circ.gates) + 1
    assert isinstance(draw_layered_circuit(circ, max_depth=depth // 2), plt.Figure)
    assert isinstance(draw_layered_expval((1, 2), circ), plt.Figure)


if __name__ == "__main__":
    test_drawings()
