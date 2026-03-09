#!/usr/bin/env python3

import os

os.environ["MPLBACKEND"] = "Agg"


import matplotlib.pyplot as plt
import numpy as np

from qpe_toolbox.circuit.parametrized_circuits import generate_brickwall_quimb
from qpe_toolbox.circuit.plot_circuits import (
    draw_layered_circuit,
    draw_layered_expval,
    rand_high_sat_color,
)
from qpe_toolbox.circuit.serialize_circuits import (
    deserialize_to_quimb_Circuit,
    serialize_from_quimb_Circuit,
)

tol = 1e-2


def test_drawings():
    depth = 8
    circ_quimb = generate_brickwall_quimb(
        5, depth, "rz", "cx", rng=np.random.default_rng(37)
    )
    circ_dict = serialize_from_quimb_Circuit(circ_quimb)
    circ = deserialize_to_quimb_Circuit(circ_dict)
    depth = max(gate.round for gate in circ.gates) + 1
    assert isinstance(draw_layered_circuit(circ, max_depth=depth), plt.Figure)
    assert isinstance(draw_layered_expval((1, 2), circ), plt.Figure)

    circ_quimb = generate_brickwall_quimb(
        5, depth, "rz", "cx", start_ent=True, rng=np.random.default_rng(37)
    )
    circ_dict = serialize_from_quimb_Circuit(circ_quimb)
    circ = deserialize_to_quimb_Circuit(circ_dict)
    depth = max(gate.round for gate in circ.gates) + 1
    assert isinstance(draw_layered_circuit(circ, max_depth=depth // 2), plt.Figure)
    assert isinstance(draw_layered_expval((1, 2), circ), plt.Figure)

    color_arr = rand_high_sat_color()
    assert len(color_arr) == 3
    assert np.sum(color_arr) <= 3


if __name__ == "__main__":
    test_drawings()
