# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

"""
This subpackage provides a set of functions for creating and manipulating quimb circuits.
"""

from .controls import shift_control_gates
from .gate_count import count_gates, count_gates_by_qb
from .initialization import make_circ, make_circMPS
from .parametrized_circuits import (
    ansatz_circuit,
    generate_brickwall_quimb,
    generate_rand_quimb,
)
from .plot_circuits import draw_layered_circuit, draw_layered_expval
from .qaoa import (
    brute_force_maxcut,
    compute_qaoa_contraction_costs,
    study_optimization_time_costs,
)
from .serialize_circuits import (
    deserialize_to_qiskit_QuantumCircuit,
    deserialize_to_quimb_Circuit,
    deserialize_to_quimb_CircuitMPS,
    dump_quimb_Circuit_to_qasm,
    load_qasm_to_quimb_Circuit,
    serialize_from_quimb_Circuit,
)
