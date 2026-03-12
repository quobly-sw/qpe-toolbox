# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

"""
This subpackage provides a set of functions for performing different flavors of Quantum
Phase Estimation.
"""

from .hadamard_test import build_hadamard_test_circuit, run_hadamard_test
from .lcu_walk_operator import (
    build_lcu_prepare_mpo,
    build_lcu_prepare_state_mps,
    build_lcu_reflection_mpo,
    build_lcu_select_mpo,
    estimate_lcu_error,
    get_energy_from_lcu_walk_phase,
    get_lcu_weights,
    lcu_select_gates,
    run_qpe_lcu_walk_operator,
)
from .qft import iqft, iqft_swapped
from .quantum_phase_estimation import (
    qpe_energy,
    qpe_first_stage,
    qpe_sample,
    set_search_window,
)
from .robust_phase_estimation import (
    robust_phase_estimation,
    rpe_distance,
    rpe_get_hadamard_output,
    rpe_update_theta,
)
