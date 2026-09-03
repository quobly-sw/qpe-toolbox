# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------
"""Custom two-qubit gates registered with quimb's gate registry."""

import autoray
import quimb.tensor as qtn


def su4swap_gate_param_gen(params):
    """
    Return the ``SU4SWAP`` gate array, a general two-qubit gate rooted at identity.

    Equivalent to swapping the two output legs of quimb's 15-parameter ``SU4`` gate,
    allows to tend to Id as params go to zero. The gate spans the SU(4) manifolds, but
    it carries a free phase and its determinant is not 1 (not literally an SU(4) gate)

    Parameters
    ----------
    params : array_like
        The 15 real parameters of the underlying ``SU4`` gate.

    Returns
    -------
    array_like
        The gate as a ``(2, 2, 2, 2)`` array, on the same backend as ``params``.
    """
    return autoray.do("transpose", qtn.circuit.su4_gate_param_gen(params), (1, 0, 2, 3))


def register_su4swap_gate():
    """Register the ``SU4SWAP`` gate with quimb's gate registry, if not already present."""
    if "SU4SWAP" not in qtn.circuit.PARAM_GATES:
        qtn.circuit.register_param_gate("SU4SWAP", su4swap_gate_param_gen, 2)


register_su4swap_gate()
