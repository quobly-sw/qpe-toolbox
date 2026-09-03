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
import numpy as np
import quimb as qu
import quimb.tensor as qtn


def su4swap_gate_param_gen(params):
    """
    Return the ``SU4SWAP`` gate array, a general two-qubit gate rooted at identity.

    Built as ``SWAP @ SU4(params)``, reusing quimb's 15-parameter ``SU4`` gate.
    Since quimb's ``SU4`` tends to SWAP as its parameters go to zero, composing with
    ``SWAP`` maps that limit to the identity: ``SU4SWAP(0) = I``. The gate is unitary
    and spans the same manifold as ``SU4`` (``SWAP @ U(4) = U(4)``). Its determinant is
    a free phase on the unit circle, not ``1`` (this is not literally an SU(4) gate); the
    phase is a global phase on the gate and is invariant under any physical loss.

    Parameters
    ----------
    params : array_like
        The 15 real parameters of the underlying ``SU4`` gate.

    Returns
    -------
    array_like
        The gate as a ``(2, 2, 2, 2)`` array, on the same backend as ``params``.
    """
    u4 = autoray.do("reshape", qtn.circuit.su4_gate_param_gen(params), (4, 4))
    swap = autoray.do("array", np.asarray(qu.swap(2)), like=u4)
    m = swap @ u4
    return autoray.do("reshape", m, (2, 2, 2, 2))


def register_su4swap_gate():
    """Register the ``SU4SWAP`` gate with quimb's gate registry, if not already present."""
    if "SU4SWAP" not in qtn.circuit.PARAM_GATES:
        qtn.circuit.register_param_gate("SU4SWAP", su4swap_gate_param_gen, 2)


register_su4swap_gate()
