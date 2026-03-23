# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------
"""
QPE Toolbox: A Quantum Phase Estimation Toolbox.

Documentation is available in the doscstrings and online
at https://quobly-sw.github.io/qpe-toolbox/

Subpackages must be explicitly imported:
::

- qpe_toolbox.circuit           --- creation and manipulation of `quimb` circuits.
- qpe_toolbox.estimation        --- perform different flavors of Quantum Phase Estimation.
- qpe_toolbox.hamiltonian       --- class for defining Hamiltonians and interface with `pyscf` for chemistry.
- qpe_toolbox.tensor            --- manipulation of Matrix Product Operators (MPO) and Matrix Product States (MPS).
"""

import enum

__version__ = "1.0.0"


class _Exact(enum.Enum):
    EXACT = "exact"


EXACT = _Exact.EXACT
"""
Sentinel constant for requesting exact computation.

Use ``EXACT`` to replace an approximation in real world quantum computation
(e.g., time evolution or sampling)
"""
