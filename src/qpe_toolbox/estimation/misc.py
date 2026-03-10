# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

import enum


class _Exact(enum.Enum):
    EXACT = "exact"


EXACT = _Exact.EXACT
"""Sentinel constant for requesting exact computation.

Use ``Exact.EXACT`` to replace an approximation in real world quantum computation
by the exact mathematical result.
"""
