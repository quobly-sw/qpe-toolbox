#!/usr/bin/env python3

import numpy as np
import pytest
import quimb as qu

from qpe_toolbox.circuit import trotter_approx_as_MPO
from qpe_toolbox.hamiltonian import heisenberg_hamiltonian

ham = heisenberg_hamiltonian(3)
ham_dense = ham.to_dense()


def _trotter_error(trotter_order, dt):
    exact = qu.expm(-1j * dt * ham_dense)
    approx = trotter_approx_as_MPO(
        ham, dt, trotter_order=trotter_order, cutoff=1e-12, max_bond=64
    ).to_dense()
    return np.linalg.norm(approx - exact)


def test_trotter_invalid_order_raises():
    with pytest.raises(ValueError):
        trotter_approx_as_MPO(ham, 0.1, trotter_order=3, cutoff=1e-12, max_bond=64)


def test_trotter_order_accuracy_ranking():
    # a higher-order product formula must approximate the exact evolution
    # more closely than a lower-order one, at fixed dt
    dt = 0.1
    err1 = _trotter_error(1, dt)
    err2 = _trotter_error(2, dt)
    err4 = _trotter_error(4, dt)
    assert err2 < err1
    assert err4 < err2


def test_trotter_order4_scaling():
    # regression test for the triple-jump composition bug: the global error
    # of the 4th-order product formula scales as dt**5
    dts = [0.2, 0.1, 0.05]
    errs = [_trotter_error(4, dt) for dt in dts]
    slope = np.log(errs[0] / errs[-1]) / np.log(dts[0] / dts[-1])
    assert 4.5 < slope < 5.5


if __name__ == "__main__":
    test_trotter_invalid_order_raises()
    test_trotter_order_accuracy_ranking()
    test_trotter_order4_scaling()
