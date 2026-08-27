#!/usr/bin/env python3

import numpy as np
import quimb.tensor as qtn

from qpe_toolbox.circuit import ansatz_circuit_su4, tn_fit


def test_tn_fit():
    rng = np.random.default_rng(42)
    circ = ansatz_circuit_su4(2, 1, param_scaling=1.0, parametrize=False, rng=rng)
    tn1 = circ.psi
    tn2 = qtn.MPS_rand_state(2, 4)
    tn_fit(tn1, tn2, tags="SU4", steps=1000, tol=1e-10)

    ovlp = abs((tn2.H & tn1).contract())
    assert abs(ovlp - 1.0) < 1e-8


if __name__ == "__main__":
    test_tn_fit()
