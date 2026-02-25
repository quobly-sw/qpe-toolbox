#!/usr/bin/env python3

import numpy as np
import quimb.tensor as qtn
from scipy.linalg import dft

from qpe_toolbox.estimation import iqft, iqft_swapped


def test_qft():
    for n in [2, 4]:
        mat_dft = np.round(1 / np.sqrt(2**n) * dft(2**n), 6)

        circ = qtn.Circuit(n)
        reg = list(range(n))
        circ.apply_gates(iqft(reg))
        mat_iqft = np.round(circ.get_uni().to_dense(), 6)

        assert np.max(abs(mat_dft - mat_iqft)) < 1e-6


def test_gate_count():
    for m in range(1, 20):
        c = 0
        c1 = 0
        c2 = 0

        for gate in iqft_swapped(list(range(m))):
            if gate[0] == "H":
                c1 += 1
            elif gate[0] == "CPHASE":
                c2 += 1
            c += 1

        assert c1 == m
        assert c2 == int(m * (m - 1) / 2)
        assert c == int(m + m * (m - 1) / 2)


if __name__ == "__main__":
    test_qft()
    test_gate_count()
