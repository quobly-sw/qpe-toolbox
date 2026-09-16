#!/usr/bin/env python3

import numpy as np

from qpe_toolbox.circuit import su4swap_gate_param_gen


def test_su4swap_close_to_identity():
    # unlike quimb's SU4, SU4SWAP is rooted at the identity: this is what allows
    # ansatz layers to be grown from small parameters without perturbing the circuit
    assert np.allclose(su4swap_gate_param_gen(np.zeros(15)).reshape(4, 4), np.eye(4))

    # scaling one parameter vector down: the deviation must shrink linearly
    rng = np.random.default_rng(42)
    params = rng.standard_normal(15)
    previous = np.inf
    for scale in (1e-2, 1e-3, 1e-4):
        gate = su4swap_gate_param_gen(scale * params).reshape(4, 4)
        deviation = abs(gate - np.eye(4)).max()
        assert deviation < 10 * scale
        assert deviation < previous
        previous = deviation


if __name__ == "__main__":
    test_su4swap_close_to_identity()
