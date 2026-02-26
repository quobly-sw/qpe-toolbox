#!/usr/bin/env python3

import autoray
import numpy as np
import quimb.tensor as qtn

from qpe_toolbox.circuit.parametrized_circuits import (
    ansatz_circuit,
    ansatz_circuit_su4,
    ansatz_circuit_sym,
)
from qpe_toolbox.hamiltonian import heisenberg_hamiltonian

opt = "auto-hq"

n = 2
H = heisenberg_hamiltonian(n)
H_MPO = H.to_mpo()

depth = 3


def _loss_circ(circ, H_MPO):
    psi = circ.psi
    psiH = psi.H
    c = psiH & psi
    psi.align_(H_MPO, psiH)
    E = psiH & H_MPO & psi  # .full_simplify()
    return autoray.do("real", E.contract(all, optimize=opt)) / autoray.do(
        "real", c.contract(all, optimize=opt)
    )


def _my_circ_optimizer(circ):
    return qtn.TNOptimizer(
        circ,  # the tensor network we want to optimize
        _loss_circ,  # the function we want to minimize
        loss_constants={
            "H_MPO": H_MPO
        },  # supply U to the loss function as a constant TN
        autodiff_backend="jax",  # use 'autograd' for non-compiled optimization
        optimizer="L-BFGS-B",
        progbar=False,
    )


def test_ansatz_circuit():
    circ = ansatz_circuit(n, depth)
    assert len(circ.gates) == depth * ((n - 1) + n)


def test_ansatz_circuit_su4():
    circ = ansatz_circuit_su4(n, depth)
    c = 0
    for g in circ.gates:
        if g.label == "SU4":
            c += 1
    assert c == depth * (n - 1)


def test_ansatz_circuit_sym():
    circ = ansatz_circuit_sym(n, depth, gate_round=0)
    assert circ.gates[0].label == "X"
    assert circ.gates[-1].label == "RZ"


def test_ansatz_circuit_opt():
    circ = ansatz_circuit(n, 1)
    my_circ_optimizer_ = _my_circ_optimizer(circ)
    circ_opt = my_circ_optimizer_.optimize(10)
    circ = ansatz_circuit(n, 2, random_coeff=1e-4)
    circ.set_params(circ_opt.get_params())
    my_circ_optimizer_ = _my_circ_optimizer(circ)
    circ_opt = my_circ_optimizer_.optimize(10)

    assert np.isclose(my_circ_optimizer_.loss, 0.25)


if __name__ == "__main__":
    test_ansatz_circuit()
    test_ansatz_circuit_su4()
    test_ansatz_circuit_sym()
    test_ansatz_circuit_opt()
