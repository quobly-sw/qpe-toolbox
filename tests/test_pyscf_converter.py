#!/usr/bin/env python3

import numpy as np
from openfermion.transforms import jordan_wigner
from pyscf import gto, scf

from qpe_toolbox.hamiltonian.pyscf_converter import (
    do_df,
    do_sf,
    get_integrals_rhf,
    get_integrals_rhf_cas,
    get_integrals_uhf,
    get_integrals_uhf_cas,
    make_fermionic_hamiltonian_rhf,
    make_fermionic_hamiltonian_uhf,
)


def test_basics_H2():
    molecule = gto.M(
        atom=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))],
        unit="A",
        basis="sto-3g",
    )
    hf = scf.RHF(molecule)
    hf.verbose = 0
    hf.kernel()
    _ncas, _nelec, ecore, hpq, hpqrs = get_integrals_rhf(hf)

    fermionic_operator = make_fermionic_hamiltonian_rhf(ecore, hpq, hpqrs)
    _qubit_operator = jordan_wigner(fermionic_operator)
    # qubit_operator.terms contains a dictionary of the Pauli terms along with its coefficients
    # the keys are the Pauli strings and the value are the coefficients
    # the pauli terms are written as a tuple
    # eg. X_4 Z_3 Y_1 is written as ((1, 'Y'), (3, 'Z'), (4, 'X'))
    # so far the output is in ascending qubit order but I don't know if that order is guaranteed

    _rank, _diff, hpqrs_sf = do_sf(hpqrs)
    fermionic_operator = make_fermionic_hamiltonian_rhf(ecore, hpq, hpqrs_sf)
    _qubit_operator = jordan_wigner(fermionic_operator)

    _neig, _rank, _df_factors, hpqrs_df, _diff = do_df(hpqrs)
    fermionic_operator = make_fermionic_hamiltonian_rhf(ecore, hpq, hpqrs_df)
    _qubit_operator = jordan_wigner(fermionic_operator)


def test_get_integrals_rhf_cas_full_space():
    # A CASCI over the full orbital space (no frozen core) must reproduce the
    # plain RHF integrals.
    molecule = gto.M(
        atom=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))],
        unit="A",
        basis="sto-3g",
    )
    hf = scf.RHF(molecule)
    hf.verbose = 0
    hf.kernel()

    norb, nelec, e_nuc, hpq, hpqrs = get_integrals_rhf(hf)
    ncas, ncas_nelec, ecore, hpq_cas, hpqrs_cas = get_integrals_rhf_cas(
        hf, norb, nelec, ncore=0
    )

    assert ncas == norb
    assert sum(ncas_nelec) == nelec
    assert np.isclose(ecore, e_nuc)
    assert np.allclose(hpq_cas, hpq)
    assert np.allclose(hpqrs_cas, hpqrs)

    # the CAS integrals feed the documented make_fermionic_hamiltonian_rhf path
    fermionic_operator = make_fermionic_hamiltonian_rhf(ecore, hpq_cas, hpqrs_cas)
    jordan_wigner(fermionic_operator)


def test_get_integrals_uhf_cas_full_space():
    # Full-space UCASCI must reproduce the plain UHF integrals.
    molecule = gto.M(
        atom=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))],
        unit="A",
        basis="sto-3g",
    )
    uhf = scf.UHF(molecule)
    uhf.verbose = 0
    uhf.kernel()

    norb, nelec, e_nuc, hpq, hpqrs = get_integrals_uhf(uhf)
    ncas, ncas_nelec, ecore, hpq_cas, hpqrs_cas = get_integrals_uhf_cas(
        uhf, norb, (nelec // 2, nelec // 2), ncore=0
    )

    assert ncas == norb
    assert sum(ncas_nelec) == nelec
    assert np.isclose(ecore, e_nuc)
    for h_cas, h in zip(hpq_cas, hpq, strict=True):
        assert np.allclose(h_cas, h)
    for g_cas, g in zip(hpqrs_cas, hpqrs, strict=True):
        assert np.allclose(g_cas, g)

    # the CAS integrals feed the documented make_fermionic_hamiltonian_uhf path
    fermionic_operator = make_fermionic_hamiltonian_uhf(ecore, hpq_cas, hpqrs_cas)
    jordan_wigner(fermionic_operator)


if __name__ == "__main__":
    test_basics_H2()
    test_get_integrals_rhf_cas_full_space()
    test_get_integrals_uhf_cas_full_space()
