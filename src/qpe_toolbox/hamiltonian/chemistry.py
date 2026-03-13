# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

from openfermion.transforms import jordan_wigner
from pyscf import cc, fci, scf

from .hamiltonian import Hamiltonian
from .pyscf_converter import (
    do_df,
    do_sf,
    get_integrals_rhf,
    get_integrals_uhf,
    make_fermionic_hamiltonian_rhf,
    make_fermionic_hamiltonian_uhf,
)


def chemistry_hamiltonian(
    molecule,
    hf_mode,
    *,
    encoding="original",
    do_fci=False,
    do_ccsd=False,
):
    """
    Build a molecular electronic Hamiltonian mapped to qubits.

    This function performs a PySCF Hartree-Fock calculation, optionally followed
    by FCI and/or CCSD, constructs the second-quantized fermionic Hamiltonian,
    and maps it to a qubit Hamiltonian using the Jordan-Wigner transformation.

    Parameters
    ----------
    molecule : :pyscf-api:`pyscf.gto.Mole <gto.html#pyscf.gto.mole.Mole>`
        PySCF molecule object defining geometry and basis.
    hf_mode : {'rhf', 'uhf'}, optional
        Type of Hartree-Fock calculation. Default is ``'rhf'``.
    encoding : {'original', 'sf', 'df'}, optional
        Encoding of two-body integrals:
        - ``'original'``: full integrals
        - ``'sf'``: single-factorized
        - ``'df'``: double-factorized
    do_fci : bool, optional
        Whether to compute the FCI ground-state energy.
    do_ccsd : bool, optional
        Whether to compute the CCSD energy.

    Returns
    -------
    Hamiltonian
        Qubit Hamiltonian with additional attributes:
        ``norb``, ``nelec``, ``e_hf``, and optionally ``e_fci`` and ``e_ccsd``.
    """
    e_fci, e_ccsd, hf = do_pyscf(molecule, hf_mode, do_fci, do_ccsd)

    hamiltonian = make_qubit_hamiltonian(hf, hf_mode, encoding)
    hamiltonian.norb = hf.mo_coeff.shape[1]
    hamiltonian.nelec = hf.mol.nelectron
    hamiltonian.e_hf = hf.e_tot

    if do_fci:
        hamiltonian.e_fci = e_fci
    if do_ccsd:
        hamiltonian.e_ccsd = e_ccsd

    return hamiltonian


def do_pyscf(molecule, hf_mode, do_fci, do_ccsd):
    """
    Run PySCF electronic-structure calculations.

    Performs a Hartree-Fock calculation, and optionally FCI and CCSD,
    returning the HF object needed to extract molecular integrals.

    Parameters
    ----------
    molecule : :pyscf-api:`pyscf.gto.Mole <gto.html#pyscf.gto.mole.Mole>`
        Molecular system.
    hf_mode : {'rhf', 'uhf'}
        Hartree-Fock flavor.
    do_fci : bool
        Whether to compute FCI energy.
    do_ccsd : bool
        Whether to compute CCSD energy.

    Returns
    -------
    e_fci : float or None
        FCI ground-state energy.
    e_ccsd : float or None
        CCSD ground-state energy.
    hf : :pyscf-api:`pyscf.scf.hf.SCF <scf.html>`
        Hartree-Fock object containing molecular orbitals and integrals.
    """
    if hf_mode.lower() == "rhf":
        hf = scf.RHF(molecule)
    elif hf_mode.lower() == "uhf":
        hf = scf.UHF(molecule)
    else:
        raise ValueError("only RHF or UHF implemented")
    hf.kernel()

    if molecule.verbose > 0:
        print(f"nOrb   : {hf.mo_coeff.shape[1]}")
        print(f"nElec  : {hf.mol.nelectron}")
        print(f"E_HF   : {hf.e_tot:.10f}", flush=True)

    if do_fci:
        ci = fci.FCI(hf)
        e_fci = ci.kernel()[0]
        if molecule.verbose > 0:
            print(f"E_CI   : {e_fci:.10f}", flush=True)
    else:
        e_fci = None

    if do_ccsd:
        ccsd = cc.CCSD(hf)
        ccsd.kernel()
        e_ccsd = ccsd.e_tot
        if molecule.verbose > 0:
            print(f"E_CCSD: {e_ccsd:.10f}", flush=True)
    else:
        e_ccsd = None

    return e_fci, e_ccsd, hf


def make_qubit_hamiltonian(hf, hf_mode, encoding):
    """
    Construct a qubit Hamiltonian from a PySCF Hartree-Fock object.

    Parameters
    ----------
    hf : :pyscf-api:`pyscf.scf.hf.SCF <scf.html>`
        Converged HF object.
    hf_mode : {'rhf', 'uhf'}
        Hartree-Fock type.
    encoding : {'original', 'sf', 'df'}
        Integral encoding scheme.

    Returns
    -------
    Hamiltonian
        Qubit Hamiltonian with constant energy shift stored in ``e_const``.
    """
    if hf_mode.lower() == "rhf":
        ncas, _nelec, ecore, hpq, hpqrs = get_integrals_rhf(hf)
    else:
        ncas, _nelec, ecore, hpq, hpqrs = get_integrals_uhf(hf)

    fermionic_operator = make_fermionic_hamiltonian(
        ncas, ecore, hpq, hpqrs, hf_mode=hf_mode, encoding=encoding
    )
    qubit_operator = jordan_wigner(fermionic_operator)

    terms, e_const = terms_from_openfermion(qubit_operator)
    hamiltonian = Hamiltonian(terms, 2 * ncas)
    hamiltonian.e_const = e_const

    return hamiltonian


def make_fermionic_hamiltonian(
    ncas, ecore, hpq, hpqrs, hf_mode, encoding, *, verbosity=0
):
    """
    Build a fermionic second-quantized Hamiltonian.

    Parameters
    ----------
    ncas : int
        Number of active spatial orbitals.
    ecore : float
        Core (constant) energy contribution.
    hpq : ndarray
        One-body integrals.
    hpqrs : ndarray or tuple of ndarray
        Two-body integrals.
    hf_mode : {'rhf', 'uhf'}
        Hartree-Fock type.
    encoding : {'original', 'sf', 'df'}
        Integral compression scheme.
    verbosity : int, optional
        Verbosity level.

    Returns
    -------
    :openfermion-ops:`FermionOperator`
        Fermionic Hamiltonian operator.
    """
    hf_mode = hf_mode.lower()
    if hf_mode not in ("rhf", "uhf"):
        raise ValueError("Invalid HF mode")

    if verbosity > 0:
        print(f"{encoding.upper()} HAMILTONIAN")

    match encoding.lower():
        case "original":
            if hf_mode == "rhf":
                return make_fermionic_hamiltonian_rhf(ecore, hpq, hpqrs)
            return make_fermionic_hamiltonian_uhf(ecore, hpq, hpqrs)

        case "sf":
            if hf_mode == "rhf":
                rank, diff, hpqrs_sf = do_sf(hpqrs)
                if verbosity > 0:
                    print(f"ncas: {ncas}; rank: {rank}; diff: {diff}")
                return make_fermionic_hamiltonian_rhf(ecore, hpq, hpqrs_sf)

            hpqrs_sf = []
            for g in hpqrs:
                rank, diff, g_sf = do_sf(g.copy())
                if verbosity > 0:
                    print(f"ncas: {ncas}; rank: {rank}; diff: {diff}")
                hpqrs_sf.append(g_sf)
            return make_fermionic_hamiltonian_uhf(ecore, hpq, hpqrs_sf)

        case "df":
            if hf_mode == "rhf":
                neig, rank, _df_factors, hpqrs_df, diff = do_df(hpqrs)
                if verbosity > 0:
                    print(f"ncas: {ncas}; neig: {neig}; rank: {rank}; diff: {diff}")
                return make_fermionic_hamiltonian_rhf(ecore, hpq, hpqrs_df)
            hpqrs_df = []
            for g in hpqrs:
                neig, rank, _df_factors, g_df, diff = do_df(g.copy())
                if verbosity > 0:
                    print(f"ncas: {ncas}; neig: {neig}; rank: {rank}; diff: {diff}")
                hpqrs_df.append(g_df)
            return make_fermionic_hamiltonian_uhf(ecore, hpq, hpqrs_df)

    msg = f"encoding {encoding} not implemented"
    raise ValueError(msg)


def terms_from_openfermion(qubit_operator):
    """
    Convert an OpenFermion qubit operator into quimb Hamiltonian terms.

    This function separates the constant (identity) contribution from
    Pauli-string terms and converts them into the format expected by
    the ``Hamiltonian`` class.

    Parameters
    ----------
    qubit_operator : :openfermion-ops:`QubitOperator`
        Qubit Hamiltonian expressed as a sum of Pauli strings.

    Returns
    -------
    term_list : list of tuple
        List of Hamiltonian terms in quimb format:
        ``(coefficient, pauli_string, qubits)``.
    e_const : float
        Constant energy offset corresponding to the identity term.
    """
    term_list = []
    e_const = 0

    terms_of = qubit_operator.terms
    for tup in terms_of:
        if tup == ():
            e_const += terms_of[tup]
        else:
            coeff = terms_of[tup]
            paulis = "".join([tt[1] for tt in tup])
            qubits = [tt[0] for tt in tup]
            term_list.append((coeff, paulis.lower(), qubits))

    return term_list, e_const
