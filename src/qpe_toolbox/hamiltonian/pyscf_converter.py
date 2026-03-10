# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

# the openfermionpyscf is just for testing purposes
# as we write our own link between openfermion and pyscf
# explained in make_fermionic_hamiltonian function
import numpy as np
import scipy as sp
from openfermion.ops import InteractionOperator
from pyscf import ao2mo, mcscf, scf

# pyscf lib.einsum is almost the same with np.einsum
# output subscripts must be explicit
# pyscf says its more efficient
from pyscf.lib import einsum


def get_integrals_rhf(rhf):
    """Obtain hpq and hpqrs integrals for spin restricted systems

    Parameters
    ----------
    rhf : :pyscf-api:`pyscf.scf.RHF <scf.html>`

    Returns
    -------
    ncas : int
        number of orbitals
    nelec : int
        number of electrons
    constant : float
        the zero-body (constant) term
    hpq : :numpy-api:`ndarray`
        the one-body term
    hpqrs : :numpy-api:`ndarray`
        the two-body term in chemist notation

    """
    ncas = rhf.mo_coeff.shape[1]
    hpq = rhf.mo_coeff.T.conj() @ rhf.get_hcore() @ rhf.mo_coeff
    hpqrs = ao2mo.full(rhf.mol, rhf.mo_coeff, compact=False)
    hpqrs = hpqrs.reshape((ncas, ncas, ncas, ncas))
    return ncas, rhf.mol.nelectron, rhf.mol.energy_nuc(), hpq, hpqrs


# u is spin up
# d is spin down
def get_integrals_uhf(uhf):
    """Obtain hpq and hpqrs integrals for spin unrestricted systems

    Parameters
    ----------
    uhf : :pyscf-api:`pyscf.scf.UHF <scf.html>`

    Returns
    -------
    ncas : int
        number of orbitals
    nelec : int
        number of electrons
    constant : float
        the zero-body (constant) term
    hpq : tuple of :numpy-api:`ndarray`
        the one-body terms (u and d)
    hpqrs : tuple of :numpy-api:`ndarray`
        the two-body terms in chemist notation (uu, ud, and dd)

    """
    mo_u, mo_d = uhf.mo_coeff
    ncas = mo_u.shape[1]
    hcore = uhf.get_hcore()

    h1e_u = mo_u.T.conj() @ hcore @ mo_u
    h1e_d = mo_d.T.conj() @ hcore @ mo_d

    hpqrs_uu = ao2mo.full(uhf.mol, mo_u, compact=False)
    hpqrs_uu = hpqrs_uu.reshape((ncas, ncas, ncas, ncas))
    hpqrs_ud = ao2mo.general(uhf.mol, (mo_u, mo_u, mo_d, mo_d), compact=False)
    hpqrs_ud = hpqrs_ud.reshape((ncas, ncas, ncas, ncas))
    hpqrs_dd = ao2mo.full(uhf.mol, mo_d, compact=False)
    hpqrs_dd = hpqrs_dd.reshape((ncas, ncas, ncas, ncas))

    # due to symmetry, hpqrs_du = hpqrs_ud.transpose((1, 0, 3, 2))

    return (
        ncas,
        uhf.mol.nelectron,
        uhf.mol.energy_nuc(),
        (h1e_u, h1e_d),
        (hpqrs_uu, hpqrs_ud, hpqrs_dd),
    )


# Active space method
def get_integrals_rhf_cas(rhf, ncas, nelecas, *, ncore=None):
    """Obtain hpq and hpqrs integrals for spin restricted systems,
    with parameters to enable active spaces

    Parameters
    ----------
    rhf : :pyscf-api:`pyscf.scf.RHF <scf.html>`
    ncas : int
        number of active orbitals
    nelecas : int or tuple of int
        number of active electrons
    ncore : int or tuple of int or None, default: None
        number of core electrons

    Returns
    -------
    ncas : int
        number of active orbitals
    nelec : int or tuple of int
        number of active electrons
    constant : float
        the zero-body (constant) term
    hpq : :numpy-api:`ndarray`
        the one-body term
    hpqrs : :numpy-api:`ndarray`
        the two-body term in chemist notation

    """
    mc = mcscf.CASCI(rhf, ncas, nelecas, ncore=ncore)
    hpq, ecore = mc.get_h1cas()
    hpqrs = ao2mo.restore("s1", mc.get_h2cas(), ncas)

    return mc.ncas, mc.nelecas, ecore, hpq, hpqrs


def get_integrals_uhf_cas(uhf, ncas, nelecas, *, ncore=None):
    """Obtain hpq and hpqrs integrals for spin unrestricted systems

    Parameters
    ----------
    uhf : :pyscf-api:`pyscf.scf.UHF <scf.html>`
    ncas : int
        number of active orbitals
    nelecas : int or tuple of int
        number of active electrons
    ncore : int or tuple of int or None, default: None
        number of core electrons

    Returns
    -------
    ncas : int
        number of active orbitals
    nelec : int or tuple of int
        number of active electrons
    constant : float
        the zero-body (constant) term
    hpq : tuple of :numpy-api:`ndarray`
        the one-body terms (u and d)
    hpqrs : tuple of :numpy-api:`ndarray`
        the two-body terms in chemist notation (uu, ud, and dd)

    """
    mc = mcscf.UCASCI(uhf, ncas, nelecas, ncore=ncore)
    hpq, ecore = mc.get_h1cas()
    hpqrs = mc.get_h2cas()

    return mc.ncas, mc.nelecas, ecore, hpq, hpqrs


def make_fermionic_hamiltonian_rhf(constant, hpq, hpqrs, *, orbital_major=True):
    """Construct the Hamiltonian, as an :openfermion-ops:`InteractionOperator`,
    using the results of get_integrals_rhf/get_integrals_rhf_cas

    Parameters
    ----------
    constant : float
    hpq : :numpy-api:`ndarray`
    hpqrs : :numpy-api:`ndarray`
        results from get_integrals_rhf/get_integrals_rhf_cas
    orbital_major: bool, default : True
        if True
          * qubit [0, norb) will be for spin up
          * qubit [norb, 2*norb) will be for spin down
        otherwise
          * even qubits (starting from 0) will be for spin up
          * odd qubits will be for spin down

    Returns
    -------
    :openfermion-ops:`InteractionOperator`

    """
    norb = hpq.shape[0]
    nqubit = norb * 2
    if hpq.shape != (norb, norb):
        raise ValueError("Invalid hpq shape")
    if hpqrs.shape != (norb, norb, norb, norb):
        raise ValueError("Invalid hpqrs shape")

    # hpqrs is in chemist ordering, change to qc ordering
    # then divide by 2
    hpqrs_qc = hpqrs.transpose((0, 2, 3, 1)).copy() / 2.0

    one_elec_tensor = np.zeros((nqubit, nqubit), dtype=hpq.dtype)
    two_elec_tensor = np.zeros((nqubit, nqubit, nqubit, nqubit), dtype=hpqrs.dtype)
    if orbital_major:
        up = slice(0, norb)
        dw = slice(norb, nqubit)
    else:
        up = slice(0, nqubit, 2)
        dw = slice(1, nqubit, 2)

    one_elec_tensor[up, up] = hpq
    one_elec_tensor[dw, dw] = hpq

    two_elec_tensor[up, up, up, up] = hpqrs_qc
    two_elec_tensor[up, dw, dw, up] = hpqrs_qc
    two_elec_tensor[dw, up, up, dw] = hpqrs_qc
    two_elec_tensor[dw, dw, dw, dw] = hpqrs_qc

    return InteractionOperator(constant, one_elec_tensor, two_elec_tensor)


def make_fermionic_hamiltonian_uhf(constant, hpq, hpqrs, *, orbital_major=True):
    """Construct the Hamiltonian, as an :openfermion-ops:`InteractionOperator`,
    using the results of get_integrals_uhf/get_integrals_uhf_cas

    Parameters
    ----------
    constant : float
    hpq : tuple of :numpy-api:`ndarray`
    hpqrs : tuple of :numpy-api:`ndarray`
        results from get_integrals_uhf/get_integrals_uhf_cas
    orbital_major: bool, default : True
        if True
          * qubit [0, norb) will be for spin up
          * qubit [norb, 2*norb) will be for spin down
        otherwise
          * even qubits (starting from 0) will be for spin up
          * odd qubits will be for spin down

    Returns
    -------
    :openfermion-ops:`InteractionOperator`

    """
    hpq_u, hpq_d = hpq
    hpqrs_uu, hpqrs_ud, hpqrs_dd = hpqrs

    norb = hpq_u.shape[0]
    nqubit = norb * 2

    if not (hpq_u.shape == hpq_d.shape == (norb, norb)):
        raise ValueError("Invalid shape for hpq")
    if not (
        hpqrs_uu.shape == hpqrs_ud.shape == hpqrs_dd.shape == (norb, norb, norb, norb)
    ):
        raise ValueError("Invalid shape for hpqrs")

    one_elec_tensor = np.zeros((nqubit, nqubit), dtype=hpq_u.dtype)
    two_elec_tensor = np.zeros((nqubit, nqubit, nqubit, nqubit), dtype=hpqrs_uu.dtype)
    if orbital_major:
        up = slice(0, norb)
        dw = slice(norb, nqubit)
    else:
        up = slice(0, nqubit, 2)
        dw = slice(1, nqubit, 2)

    one_elec_tensor[up, up] = hpq_u
    one_elec_tensor[dw, dw] = hpq_d

    two_elec_tensor[up, up, up, up] = hpqrs_uu.transpose((0, 2, 3, 1)) / 2.0
    two_elec_tensor[up, dw, dw, up] = hpqrs_ud.transpose((0, 2, 3, 1)) / 2.0
    two_elec_tensor[dw, up, up, dw] = hpqrs_ud.transpose((2, 0, 1, 3)) / 2.0
    two_elec_tensor[dw, dw, dw, dw] = hpqrs_dd.transpose((0, 2, 3, 1)) / 2.0

    return InteractionOperator(constant, one_elec_tensor, two_elec_tensor)


# can select the get_integrals_rhf/get_integrals_uhf function manually
# or automatically
def make_fermionic_hamiltonian_auto(mf, *, orbital_major=True):
    """Construct the Hamiltonian, as an :openfermion-ops:`InteractionOperator`,
    from an :pyscf-api:`pyscf.scf.RHF <scf.html>`/:pyscf-api:`pyscf.scf.ROHF <scf.html>`/:pyscf-api:`pyscf.scf.UHF <scf.html>` object

    Parameters
    ----------
    mf : :pyscf-api:`pyscf.scf.RHF <scf.html>` or :pyscf-api:`pyscf.scf.ROHF <scf.html>` or :pyscf-api:`pyscf.scf.UHF <scf.html>`
    orbital_major: bool, default : True
        if True
          * qubit [0, norb) will be for spin up
          * qubit [norb, 2*norb) will be for spin down
        otherwise
          * even qubits (starting from 0) will be for spin up
          * odd qubits will be for spin down

    Returns
    -------
    :openfermion-ops:`InteractionOperator`

    """
    if isinstance(mf, (scf.hf.RHF, scf.rohf.ROHF)):
        _, _, ecore, hpq, hpqrs = get_integrals_rhf(mf)
        return make_fermionic_hamiltonian_rhf(
            ecore, hpq, hpqrs, orbital_major=orbital_major
        )
    if isinstance(mf, scf.uhf.UHF):
        _, _, ecore, hpq, hpqrs = get_integrals_uhf(mf)
        return make_fermionic_hamiltonian_uhf(
            ecore, hpq, hpqrs, orbital_major=orbital_major
        )
    raise NotImplementedError(f"get_integrals for {mf.__class__} not implemented")


# for rhf, input the hpqrs
# for uhf, input the hpqrs_uu, hpqrs_ud, hpqrs_dd separately
def do_sf(hpqrs, *, threshold=1.6e-3):
    """Perform Cholesky decomposition for the hpqrs term"""
    ncas = hpqrs.shape[0]
    n2 = ncas * ncas
    hpqrs = hpqrs.reshape((n2, n2))

    # scipy guarantees order in ascending eval
    eigvals, eigvects = sp.linalg.eigh(hpqrs)

    hpqrs_trunc = np.zeros((n2, n2), dtype=np.float64)
    for idx in range(n2):
        iidx = n2 - idx - 1
        hpqrs_trunc += eigvals[iidx] * np.outer(eigvects[:, iidx], eigvects[:, iidx])
        diff = np.sqrt(np.sum((hpqrs - hpqrs_trunc) ** 2))
        if diff < threshold:
            return idx + 1, diff, hpqrs_trunc.reshape((ncas, ncas, ncas, ncas))
    return n2, 0.0, hpqrs_trunc.reshape((ncas, ncas, ncas, ncas))


def do_df(hpqrs, *, threshold=1.6e-3):
    """Perform double factorization for the hpqrs term"""
    # copied from openfermion.resource_estimates.df.factorize_df
    ncas = hpqrs.shape[0]
    n2 = ncas * ncas
    hpqrs_2 = hpqrs.reshape((n2, n2))

    eigvals, eigvects = sp.linalg.eigh(hpqrs_2)
    L = einsum("ij,j->ij", eigvects, np.sqrt(eigvals))
    L = L.reshape((ncas, ncas, -1))

    sf_rank = L.shape[2]

    neig = 0
    df_factors = []
    rank = 0
    for irank in range(sf_rank):
        Lij = L[:, :, sf_rank - irank - 1]
        eigvals, eigvects = sp.linalg.eigh(Lij)
        normSC = np.sum(np.abs(eigvals))

        truncation = normSC * np.abs(eigvals)

        indices = truncation > threshold
        to_add = np.sum(indices)
        neig += to_add
        rank += 1

        if to_add == 0:
            break

        eval_selected = np.diag(eigvals[indices])
        eigvects_selected = eigvects[:, indices]

        Lij_selected = eigvects_selected @ eval_selected @ eigvects_selected.T
        df_factors.append(Lij_selected)

    df_factors = np.asarray(df_factors).T
    hpqrs_truncated = einsum("ijP,klP->ijkl", df_factors, df_factors)

    diff = np.sqrt(np.sum((hpqrs - hpqrs_truncated) ** 2))

    return neig, rank, df_factors, hpqrs_truncated, diff
