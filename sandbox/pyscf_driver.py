"""
PySCF driver with correlated methods, orbital localization, GA ordering,
double factorization, and BLISS symmetry shift.
"""

# OpenFermion's MolecularData keeps orbital / DF / PySCF data in private
# attributes; this driver intentionally reads and writes them.
# ruff: noqa: SLF001

from functools import partial, reduce

import numpy as np
from deap import algorithms, base, creator, tools
from openfermion import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner
from pyscf import ao2mo, cc, fci, gto, mcscf, scf
from pyscf.lo import boys, cholesky, orth_ao
from quimb.tensor import DMRG2
from scipy.optimize import minimize

from qpe_toolbox.hamiltonian import Hamiltonian


# ----------------------------------------------------------------------
# BLISS symmetry shift utilities
# ----------------------------------------------------------------------
def _symmetry_shift_terms(params, xi_idx, norb):
    """Unpack (k1, k2, xi, N1, N2, F); N2 and F are in chemist (pq|rs)."""
    (k1, k2), xi_vec = params[:2], params[2:]
    if not xi_vec.size:
        xi_vec = np.zeros(len(xi_idx[0]))
    xi = np.zeros((norb, norb))
    xi[xi_idx], xi[xi_idx[::-1]] = xi_vec, xi_vec

    N1 = np.eye(norb)
    # two-body part of N^2 - N is 2 δ_{pq} δ_{rs} (chemist)
    N2 = 2.0 * np.einsum("pq,rs->pqrs", N1, N1)

    # ξ two-body pattern F_{pqrs} = ξ_{pq} δ_{rs} + ξ_{rs} δ_{pq}
    T_ = np.einsum("pq,rs->pqrs", xi, N1)
    F_ = T_ + np.transpose(T_, (2, 3, 0, 1))
    return k1, k2, xi, N1, N2, F_


def _symmetry_shift_two_body_loss(params, two, xi_idx):
    _, k2, _, _, N2, F_ = _symmetry_shift_terms(params, xi_idx, two.shape[0])
    return np.linalg.norm(two - k2 * N2 - F_ / 4)


def _symmetry_shift_one_body_loss(x, one, xi, k2, n_elec, norb):
    """Step-2 loss; only k1 = x[0] is free, ξ and k2 are frozen."""
    k1 = x[0]
    new_one = one - k1 * np.eye(norb) - k2 * np.eye(norb) + (n_elec - 1) * xi / 4
    return np.linalg.norm(new_one)


def symmetry_shift(
    core, one_electron, two_electron, n_elec, method="L-BFGS-B", **method_kwargs
):
    norb = one_electron.shape[0]
    xi_mat = np.zeros((norb, norb))
    xi_idx = np.tril_indices_from(xi_mat)
    params0 = np.hstack(([0.0, 0.0], xi_mat[xi_idx]))

    # Step 1: two-body norm minimization over (k2, ξ)
    cost_two = partial(_symmetry_shift_two_body_loss, two=two_electron, xi_idx=xi_idx)
    res_two = minimize(cost_two, params0, method=method, **method_kwargs)
    _, k2, xi, _, N2, F_ = _symmetry_shift_terms(res_two.x, xi_idx, norb)
    new_two = two_electron - k2 * N2 - F_ / 4

    # Step 2: one-body norm minimization over k1 alone (ξ, k2 frozen)
    cost_one = partial(
        _symmetry_shift_one_body_loss,
        one=one_electron,
        xi=xi,
        k2=k2,
        n_elec=n_elec,
        norb=norb,
    )
    res_one = minimize(cost_one, np.array([0.0]), method=method)
    k1 = float(res_one.x[0])

    new_core = core + k1 * n_elec + k2 * n_elec**2
    new_one = (
        one_electron - k1 * np.eye(norb) - k2 * np.eye(norb) + (n_elec - 1) * xi / 4
    )

    return new_core, new_one, new_two


class PySCFDriver:
    """
    Driver to run PySCF calculations and populate an OpenFermion
    MolecularData object.

    Supported correlated methods:
      - CCSD         (with optional natural-orbital rotation)
      - FCI          (with optional natural-orbital rotation)
      - CASCI        (optionally shrinks the molecule to the CAS effective
                      Hamiltonian via get_h1cas / get_h2cas)

    Additional features:
      - Orbital localization (Cholesky, Boys)
      - GA orbital ordering
      - Double factorization
      - BLISS symmetry shift
    """

    def __init__(
        self,
        geometry,
        basis,
        charge=0,
        multiplicity=1,
        verbose=0,
        *,
        remove_linear_dep=True,
        linear_dep_threshold=1e-6,
        linear_dep_lindep=1e-7,
    ):
        self.geometry = geometry
        self.basis = basis
        self.charge = charge
        self.multiplicity = multiplicity
        self.verbose = verbose
        self.remove_linear_dep = remove_linear_dep
        self.linear_dep_threshold = linear_dep_threshold
        self.linear_dep_lindep = linear_dep_lindep

        # Build PySCF molecule
        self.pyscf_mol = gto.Mole()
        self.pyscf_mol.atom = geometry
        self.pyscf_mol.basis = basis
        self.pyscf_mol.spin = multiplicity - 1
        self.pyscf_mol.charge = charge
        self.pyscf_mol.symmetry = False
        self.pyscf_mol.build()

        # Initialize MolecularData
        self.molecule = MolecularData(
            geometry, basis, multiplicity, charge, description="", filename=""
        )
        self.molecule.n_orbitals = int(self.pyscf_mol.nao_nr())
        self.molecule.n_qubits = 2 * self.molecule.n_orbitals
        self.molecule.nuclear_repulsion = float(self.pyscf_mol.energy_nuc())

        # SCF
        self.pyscf_scf = self._run_scf()
        self.molecule.hf_energy = float(self.pyscf_scf.e_tot)

        self.molecule._pyscf_data = {
            "mol": self.pyscf_mol,
            "scf": self.pyscf_scf,
        }

        # Populate orbital / integral data (full-space HF basis)
        self.molecule._canonical_orbitals = self.pyscf_scf.mo_coeff.astype(float).copy()
        self.molecule.orbital_energies = self.pyscf_scf.mo_energy.astype(float)
        self.molecule.overlap_integrals = self.pyscf_scf.get_ovlp()
        one, two = self._compute_integrals(self.pyscf_scf.mo_coeff)
        self.molecule.one_body_integrals = one
        self.molecule.two_body_integrals = two

        # Track current MO occupancies
        self._mo_occ = np.asarray(self.pyscf_scf.mo_occ, dtype=float).copy()

        # Correlated-method placeholders
        self._ccsd = None
        self._fci = None
        self._fci_ci = None
        self._casci = None
        self._casci_active_space = None
        self.orbital_permutation = None

        # BLISS symmetry shift state
        self._bliss_applied = False
        self._bliss_params = None

    # ------------------------------------------------------------------
    # SCF / integrals
    # ------------------------------------------------------------------
    def _run_scf(self):
        if self.pyscf_mol.spin:
            scf_obj = scf.ROHF(self.pyscf_mol)
        else:
            scf_obj = scf.RHF(self.pyscf_mol)

        if self.remove_linear_dep:
            scf_obj = scf.addons.remove_linear_dep_(
                scf_obj,
                threshold=self.linear_dep_threshold,
                lindep=self.linear_dep_lindep,
            )

        scf_obj.verbose = self.verbose
        scf_obj.run()
        print(
            f"Hartree-Fock energy for {self.molecule.name} "
            f"({self.molecule.n_electrons} electrons) = "
            f"{scf_obj.e_tot:.8f}"
        )
        return scf_obj

    def _compute_integrals(self, mo_coeff):
        """
        Compute one- and two-electron integrals in a given MO basis.
        Two-body tensor is returned in OpenFermion convention:
            two[p,q,r,s] = (ps|qr)   (chemist notation)
        """
        n_orbitals = mo_coeff.shape[1]
        one = reduce(np.dot, (mo_coeff.T, self.pyscf_scf.get_hcore(), mo_coeff))
        one = one.reshape(n_orbitals, n_orbitals).astype(float)

        two_compressed = ao2mo.kernel(self.pyscf_mol, mo_coeff)
        two = ao2mo.restore(1, two_compressed, n_orbitals)
        two = np.asarray(two.transpose(0, 2, 3, 1), order="C")
        return one, two

    # ------------------------------------------------------------------
    # Two update paths for MolecularData
    # ------------------------------------------------------------------
    def _set_molecule_basis(self, mo_coeff_ao, mo_occ=None):
        """
        Full-space rotation: replace MO coefficients and recompute
        one-/two-electron integrals in that basis.  Preserves
        n_orbitals / n_qubits / n_electrons.
        """
        mo_coeff_ao = np.asarray(mo_coeff_ao, dtype=float)
        self.molecule._canonical_orbitals = mo_coeff_ao
        one, two = self._compute_integrals(mo_coeff_ao)
        self.molecule.one_body_integrals = one
        self.molecule.two_body_integrals = two
        self.molecule.orbital_energies = None
        if mo_occ is not None:
            self._mo_occ = np.asarray(mo_occ, dtype=float)
        # Invalidate stale DF factors and BLISS state
        self._bliss_applied = False
        self._bliss_params = None

    def _set_cas_basis_from_casci(self):
        """
        Shrink the MolecularData to the CAS effective Hamiltonian obtained
        from the current CASCI object.

        Updates n_orbitals, n_qubits, n_electrons, nuclear_repulsion,
        one_body_integrals (h1eff), two_body_integrals (h2cas in OF layout),
        and _canonical_orbitals (CAS MO block in AO basis).
        """
        if self._casci is None:
            raise RuntimeError("Run run_casci() first.")

        mc = self._casci
        ncore, ncas = int(mc.ncore), int(mc.ncas)
        nelecas = mc.nelecas
        if isinstance(nelecas, (tuple, list)):
            n_act_elec = int(nelecas[0] + nelecas[1])
        else:
            n_act_elec = int(nelecas)

        # Effective one-body + core constant
        h1eff, ecore = mc.get_h1cas()

        # Two-body in the CAS (chemist (ij|kl))
        h2cas = mc.get_h2cas()
        if h2cas.ndim == 2:  # some versions return packed
            h2cas = ao2mo.restore(1, h2cas, ncas)
        # chemist (ij|kl) -> OpenFermion (ps|qr)
        two_of = np.asarray(h2cas.transpose(0, 2, 3, 1), order="C")

        # Replace full-space metadata with CAS metadata
        self.molecule.n_orbitals = ncas
        self.molecule.n_qubits = 2 * ncas
        self.molecule.n_electrons = n_act_elec
        self.molecule.nuclear_repulsion = float(ecore)
        self.molecule.one_body_integrals = np.asarray(h1eff, dtype=float)
        self.molecule.two_body_integrals = two_of
        self.molecule._canonical_orbitals = np.asarray(
            mc.mo_coeff[:, ncore : ncore + ncas], dtype=float
        )
        self.molecule.orbital_energies = None
        self._mo_occ = None
        self._bliss_applied = False
        self._bliss_params = None

    # ------------------------------------------------------------------
    # Natural orbitals
    # ------------------------------------------------------------------
    def _natural_orbitals_from_result(self, method):
        method = method.lower()
        if method == "ccsd":
            noons, natorbs_ao = mcscf.addons.make_natural_orbitals(self._ccsd)
            return noons, natorbs_ao

        if method == "fci":
            norb = self.molecule.n_orbitals
            nelec = self.molecule.n_electrons
            rdm1 = self._fci.make_rdm1(self._fci_ci, norb, nelec)
        elif method == "casci":
            rdm1 = self._casci.make_rdm1()
        else:
            raise ValueError("method must be 'ccsd', 'fci', or 'casci'")

        eigvals, eigvecs = np.linalg.eigh(rdm1)
        idx = np.argsort(eigvals)[::-1]
        noons = eigvals[idx]
        natorbs_mo = eigvecs[:, idx]
        natorbs_ao = np.dot(self.pyscf_scf.mo_coeff, natorbs_mo)
        return noons, natorbs_ao

    # ------------------------------------------------------------------
    # Correlated methods
    # ------------------------------------------------------------------
    def run_ccsd(self, *, with_lambda=False, natural_orbitals=False):
        """
        Run CCSD and optionally solve lambda equations.

        If natural_orbitals=True, the molecule's MO basis, one- and
        two-body integrals are replaced by the CCSD natural-orbital
        quantities (full-space rotation).
        """
        mycc = cc.CCSD(self.pyscf_scf)
        mycc.verbose = self.verbose
        mycc.run()
        if with_lambda or natural_orbitals:
            mycc.solve_lambda()
        self._ccsd = mycc

        print(
            f"CCSD energy for {self.molecule.name} ({self.molecule.n_electrons} electrons) is {mycc.e_tot}."
        )

        if natural_orbitals:
            noons, natorbs_ao = self._natural_orbitals_from_result("ccsd")
            self._set_molecule_basis(natorbs_ao, mo_occ=noons)
            if self.verbose:
                print("CCSD natural orbitals written into the molecule.")

        return mycc

    def run_fci(self, *, natural_orbitals=False):
        """
        Run FCI.  If natural_orbitals=True, rotate into the FCI NO basis.
        """
        fci_solver = fci.FCI(self.pyscf_mol, self.pyscf_scf.mo_coeff)
        fci_solver.verbose = self.verbose
        e_fci, civec = fci_solver.kernel()
        self._fci = fci_solver
        self._fci_ci = civec

        print(
            f"FCI energy for {self.molecule.name} ({self.molecule.n_electrons} electrons) is {e_fci}."
        )

        if natural_orbitals:
            noons, natorbs_ao = self._natural_orbitals_from_result("fci")
            self._set_molecule_basis(natorbs_ao, mo_occ=noons)
            if self.verbose:
                print("FCI natural orbitals written into the molecule.")

        return fci_solver

    def run_casci(
        self,
        n_active_electrons=None,
        n_active_orbitals=None,
        *,
        use_cas_integrals=True,
        natural_orbitals=False,
    ):
        """
        Run CASCI in a given active space.

        Parameters
        ----------
        n_active_electrons : int or None
            Electrons in the active space.  None -> full space.
        n_active_orbitals : int or None
            Active spatial orbitals.  None -> all remaining (after core).
        use_cas_integrals : bool, default True
            If True, the effective active-space integrals are written into
            the MolecularData object via `_set_cas_basis_from_casci()`, and
            n_orbitals / n_qubits / n_electrons / nuclear_repulsion are
            updated to describe the CAS.  This is the recommended path.
        natural_orbitals : bool, default False
            If True (and use_cas_integrals=False), rotate the full-space
            MO basis into the CASCI natural-orbital basis.  Only used when
            the CAS is the full space, since otherwise the full-space
            rotation is not meaningful without including the core.
        """
        n_total_orbitals = self.molecule.n_orbitals
        n_total_electrons = self.molecule.n_electrons

        if n_active_electrons is None:
            n_active_electrons = n_total_electrons
            n_core_orbitals = 0
            if n_active_orbitals is None:
                n_active_orbitals = n_total_orbitals
        else:
            if (n_total_electrons - n_active_electrons) % 2 != 0:
                raise ValueError(
                    "n_active_electrons must have the same parity as "
                    f"the total electron count ({n_total_electrons})."
                )
            n_core_orbitals = (n_total_electrons - n_active_electrons) // 2
            if n_active_orbitals is None:
                n_active_orbitals = n_total_orbitals - n_core_orbitals

        if n_active_orbitals + n_core_orbitals > n_total_orbitals:
            raise ValueError(
                f"Active space ({n_core_orbitals} core + "
                f"{n_active_orbitals} active) exceeds the "
                f"{n_total_orbitals} available orbitals."
            )

        self._casci_active_space = (n_active_electrons, n_active_orbitals)

        mc = mcscf.CASCI(self.pyscf_scf, n_active_orbitals, n_active_electrons)
        mc.verbose = self.verbose
        mc.kernel()
        self._casci = mc

        print(
            f"CASCI energy for {self.molecule.name} ({n_total_electrons} electrons, "
            f"{n_active_electrons}/{n_active_orbitals} active in {n_active_orbitals} orbitals) is {mc.e_tot}."
        )

        if use_cas_integrals:
            self._set_cas_basis_from_casci()
            if self.verbose:
                print(
                    "CAS integrals written into the molecule: "
                    f"n_orbitals={self.molecule.n_orbitals}, "
                    f"n_electrons={self.molecule.n_electrons}, "
                    f"E_core={self.molecule.nuclear_repulsion:.8f}"
                )

        elif natural_orbitals:
            noons, natorbs_ao = self._natural_orbitals_from_result("casci")
            self._set_molecule_basis(natorbs_ao, mo_occ=noons)
            if self.verbose:
                print("CASCI natural orbitals written into the molecule.")

        return mc

    # ------------------------------------------------------------------
    # Natural orbital wrappers
    # ------------------------------------------------------------------
    def compute_natural_orbitals(self, method):
        """
        Return (noons, natorbs_ao) from a correlated method.  Does not
        mutate the molecule.
        """
        if method is None:
            return (self.pyscf_scf.mo_occ.copy(), self.pyscf_scf.mo_coeff.copy())

        method = method.lower()
        if method == "ccsd" and self._ccsd is None:
            self.run_ccsd(with_lambda=True)
        elif method == "fci" and self._fci is None:
            self.run_fci()
        elif method == "casci" and self._casci is None:
            self.run_casci()
        elif method not in ("ccsd", "fci", "casci"):
            raise ValueError("method must be None, 'ccsd', 'fci', or 'casci'")

        return self._natural_orbitals_from_result(method)

    def transform_to_natural_orbitals(self, method=None):
        """
        Full-space rotation into the natural-orbital basis of `method`.
        """
        noons, natorbs_ao = self.compute_natural_orbitals(method)
        self._set_molecule_basis(natorbs_ao, mo_occ=noons)
        return noons

    # ------------------------------------------------------------------
    # Orbital localisation
    # ------------------------------------------------------------------
    def localize_orbitals_cholesky(self):
        """
        Cholesky localisation of the occupied MO block in the *current* MO
        basis.  Preserves the occupied/virtual separation.
        """
        if self._casci is not None:
            raise RuntimeError(
                "Cholesky localization on the full space is not meaningful "
                "after CASCI.  Localize the CAS block directly if needed."
            )
        if self._mo_occ is None:
            raise RuntimeError(
                "Occupancy vector is not available; cannot identify "
                "occupied orbitals.  Re-run SCF or a correlated method "
                "first."
            )

        mo_coeff = np.asarray(self.molecule._canonical_orbitals, dtype=float)
        occ_mask = self._mo_occ > 0.5
        if not np.any(occ_mask):
            raise RuntimeError("No occupied orbitals found in the current basis.")

        occ_coeff = mo_coeff[:, occ_mask]
        loc_occ = cholesky.cholesky_mos(occ_coeff)

        new_mo = np.array(mo_coeff, dtype=float, copy=True)
        new_mo[:, occ_mask] = loc_occ

        self._set_molecule_basis(new_mo)
        if self.verbose:
            print("Cholesky localization of occupied orbitals completed.")

    def localize_orbitals_boys(self, *, localize_all=False):
        """
        Foster-Boys localization of the *current* MO basis.  After CASCI
        has been run, use `localize_cas_orbitals_boys` instead.
        """
        if self._casci is not None:
            raise RuntimeError(
                "Full-space Boys localization is not meaningful after "
                "CASCI with use_cas_integrals=True.  Use "
                "localize_cas_orbitals_boys() instead."
            )

        mo_coeff = np.asarray(self.molecule._canonical_orbitals, dtype=float)

        if localize_all:
            bopt = boys.Boys(self.pyscf_mol, mo_coeff)
            bopt.init_guess = None
            loc_mo = bopt.kernel()
            new_occ = None
        else:
            if self._mo_occ is None:
                raise RuntimeError(
                    "Occupancy vector is not available; cannot identify "
                    "occupied orbitals."
                )
            occ_mask = self._mo_occ > 0.5
            occ_coeff = mo_coeff[:, occ_mask]
            bopt = boys.Boys(self.pyscf_mol, occ_coeff)
            bopt.init_guess = None
            loc_occ = bopt.kernel()
            loc_mo = np.array(mo_coeff, dtype=float, copy=True)
            loc_mo[:, occ_mask] = loc_occ
            new_occ = self._mo_occ

        self._set_molecule_basis(loc_mo, mo_occ=new_occ)
        if new_occ is None:
            self._mo_occ = None

        if self.verbose:
            metric = self._boys_localization_metric(loc_mo, self.pyscf_scf.get_ovlp())
            scope = "all orbitals" if localize_all else "occupied orbitals"
            print(
                f"Foster-Boys localization of {scope} completed "
                f"(Boys objective = {metric:.6f})."
            )
        return loc_mo

    def localize_cas_orbitals_boys(self):
        """
        Foster-Boys localization restricted to the CAS block, then
        re-extract the effective CAS Hamiltonian.  Requires
        `run_casci(use_cas_integrals=True)`.
        """
        if self._casci is None:
            raise RuntimeError("Run run_casci() first.")

        mc = self._casci
        ncore, ncas = int(mc.ncore), int(mc.ncas)
        cas_coeff = np.asarray(mc.mo_coeff[:, ncore : ncore + ncas], dtype=float)

        bopt = boys.Boys(self.pyscf_mol, cas_coeff)
        bopt.init_guess = None
        loc_cas = bopt.kernel()

        new_mo = np.array(mc.mo_coeff, dtype=float, copy=True)
        new_mo[:, ncore : ncore + ncas] = loc_cas
        mc.mo_coeff = new_mo

        self._set_cas_basis_from_casci()

        if self.verbose:
            metric = self._boys_localization_metric(loc_cas, self.pyscf_scf.get_ovlp())
            print(
                f"Foster-Boys localization of CAS orbitals completed "
                f"(Boys objective = {metric:.6f})."
            )
        return loc_cas

    def _boys_localization_metric(self, mo_coeff, ovlp=None):
        """
        Boys objective f = sum_i |<i|r|i>|^2  for the given MO coefficients.
        Higher = more localized.
        """
        dip = boys.dipole_integral(self.pyscf_mol, mo_coeff)
        centers = np.einsum("xii->ix", dip)
        return float(np.sum(centers**2))

    def orthogonalize_ao_lowdin(self):
        """Lowdin-orthogonalized AO basis (replaces the whole MO basis)."""
        X = orth_ao(self.pyscf_mol, method="lowdin")
        self._set_molecule_basis(X)
        if self.verbose:
            print("Lowdin orthogonalization of AO basis completed.")

    # ------------------------------------------------------------------
    # GA orbital ordering
    # ------------------------------------------------------------------
    def optimize_orbital_ordering_ga(
        self,
        population_size=50,
        n_generations=100,
        cxpb=0.7,
        mutpb=0.2,
        tournsize=10,
        indpb=0.05,
        weight="linear",
        tol=1e-12,
        seed=None,
        *,
        verbose_ga=False,
    ):
        """
        Optimize the ordering of the current MO block using a GA on the
        exchange-integral bandwidth cost.

        Returns
        -------
        perm : ndarray, shape (nmo,)
            Optimal permutation: perm[k] = old orbital placed at new
            position k.
        cost : float
            Final GA fitness value.
        """
        mo = np.asarray(self.molecule._canonical_orbitals, dtype=float)
        nmo = mo.shape[1]

        two_compressed = ao2mo.kernel(self.pyscf_mol, mo)
        eri_4d = ao2mo.restore(1, two_compressed, nmo)
        K = np.abs(np.einsum("ijji->ij", eri_4d))
        np.fill_diagonal(K, 0.0)

        pairs = [
            (i, j, K[i, j])
            for i in range(nmo)
            for j in range(i + 1, nmo)
            if K[i, j] > tol
        ]

        if not pairs:
            raise RuntimeError(
                "All exchange integrals are below tolerance; nothing to optimize."
            )

        def evaluate(perm):
            pos = np.empty(nmo, dtype=np.int64)
            pos[list(perm)] = np.arange(nmo)
            cost = 0.0
            if weight == "quadratic":
                for i, j, k in pairs:
                    d = pos[i] - pos[j]
                    cost += k * d * d
            else:
                for i, j, k in pairs:
                    cost += k * abs(pos[i] - pos[j])
            return (cost,)

        rng = np.random.default_rng(seed)

        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("indices", rng.permutation, nmo)
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, toolbox.indices
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxOrdered)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=indpb)
        toolbox.register("select", tools.selTournament, tournsize=tournsize)

        pop = toolbox.population(n=population_size)
        algorithms.eaSimple(
            pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_generations, verbose=verbose_ga
        )

        best_ind = tools.selBest(pop, 1)[0]
        best_perm = np.asarray(best_ind, dtype=np.int64)
        best_cost = evaluate(best_perm)[0]
        identity_cost = evaluate(np.arange(nmo))[0]

        if self.verbose:
            print(
                f"GA orbital ordering ({weight}): "
                f"cost {best_cost:.6f} "
                f"(identity: {identity_cost:.6f})"
            )

        self.molecule._canonical_orbitals = mo[:, best_perm]

        h1 = self.molecule.one_body_integrals
        h2 = self.molecule.two_body_integrals
        self.molecule.one_body_integrals = np.asarray(
            h1[np.ix_(best_perm, best_perm)], order="C"
        )
        self.molecule.two_body_integrals = np.asarray(
            h2[np.ix_(best_perm, best_perm, best_perm, best_perm)], order="C"
        )

        if self._mo_occ is not None:
            self._mo_occ = self._mo_occ[best_perm]

        self.molecule.orbital_energies = None
        self.orbital_permutation = best_perm
        # Invalidate BLISS state (integrals have been permuted)
        self._bliss_applied = False
        self._bliss_params = None

        return best_perm, best_cost

    # ------------------------------------------------------------------
    # BLISS symmetry shift
    # ------------------------------------------------------------------
    def apply_symmetry_shift(self, method="BFGS", tol=1e-10, **method_kwargs):
        """
        Apply the BLISS symmetry shift to the current Hamiltonian.

        This reduces the one-norm of the Hamiltonian without changing its
        eigenspectrum in the target electron-number sector.  The shift is
        written back into the MolecularData object.

        Parameters
        ----------
        method : str or callable
            Solver method for ``scipy.optimize.minimize``.  Default is
            ``"L-BFGS-B"``.
        tol : float
            Convergence tolerance passed to the optimizer.
        **method_kwargs
            Additional keyword arguments passed to ``scipy.optimize.minimize``.

        Returns
        -------
        tuple
            ``(new_core, new_one, new_two_of)`` where ``new_two_of`` is in
            OpenFermion convention.
        """
        n_orb = self.molecule.n_orbitals
        n_elec = self.molecule.n_electrons
        core = self.molecule.nuclear_repulsion
        one = self.molecule.one_body_integrals

        # Convert OpenFermion (ps|qr) -> chemist (pq|rs)
        two_of = self.molecule.two_body_integrals
        two_chem = np.asarray(two_of.transpose(0, 3, 1, 2), order="C")

        # Record original one-norm for reporting
        orig_one_norm = np.sum(np.abs(two_chem)) + np.sum(np.abs(one)) + abs(core)

        # Call BLISS
        if self.verbose:
            print(f"Running BLISS symmetry shift (n_orb={n_orb}, n_elec={n_elec}) ...")

        new_core, new_one, new_two_chem = symmetry_shift(
            core,
            one,
            two_chem,
            n_elec,
            method=method,
            options={"ftol": tol, "gtol": tol, **method_kwargs.pop("options", {})},
            **method_kwargs,
        )

        # Convert back to OpenFermion convention
        new_two_of = np.asarray(new_two_chem.transpose(0, 2, 3, 1), order="C")

        # Compute new one-norm
        new_one_norm = (
            np.sum(np.abs(new_two_of)) + np.sum(np.abs(new_one)) + abs(new_core)
        )

        # Write back into MolecularData
        self.molecule.nuclear_repulsion = float(new_core)
        self.molecule.one_body_integrals = np.asarray(new_one, dtype=float)
        self.molecule.two_body_integrals = new_two_of

        # Invalidate derived data
        self.molecule.orbital_energies = None
        self._bliss_applied = True
        self._bliss_params = (new_core, new_one, new_two_chem)

        if self.verbose:
            print(
                f"BLISS symmetry shift completed. "
                f"One-norm: {orig_one_norm:.6f} -> {new_one_norm:.6f} "
                f"(reduction: {100 * (1 - new_one_norm / orig_one_norm):.2f}%)"
            )

        return new_core, new_one, new_two_of

    def get_molecular_data(self):
        return self.molecule


# ----------------------------------------------------------------------
# Hamiltonian generation utilities
# ----------------------------------------------------------------------
def get_interaction_operator(mol_data, n_active_electrons=None, n_active_orbitals=None):
    """
    Build the molecular Hamiltonian (InteractionOperator) in an active
    space.  If active-space parameters are None, the full (current)
    MolecularData space is used.
    """
    if n_active_electrons is None:
        n_core_orbitals = 0
        occupied_indices = None
    else:
        n_core_orbitals = (mol_data.n_electrons - n_active_electrons) // 2
        occupied_indices = list(range(n_core_orbitals))

    if n_active_orbitals is None:
        active_indices = None
    else:
        active_indices = list(
            range(n_core_orbitals, n_core_orbitals + n_active_orbitals)
        )

    return mol_data.get_molecular_hamiltonian(
        occupied_indices=occupied_indices,
        active_indices=active_indices,
    )


def get_quimb_hamiltonian(interaction_op, n_qubits, cutoff):
    """Convert an InteractionOperator to a qpe_toolbox Hamiltonian."""
    qubit_op = jordan_wigner(get_fermion_operator(interaction_op))
    term_list = []
    e_const = 0.0
    for term, coeff in qubit_op.terms.items():
        if term == ():
            e_const += coeff
        else:
            pauli_string = "".join([op[1] for op in term]).lower()
            qubit_indices = [op[0] for op in term]
            if abs(coeff) > cutoff:
                term_list.append((coeff, pauli_string, qubit_indices))

    ham = Hamiltonian(term_list, n_qubits)
    ham.e_const = e_const
    return ham


# ----------------------------------------------------------------------
# Example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # geometry = [("H", (0, 0, 0)), ("H", (0, 0, 0.7414))]
    geometry = [("H", (0, 0, 0)), ("F", (0, 0, 0.956))]
    basis = "sto-3g"

    # 1. Driver + SCF
    driver = PySCFDriver(geometry, basis, charge=0, multiplicity=1, verbose=1)

    # 2. BLISS symmetry shift (optional, can be applied before/after DF)
    #    This reduces the one-norm of the Hamiltonian.
    driver.apply_symmetry_shift(method="L-BFGS-B", tol=1e-10)

    # 3. Build the qubit Hamiltonian.
    molecule = driver.get_molecular_data()
    interaction_op = get_interaction_operator(molecule)
    n_qubits = molecule.n_qubits
    mat_H = get_quimb_hamiltonian(interaction_op, n_qubits, cutoff=1e-6)
    mpo_H = mat_H.to_mpo()

    # 4. DMRG
    dmrg = DMRG2(mpo_H)
    dmrg.solve(max_sweeps=64, tol=1e-8, bond_dims=128, cutoffs=1e-12, verbosity=0)
    dmrg_energy = np.real(dmrg.energy) + np.real(mat_H.e_const)
    print("DMRG energy :", np.real(dmrg.energy))
    print("DMRG energy (including E_core):", dmrg_energy)
