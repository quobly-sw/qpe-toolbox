# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chemistry to qubit Hamiltonians
#
# This tutorial describes how to load a molecular Hamiltonian from [$\texttt{pyscf}$](https://pyscf.org/) and convert it to a qubit Hamiltonian object of the $\texttt{qpe-toolbox}$'s `Hamiltonian` class.
# The `Hamiltonian` is designed to be compatible with $\texttt{quimb}$.
#
# Here we show how to use $\texttt{quimb}$ to find the ground state energy with a classical algorithm: the Density-Matrix Renormalization Group (DMRG). In the other tutorials we encode the Hamiltonian spectrum into unitary operators using Trotterization or Block Encoding, and compute the energy with a quantum algorithm: the Quantum Phase Estimation algorithm, using $\texttt{quimb}$'s quantum circuit simulation mode.
#
# This notebook assumes the reader is already familiar with basic concepts of DMRG and Matrix Product States (MPS). For a review, we refer to e.g. [Schollwöck, The density-matrix renormalization group in the age of matrix product states](https://arxiv.org/abs/1008.3477)

# %%
import numpy as np
from IPython.display import display
from pyscf import gto
from quimb.tensor import DMRG2

from qpe_toolbox.hamiltonian import chemistry_hamiltonian

# %% [markdown]
# The `chemistry_hamiltonian` function is built on the `Hamiltonian` class. It takes a molecule described with a [`Mole`](https://pyscf.org/user/gto.html) object from $\texttt{pyscf}$, performs the Hartree-Fock calculation and converts the molecular Hamiltonian into a `Hamiltonian` object describing the qubit Hamiltonian.

# %% [markdown]
# ## $H_2$
#
# Let us start with the $H_2$ molecule in the minimal atomic orbital basis set STO-3G. We give the geometry and basis set  as keyword arguments to the function `pyscf.gto.M` to initialize the molecular structure object (spin and charge can also be specified, see below $O_2$).
# Then we call the `chemistry_hamiltonian`.
#
# * The `hf_mode` parameter describes how spin is assigned to the spatial molecular orbitals in the Hartree Fock calculations. Restricted Hartree-Fock (`"rhf"`) uses the same molecular orbital twice, one for each state in the spin doublet, while [Unrestricted Hartree-Fock](https://en.wikipedia.org/wiki/Unrestricted_Hartree%E2%80%93Fock) (`"uhf"`) uses different molecular orbitals for the different eigenstates of $S_z$. Unrestricted Hartree-Fock is mostly used for open-shell molecules.
# * Optionally, the FCI or CCSD energies can be computed.

# %%
mol = gto.M(
    atom=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))],
    basis="sto-3g",
)

h2_ham_sto3g = chemistry_hamiltonian(mol, "rhf", do_fci=True, do_ccsd=True)
print(f"n_qubits = {h2_ham_sto3g.n_qubits}")

# %% [markdown]
# With this choice of atomic orbital basis set, the molecule is described with $2$ molecular orbitals (a single orbital per atom), giving $4$ qubits when spin is taken into account.
#
# The FCI energy is given by exact diagonalization of the Hamiltonian. It is the best ground state energy estimation within this choice of basis set.
# In chemistry, an energy estimation must reach the chemical accuracy standard, defined to be $1.6 \text{mHa} \approx 500 \text{K}$.
# The HF energy is far from the FCI energy by $20 \text{mHa}$, while the CCSD energy is close within $10^{-6} \text{Ha}$.
#
# However, the basis set itself only gives an approximate description of the true solutions to the Schrödinger equation. Indeed, note that with this small basis set, the FCI energy is $-1.13$ Ha, while reported values in the literature ([here](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.83.2541)) for $H_2$ are around $-1.16$ Ha. Even with exact diagonalization, the accuracy is at best $30 \text{mHa}$ in this basis.
# Hence, reaching chemical accuracy will always mean using much larger basis sets.
# For a more precise discussion on the importance of atomic orbitals basis sets for the quality of energy estimation, see [E.V. Elfving et al., arXiv:2009.12472](https://arxiv.org/pdf/2009.12472).
#
# In the following, we consider the cc-pvdz basis set with $5$ orbitals per Hydrogen atom.

# %%
mol = gto.M(
    atom=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))],
    basis="cc-pvdz",
)

h2_ham = chemistry_hamiltonian(mol, "rhf", do_fci=True, do_ccsd=True)
print(f"n_qubits = {h2_ham.n_qubits}")

# %% [markdown]
# The FCI energy is now $-1.1632 \text{Ha}$, within chemical accuracy compared to the reported value of $-1.1640 \text{Ha}$.
#
# Let us now build the MPO representation of the qubit Hamiltonian $H$ using the `to_mpo` method from the `Hamiltonian` class, and perform a DMRG calculation of the energy using $\texttt{quimb}$'s `DMRG2` solver.

# %%
h2_mpo = h2_ham.to_mpo()
display(h2_mpo)

# %% [markdown]
# Here we use directly $\texttt{quimb}$'s solver to introduce the different settings of a DMRG calculation. Note that the `do_dmrg` function of the `hamiltonian` module performs a basic DMRG calculation with default settings.
#
# In DMRG, the wavefunction is represented as an MPS. The energy is obtained as `(MPS.H) @ MPO @ MPS` where `@` indicates a contraction and `MPS.H` is the hermitian conjugate. The DMRG algorithm optimizes successively the different tensors of the MPS so as to minimize the energy. Once all the tensors have been optimized in a forward pass, the pass is carried backward: this forth-and-back optimization constitutes a full DMRG sweep. The maximum number of sweeps can be set with `max_sweeps`. Compression is performed along the algorithm, controlled by setting a maximal bond dimension and a cutoff. The `bond_dims` and `cutoffs` arguments take either a single value or a sequence of values (if one wants to assign different parameters to different sweeps).
#
# For an introduction on DMRG, see [this presentation](https://tensornetwork.org/mps/algorithms/dmrg/) and references therein.

# %%
# %%time

dmrg = DMRG2(h2_mpo)
dmrg.solve(max_sweeps=16, bond_dims=64, verbosity=1, cutoffs=1e-12);

# %%
e_dmrg = dmrg.energy + h2_ham.e_const
print(abs(h2_ham.e_fci - e_dmrg))

# %% [markdown]
# ## $O_2$
#
# Let us now consider a bigger molecule, dioxygen. We take the minimal STO-3G basis for simplicity.
# Since $O_2$ is an open-shell molecule, we must run Hartree-Fock in unrestricted mode. We must also specify the spin of the molecule since it is non trivial.

# %%
basis = "STO-3G"

mol = gto.M(
    atom=[("O", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.2))],
    basis=basis,
    spin=2,
)

o2_ham = chemistry_hamiltonian(mol, "uhf", do_fci=True, do_ccsd=True)
print(f"n_qubits = {o2_ham.n_qubits}")

# %% [markdown]
# Let us build the Hamiltonian MPO representation and compress it to reduce its bond dimension (hence reducing the cost of running DMRG).

# %%
o2_mpo_original = o2_ham.to_mpo()
display(o2_mpo_original)

# %% [markdown]
# To make sure that we don't lose physical information when compressing, we measure the MPO norm

# %%
norm = o2_mpo_original.norm()
print(norm)

# %% [markdown]
# Let us compress using a cutoff (alternatively, one can set a maximal bond dimension with the option `max_bond`).

# %%
o2_mpo = o2_mpo_original.copy()
o2_mpo.compress(cutoff=1e-6)
display(o2_mpo)

# %% [markdown]
# We then make sure that the compression doesn't affect the norm and measure the overlap with the original uncompressed MPO.
#

# %%
print(o2_mpo.norm() / norm)
print(np.sqrt(o2_mpo.H @ o2_mpo_original) / norm)

# %% [markdown]
# We now run DMRG

# %%
# %%time
dmrg = DMRG2(o2_mpo)
dmrg.solve(max_sweeps=16, bond_dims=[200], verbosity=1, cutoffs=1e-12);

# %% [markdown]
# Below we print the results and computation time for running DMRG in the larger 6-31G basis set.
#

# %%
print(
    "6-31G DMRG energy with max_bond 128 = -149.78252800735535 in 2 hours w MPO bond dim = 660"
)
print(
    "6-31G DMRG energy with max bond 200 = -149.78507049242708 in 1 hour w MPO bond dim = 190"
)
# Current results
print(f"{basis} DMRG energy = {np.real(dmrg.energy) + o2_ham.e_const}")
print(f"{basis} UCCSD energy = {o2_ham.e_ccsd}")

# %% [markdown]
# ## $H_2O$
#
# We are of course not restricted to molecules with a single atomic element. A lot of molecules' descriptions including geometry, spin, charge, symmetry arguments compatible with $\texttt{pyscf}$ can be found on the internet. We are only limited by computational power. In the following we take the water molecule in the minimal STO-3G basis.

# %%
basis = "STO-3G"
mol = gto.M(
    atom=[
        ("O", (0.0, 0.0, 0.117790)),
        ("H", (0.0, 0.755453, -0.471161)),
        ("H", (0.0, -0.755453, -0.471161)),
    ],
    basis=basis,
)

h2o_ham = chemistry_hamiltonian(mol, "rhf", do_fci=True, do_ccsd=True)
print(f"n_qubits = {h2o_ham.n_qubits}")

# %%
h2o_mpo = h2o_ham.to_mpo()
display(h2o_mpo)

# %%
# %%time
dmrg = DMRG2(h2o_mpo)
dmrg.solve(max_sweeps=16, bond_dims=64, verbosity=1, cutoffs=1e-12);

# %%
print(f"===== H2O -- {basis} =====")
print(f"CCSD energy = {h2o_ham.e_ccsd:.10f}")
print(f"FCI energy = {h2o_ham.e_fci:.10f}")
print(f"DMRG energy = {np.real(dmrg.energy + h2o_ham.e_const):.10f}")

# %%
