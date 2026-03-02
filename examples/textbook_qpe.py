# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Textbook QPE
#
# We introduce the Quantum Phase Estimation algorithm and show how to compute the ground state energy of a Hamiltonian $H$. We consider a small system where the exponentiation of the Hamiltonian can be performed exactly to get the exact time evolution operator $U(t) = \exp(-iHt)$.
#
# First, let us briefly introduce the algorithm. For a more detailed introduction, we refer the reader to the famous book by Michael A. Nielsen and Isaac L. Chuang on Quantum Computation and Quantum Information, or to the [Quantum phase estimation algorithm wikipedia page](https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm).
#
# Consider a unitary operator $U$ and an eigenstate $\ket{u}$ of $U$: $U \ket{u} = e^{i \theta} \ket{u}$. We want to measure $\theta$ with $t$-bits precision.
#
# The QPE circuit contains two registers: a physical register with $n$ qubits and a phase register with $m$ qubits, with $m \geq t$.
#
# <img src="./figures/qpe.png" align="center">
#
# 1. The physical register in initially in state $\ket{\psi}$, where $\ket{\psi}$ is an estimate of $\ket{u}$ with fidelity $\Omega = \vert \langle \psi \vert u \rangle \vert^2$.
# 2. The phase register is initially in state $\ket{0}$.
# 3. The circuit starts with a Hadamard wall to put the phase register into a superposition state.
# 4. Then we *encode* the phase into the phase register via a sequence of controlled powers of $U$: $U^{2^k}, k=0,1,...,m-1$ is applied to the physical register, conditioned on the $k$-th phase qubit.
# 5. Finally to *decode* the phase, we apply the inverse Quantum Fourier Transform on the phase register.
# 6. We measure the phase register and find a $t$-bits approximation to $\theta$ with probability $> \Omega (1-\alpha)$ provided $m = t + \lceil \log(2+1/\alpha) \rceil$.
# 7. After the measure, the physical register has been projected onto $\ket{u}$.
#
# The notebook is organised as follows:
# In the first section, we illustrate and detail the different parts of the algorithm on a small 1D Heisenberg Hamiltonian.
# In the second section, we study the precision and success probability of the algorithm in more details.
# Finally in the third section we focus on the influence of the initial overlap $\Omega$.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from quimb.tensor import MatrixProductState
from tqdm import notebook as tqdm

import qpe_toolbox.estimation as qpe
from qpe_toolbox.circuit import make_circ
from qpe_toolbox.hamiltonian import do_dmrg, heisenberg_hamiltonian

# %% [markdown]
# ## Quantum phase estimation
# ### Example : 1D Heisenberg Hamiltonian
#
# First let us define a simple Hamiltonian which we can
# - diagonalize exactly
# - encode as a quantum circuit
#
# Consider Heisenberg 1D Hamiltonian
#
# $$ H = J \sum_{k=0}^{L-1} \vec{S}_k \vec{S}_{k+1} $$
# where $S_k = \sigma_k/2$ are the $S=1/2$ spin matrices, $\sigma_k$ the Pauli matrices.
#
# We take $J=1$ in the following. All energies are expressed in units of $J$.
#
# #### 1. Hamiltonian definition, circuit initialization

# %% [markdown]
# Define the Hamiltonian, perform exact diagonalization

# %%
n_qbits = 2
h_spin = heisenberg_hamiltonian(n_qbits)

# Get matrix
hamilt_matrix = h_spin.to_dense()

# Diagonalize hamiltonian
eigvals, eigvecs = np.linalg.eigh(hamilt_matrix)
# Ground state
E0 = eigvals[0]
psi0 = eigvecs[:, 0]
print(f"E_ED : {E0:.10f}")

# %%
# Ground state MPS
E0_dmrg, psi0_mps = do_dmrg(h_spin)
print(f"E_DMRG : {E0_dmrg:.10f}")

# %%
F = abs(psi0_mps.H @ MatrixProductState.from_dense(psi0)) ** 2
print(f"1 - |<psi_DMRG|psi_ED>|^2 = {abs(1 - F):.4g}")

# %% [markdown]
# Initialize the QPE circuit with a data register containing $|\psi_0\rangle$ and a phase register with $m=4$ phase qubits, measure the energy from the circuit

# %%
n_phase_bits = 4

psi_target = psi0_mps
initial_circ = make_circ(n_phase_bits, psi_target)

data_reg = list(range(n_phase_bits, n_phase_bits + n_qbits))
print(
    f"measure H = {initial_circ.local_expectation(G=h_spin.to_dense(), where=data_reg):.10f}"
)

# %% [markdown]
# #### First stage of Quantum Phase Estimation Algorithm
#
# See e.g. Nielsen and Chuang.
# - First, initialize the phase register with a "Hadamard wall"
# - Then build the operator $U = \exp(-i H t)$ for a given evolution time $t$ and apply a sequence of gates ctrl-$U^k$ on the qubit-register conditioned on the $k$-th phase qubit. Since $|\psi_0 \rangle$ is an eigenstate of $H$, we have $U |\psi \rangle = \exp(-i2\pi \theta) |\psi \rangle$ with $0 \leq \theta \leq 1$ ($U$ is unitary by hermiticity of $H$). The final state of the phase register is then
#
# $$ \frac{1}{\sqrt{2^m}} \sum_{q=0}^{2^m-1} e^{i2\pi \theta q} |q \rangle$$
# (the data register stays in the state $|\psi_0\rangle$)

# %%
E_target = E0 + 0.2
size_interval = 2

Emax = E_target + size_interval / 2
evolution_time = 2 * np.pi / size_interval
global_phase = Emax * evolution_time

traces, circ = qpe.qpe_first_stage(
    h_spin, initial_circ, evolution_time, "exact", global_phase
)

circ.psi.retag_({f"I{i}": f"I_phase{i}" for i in range(n_phase_bits)})
phase_reg = list(range(n_phase_bits))
psi = circ.psi.copy()
display(psi)

# %%
# Visualize the circuit as a tensor network
### represent controlled-U gates as a single node
for i in [n_phase_bits + 1 + 2 * j for j in range(4)]:
    psi.contract_(tags={f"GATE_{i}"})

psi.draw(
    figsize=(12, 8),
    show_tags=True,
    color={"PSI0", "CU"},
    edge_scale=1,
    layout="kamada_kawai",
    edge_color=True,
)

# %% [markdown]
# #### Second stage: Inverse Fourier Transform
#
# If we suppose that $\theta = 0.\theta_1...\theta_m$, i.e. that $\theta$ may exactly be expressed in $m$ bits, then the previous expression for the state in the phase register corresponds exactly to the QFT of the product state $|\theta_1 ... \theta_m \rangle$.
# Therefore, applying the IQFT and measuring in the computational basis gives $\theta$ exactly.
#
# When $\theta$ does not exactly expressed in $m$ bits, the measurement gives with "large" (see e.g. Wikipedia or Nielsen & Chuang) probability the closest $m$-bits approximation to $\theta$.
#
# With $m$ phase qubits, we get a measure of $\theta$ with error $\varepsilon_\theta = 1/2^m$. Note that the error and depth of the circuit is independent of $n$ the number of "physical" qubits in the data register, i.e. independent of the size of the system.

# %% [markdown]
# #### A note on the evolution time and global phase
#
# $|\psi_0 \rangle$ is an eigenstate of $U$ with eigenvalue $\exp(i 2\pi \theta)$ and an eigenstate of $H$ with eigenvalue $E_0$. Therefore
# $\exp(i2\pi\theta ) = \exp( - i E t)$. This implies
#
# $$E t = 2\pi\theta~\mathrm{mod}~2 \pi.$$
#
# We fix a "gauge choice" for $\theta$ by introducing a global phase $\phi$ in $U$: setting $U = \exp( - i H t + i \phi)$ and the evolution time $t$ such that we exactly have
#
#  $$ - E t + \phi = 2 \pi \theta. $$
#
# If we know some approximation $E_{target}$ of the exact energy $E_0$ up to an error $\Delta$, then by setting
#
# $$t =2\pi/\Delta \qquad \text{and} \qquad \phi = E_{max} t,$$
# then
#
# $$ E_{min} \leq E_0 \leq E_{max} \implies 0 \leq -E_0 t + 2\phi \leq 2\pi.$$
#
# where $E_{max/min} = E_{target} \pm \Delta/2$.
#
# **Useful expression**
#
# Correspondence between the QPE output $\theta$ and energy $E$ for a given set of parameters $E_{target}$ and $\Delta$:
#
# $$\theta=\frac{E_{target} + \Delta/2 - E}{\Delta}$$

# %% [markdown]
# ## Precision of exact QPE
#
# ### An example
# In this example we start with a target energy off by 0.2 : $E_{target} = E_0 + 0.2$. Let us recall that our energy scale has been fixed by defining our Hamiltonian (using $J = 1$ in this example). We search within an interval $\Delta=2$. Measuring $E_0$ thus means measuring
#
# $$\theta=\frac{E_{target} + \Delta/2 - E_0}{\Delta} = 0.6$$
# To measure $\theta$ with error less than $10^{-2}$ requires 5 phase qubits since
#
# $$(0.101)_2 = 1/2 + 1/2^3 = (0.625)_{10}$$
#  $$(0.10011)_2 = 1/2 + 1/2^4 + 1/2^5 = (0.59375)_{10}$$

# %%
E_target = E0 + 0.2
size_interval = 2

print(f"exact theta = {(E_target + size_interval / 2 - E0) / size_interval:.6g}")

n_phase_bits = 5
print(
    f"Precision on theta = {1 / 2**n_phase_bits}, precision on energy = {1 / 2**n_phase_bits * size_interval}\n"
)

circ = make_circ(n_phase_bits=n_phase_bits, psi_mps=psi0_mps)
traces, energy = qpe.qpe_energy(
    h_spin, circ, "exact", E_target, size_interval, verbosity=1
)

print("\nBest guess =", np.real(energy))
print("exact energy =", np.real(E0))
print("theoretical error bound =", size_interval / 2**n_phase_bits)

assert abs(E0 - energy) < size_interval / 2**n_phase_bits

# %% [markdown]
# Check that second best guess is also within error $\Delta / 2^m$ of the exact value

# %%
energy_bis = -size_interval * 0.625 + E_target + size_interval / 2
print("second best guess", energy_bis)
assert abs(E0 - energy_bis < size_interval / 2**n_phase_bits)

# %% [markdown]
# ### Error and success probability
#
# Now let us say we want to see what is the success probability of measuring $\theta$ with say $p=4$ bits of precision,
# depending on the number of phase qubits $m$. Obviously we need $m \geq p$.
#
# - Take a 'worst case scenario' for $p=4$ bits precision, namely, if $k$ is the integer giving the closest $p$ bits approximation closest to $\theta$ and smaller than $\theta$ we take the case:
#
# $$ \theta = k / 2^p + 1/2^{p+1} $$
#
# For $p=4$ bits, we chose:
#
# $$ \theta = 0.5 + \frac{1}{2^5} = 0.53125 $$
#
# One possible choice of parameters is $E_{target} = E_0 + 1/2^p$ and $\Delta = 2$.
#
# - Let us first perform QPE with $m=p=4$ phase qubits

# %%
E_target = E0 + 1 / 2**4
size_interval = 2

print(f"exact theta = {(E_target + size_interval / 2 - E0) / size_interval:.6g}")

# %%
n_phase_bits = 4
print(
    f"Precision on theta = {1 / 2**n_phase_bits}, precision on energy = {1 / 2**n_phase_bits * size_interval}\n"
)

circ = make_circ(n_phase_bits, psi0_mps)
traces, energy = qpe.qpe_energy(
    h_spin, circ, "exact", E_target, size_interval, verbosity=1
)

# %%
prob = 0.40658933
energy_first = -size_interval * 0.5625 + E_target + size_interval / 2

print("\nBest guess =", energy_first, "with proba", prob)
print("exact energy =", E0)

print(f"error = {E0 - energy_first:.5g}")
print(f"size_interval / 2**(p+1) = {size_interval / 2 ** (n_phase_bits + 1)}")

energy_bis = -size_interval * 0.5 + E_target + size_interval / 2
print("\nsecond best guess", energy_bis, "with proba", prob)
print(f"error = {E0 - energy_bis:.5g}")
print(f"size_interval / 2**(p+1) = {size_interval / 2 ** (n_phase_bits + 1)}")

# %% [markdown]
# We find as expected two outputs with same probability
#
# NB: lower bound for success probability (see wikipedia) is: $4/\pi^2 = 0.40528473$

# %% [markdown]
# - Now let's add one more phase qubit

# %%
n_phase_bits = 5
print(
    f"Precision on theta = {1 / 2**n_phase_bits}, precision on energy = {1 / 2**n_phase_bits * size_interval}\n"
)

circ = make_circ(n_phase_bits, psi0_mps)
traces, energy = qpe.qpe_energy(
    h_spin, circ, "exact", E_target, size_interval, verbosity=1
)

# %%
prob = 1.0
print(f"\nBest guess = {energy:.10f} with proba {prob:.10f}")
print(f"exact energy = {E0:.10f}")

print(f"error = {E0 - energy:.10f}")
print(f"size_interval / 2**(p+1) = {size_interval / 2 ** (n_phase_bits)}")


# %% [markdown]
# The output is an exact measure of $\theta$ with probability $p=1$, since $\theta$ has an exact $p+1=5$ bits expression.

# %% [markdown]
# ### Random choice of $\delta$
#
# Let us now consider a more general case where the initial approximation $E_{target}$ is off by a random $\delta$.
#
# Nielsen and Chuang state that to measure $\theta$ with $p$ bits precison with success probability greater than $ 1 - \epsilon $, i.e.  for $m > p+1$ phase bits getting an output $r / 2^m$ such that
#
# $$ p( | r - b | \leq 2^{m-p} - 1 ) \geq 1-\epsilon $$
# where $b$ is the best $m$ bits approximation to $\theta_0$, $\theta_0 = b / 2^m + \delta$,
# requires
#
# $$ m = p + \left\lceil \mathrm{log}_2 \left( 2 + \frac{1}{2\epsilon} \right) \right\rceil $$
#
# - Note that they assume $m > p+1$
#
# - Let's chose $E_{target} - E_0$ randomly in $[0,1[$ and see how the best guess error and best guess probability evolves with $m \geq p$.
#
# - First we slightly modify the way we perform qpe in order to compute this probability
#


# %%
def qpe_with_prob_success(
    hamiltonian,
    psi0,
    theta_exact,
    n_phase_bits,
    E_target,
    size_interval,
    n_precision_bits,
    *,
    verbosity=0,
):
    """Build the circuit and perform the quantum phase estimation algorithm.
    Return the energy, probability and probability of success as defined by Nielsen and Chuang
    """

    assert size_interval > 0
    Emax = E_target + size_interval / 2
    evolution_time = 2 * np.pi / size_interval
    global_phase = Emax * evolution_time
    E_const = 0

    b = np.floor(theta_exact * 2**n_phase_bits)

    # probs = qpe_get_full_probs(hamiltonian, psi0, n_phase_bits, evolution_time, global_phase)
    initial_circ = make_circ(n_phase_bits, psi0)
    _, probs = qpe.qpe_sample(
        hamiltonian, initial_circ, evolution_time, "exact", global_phase
    )

    if verbosity:
        for ind, x in enumerate(
            sorted(enumerate(np.ravel(probs)), key=lambda x: x[1], reverse=True)
        ):
            if ind < 5:
                print(
                    f"{x[0]:b}".zfill(n_phase_bits),
                    f"|{x[0]}>",
                    x[0] / 2**n_phase_bits,
                    x[1],
                    flush=True,
                )
            else:
                break

    prob_success = 0
    if n_precision_bits + 1 < n_phase_bits:
        for x in sorted(enumerate(np.ravel(probs)), key=lambda x: x[1], reverse=True):
            if abs(x[0] - b) < 2 ** (n_phase_bits - n_precision_bits):
                prob_success += x[1]

    max_prob_state_int = np.argmax(probs)
    theta = max_prob_state_int / 2**n_phase_bits

    energy = Emax - 2 * np.pi * theta / evolution_time
    energy += E_const

    return energy, np.max(probs), prob_success


# %%
# number of target precision bits
p = 5
# random choice for delta
rng = np.random.default_rng(seed=42)
size_interval = 2
E_target = E0 + size_interval * rng.random()
# theta_0 in the above text
theta_exact = (E_target + size_interval / 2 - E0) / size_interval
print(f"exact theta = {theta_exact:.6g}")

probs_success = []
probs = []
energies = []
ms = list(range(1, p + 7))

for n_phase_bits in tqdm.tqdm(ms):
    energy, prob, prob_success = qpe_with_prob_success(
        h_spin,
        psi0_mps,
        theta_exact,
        n_phase_bits,
        E_target,
        size_interval,
        n_precision_bits=p,
    )
    probs_success.append(prob_success)
    probs.append(prob)
    energies.append(energy)


# %%
# formula for minimal number of phase bits required to reach given precision
def m_func(p, epsilon):
    return p + np.ceil(np.log2(2 + 1 / (2 * epsilon)))


# %%
fig, axs = plt.subplots(2, 1)
axs[0].plot(ms, energies, "-o")
axs[0].axhline(y=E0, color="k", linestyle="dotted")
tol = size_interval / 2**p
axs[0].fill_between(ms, [E0 - tol], [E0 + tol], alpha=0.1, facecolor="g")
axs[0].axvline(x=p, color="k", linestyle="dotted")
axs[0].set_ylabel("Energy")


eps = 0.1
print(m_func(p, eps))

axs[1].plot(ms, probs, "-o", label="best guess prob")
axs[1].plot(ms[p + ms[0] :], probs_success[p + ms[0] :], "-s", label="sucess prob")
axs[1].axvline(x=p, color="k", linestyle="dotted")
axs[1].axvline(x=m_func(p, eps), color="k", linestyle="dotted")
axs[1].fill_between(ms, [1 - eps], [1], alpha=0.1, facecolor="g")
axs[1].axhline(y=4 / np.pi**2, color="r", linestyle="dotted", label=r"$4/\pi^2$")
axs[1].set_ylabel("prob")
axs[1].legend(loc="lower left");


# %% [markdown]
# ### Performance and accuracy
#
# $E_0$ is of the order of 1 Hartree. Chemical accuracy is defined at 1mHa = 27meV = 300K. Therefore we aim for an error on energy $\simeq 10^{-3} E_0$. In this example we have fixed the energy unit $J=1$, hence we shall aim for an error $10^{-3}$.
#
# Assume we start with a first estimation of $E_0$ with error $0.1$. What is the cost in phase qubits number to lower the error to $10^{-3}$?
#
# We need $\Delta / 2^{m} \leq 10^{-3}$ i.e. $ m \geq \log_2(10^3 \Delta)$

# %%
E_target = E0 + 0.1
size_interval = 2
print("number of phase bits for chem accuracy =", int(np.log2(10**3 * size_interval)))

# %%
ms = list(range(1, 15))
energies = []
probs = []
durations = []

for n_phase_bits in tqdm.tqdm(ms):
    st = time.time()
    initial_circ = make_circ(n_phase_bits, psi0_mps)
    traces, energy = qpe.qpe_energy(
        h_spin, initial_circ, "exact", E_target, size_interval
    )
    et = time.time() - st
    energies.append(energy)
    probs.append(traces["prob"])
    durations.append(traces["ctimes"][-1])

# %%
fig, axs = plt.subplots(3, 1)

fig.suptitle(f"1D Heisenberg with {n_qbits} spins")
axs[0].semilogy(ms, [abs(E - E0) for E in energies], "-o")
axs[0].axhline(y=1e-3, color="k", linestyle="dotted")
axs[0].set_ylabel("error (units of J)")

axs[1].plot(ms, probs, "-o")
axs[1].axhline(y=4 / np.pi**2, color="r", linestyle="dotted")
axs[1].set_ylabel("best guess prob")

axs[2].plot(ms, durations, "-o")
axs[2].set_xlabel("phase qubits number")
axs[2].set_ylabel("duration (sec)")
plt.tight_layout()

# %% [markdown]
# ### Influence of system size (number of spins / physical qubits in the data register)
#
# We go up to 10 spins, which corresponds to a Hilbert space of dimension $2^{10} = 1024$, still within reach of exact diagonalization in a few seconds computation time on the laptop.
# The following cell may take a few minutes to run.

# %%
nqb_list = [4, 7, 10]
ms = [2, 4, 6, 8, 10]

res = {"E0": [], "energies": [], "probs": [], "durations": [], "durations_ed": []}

st0 = time.time()

for n_qbits in tqdm.tqdm(nqb_list):
    h_spin = heisenberg_hamiltonian(n_qbits)

    # Get matrix
    hamilt_qarray = h_spin.to_dense()

    # Diagonalize hamiltonian
    st_ed = time.time()
    eigvals, eigvecs = np.linalg.eigh(hamilt_qarray)
    res["durations_ed"].append(time.time() - st_ed)

    # Ground state
    E0 = eigvals[0]
    res["E0"].append(E0)
    psi0 = eigvecs[:, 0]
    psi0_mps = MatrixProductState.from_dense(psi0)
    E_target = E0 + 0.1
    size_interval = 2

    energies = []
    probs = []
    durations = []
    bond_dims = []
    for n_phase_bits in tqdm.tqdm(ms, leave=False):
        st = time.time()
        initial_circ = make_circ(n_phase_bits, psi0_mps)
        traces, energy = qpe.qpe_energy(
            h_spin, initial_circ, "exact", E_target, size_interval
        )
        et = time.time() - st
        energies.append(energy)
        probs.append(traces["prob"])
        durations.append(traces["ctimes"][-1])

    res["energies"].append(energies)
    res["probs"].append(probs)
    res["durations"].append(durations)

# %%
for ind, n_qbits in enumerate(nqb_list):
    plt.plot(ms, res["energies"][ind], "-o", label=f"$n_{{qb}}=${n_qbits}")
plt.ylabel("Energy")
plt.xlabel("phase qubits number")
plt.legend();

# %% [markdown]
# The energy error and success probability is independent of the number of physical qubits:

# %%
fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

for ind, n_qbits in enumerate(nqb_list):
    axs[0].semilogy(
        ms,
        [abs(E - res["E0"][ind]) for E in res["energies"][ind]],
        "-o",
        label=f"$n_{{qb}}=${n_qbits}",
    )

    axs[1].plot(ms, res["probs"][ind], "-o")

    axs[2].semilogy(ms, res["durations"][ind], "-o")


axs[0].axhline(y=1e-3, color="k", linestyle="dotted")
axs[0].set_ylabel("error (units of J)")
axs[0].legend()
axs[1].axhline(y=4 / np.pi**2, color="r", linestyle="dotted", label=r"$4/\pi^2$")
axs[1].set_ylabel("best guess prob")
axs[2].semilogy(
    ms,
    [res["durations_ed"][ind] for _ in ms],
    linestyle="dotted",
    label=f"ED, $n_{{qb}}=${n_qbits}",
)
axs[2].set_xlabel("phase qubits number")
axs[2].set_ylabel("duration (sec)")
axs[2].legend()
axs[1].legend()
plt.tight_layout()

# %% [markdown]
# ### Influence of $E_{target}$ and $\Delta$
#
# Vary $\Delta$ and $E_{target}$ within an interval $[E_0 - \Delta / 2, E_0 + \Delta/2]$. Outside of this range, we are sure to get errors because $\forall~k \in \mathbb{Z}$, $\forall~\theta \in [0,1]$, $\exp(i 2\pi \theta + i2 k \pi) = \exp(i 2\pi \theta)$.

# %%
n_qbits = 4
h_spin = heisenberg_hamiltonian(n_qbits)


E0, psi0_mps = do_dmrg(h_spin)

n_phase_bits = 10
initial_circ = make_circ(n_phase_bits, psi0_mps)

siz_list = np.arange(0.2 * abs(E0), 4 * abs(E0), 0.9 * abs(E0))


for size_interval in tqdm.tqdm(siz_list):
    assert size_interval > 0
    Etgt_list = np.linspace(E0 - 0.5 * size_interval, E0 + 0.4 * size_interval, 11)
    energies = []
    probs = []
    for E_target in tqdm.tqdm(Etgt_list, leave=False):
        traces, energy = qpe.qpe_energy(
            h_spin, initial_circ, "exact", E_target, size_interval
        )
        energies.append(energy)
        probs.append(traces["prob"])
    plt.plot(
        [x - E0 for x in Etgt_list],
        [x - E0 for x in energies],
        "-o",
        label=rf"$\Delta={size_interval / abs(E0):.2f}E_0$",
    )
plt.xlabel("$E_{target} - E_0$")
plt.ylabel("$E - E_0$")
plt.title(f"{n_phase_bits} phase qubits")
plt.legend();

# %% [markdown]
# The smallest the size $\Delta$ of the search window, the smallest the error, provided $E_0 \in [E_{target}-\Delta/2, E_{target}+\Delta/2]$.

# %% [markdown]
# ## Overlap
#
# So far we had initialized the circuit with $|\psi_0\rangle$. In practice, we don't have a priori access to the exact $|\psi_0\rangle$, but only an approximate state with some error $\alpha$.
# The probability of success of QPE is then proportional to $1 - \alpha$.
#
# For example, we consider the first excited state $\ket{\psi_1}$ and initialize the physical register in state
#
# $$ \sqrt{1-\alpha} \ket{\psi_0} + \sqrt{\alpha} \ket{\psi_1} $$

# %%
# Get matrix
hamilt_qarray = h_spin.to_dense()

# Diagonalize hamiltonian
eigvals, eigvecs = np.linalg.eigh(hamilt_qarray)

# Ground state
E0 = eigvals[0]
psi0 = eigvecs[:, 0]

# First excited
E1 = eigvals[1]
psi1 = eigvecs[:, 1]

# %%
size_interval = 2
E_target = E0 + 0.2  # 1 / 2**5 * size_interval

n_phase_bits = 5
alphas = np.linspace(0, 1, 11)

E_a = []
p_a = []
for alpha in alphas:
    psi_target = np.sqrt(1 - alpha) * psi0 + np.sqrt(alpha) * psi1
    psi_target_mps = MatrixProductState.from_dense(psi_target)

    initial_circ = make_circ(n_phase_bits, psi_target_mps)
    traces_a, energy_a = qpe.qpe_energy(
        h_spin, initial_circ, "exact", E_target, size_interval
    )
    E_a.append(energy_a)
    p_a.append(traces_a["prob"])

# %% [markdown]
# We plot the energy and probability outputs as a function of $\alpha$:

# %%
fig, (ax_e, ax_p) = plt.subplots(2, 1, sharex=True)
ax_e.plot(alphas, E_a, "-o")
ax_e.axhline(y=E0, color="k", linestyle=":", alpha=0.5)
ax_e.axhline(y=E1, color="k", linestyle=":", alpha=0.5)
ax_e.axvline(x=p_a[0] / (p_a[0] + p_a[-1]), color="k", linestyle=":", alpha=0.5)
ax_e.set_ylabel("Energy E")
ax_e.set_yticks([E0, E1], ["$E_0$", "$E_1$"])

ax_p.plot(alphas, p_a, "-o", color="tab:orange")
ax_p.plot(alphas, p_a[-1] * alphas, color="k", linestyle=":", alpha=0.5)
ax_p.plot(alphas, p_a[0] * (1 - alphas), color="k", linestyle=":", alpha=0.5)
ax_p.axvline(x=p_a[0] / (p_a[0] + p_a[-1]), color="k", linestyle=":", alpha=0.5)
ax_p.set_xticks(
    [0, p_a[0] / (p_a[0] + p_a[-1]), 1], ["0", "$\\frac{p(0)}{p(0) + p(1)}$", "1"]
)
ax_p.set_ylabel("Probability p")
ax_p.set_xlabel(r"$\alpha$")
fig.suptitle(
    r"QPE with initial state $\sqrt{1-\alpha} | \psi_0 \rangle + \sqrt{\alpha} | \psi_1 \rangle$"
)

# %% [markdown]
# * When $\alpha=0$ (resp. $\alpha=1$), the physical register is in $\ket{\psi_0}$ (resp. $\ket{\psi_1}$). The energy is close but not equal to $E_0$ (resp. $E_1$) and the probability is $<1$. The energy error and finite probability depend on the number of phase qubits and on the search window parameters $E_{target}$ and $\Delta$.
#
# * Starting from $\alpha=0$ and increasing $\alpha$, the probability decreases linearly: $p(\alpha) = p(0)(1-\alpha),$ while the energy output remains constant and close to $E_0$. This corresponds to a decreasing overlap of the initial state with the ground state.
#
# * There is a crossover for $\alpha^* = p(0)/(p(0) + p(1)),$ where we switch from measuring $E_0$ to measuring $E_1$.
#
# * For $\alpha > \alpha^*$, the probability increases linearly: $p(\alpha) = p(1) \alpha,$ while the energy output remains constant and close to $E_1$, corresponding to an increasing overlap of the initial state with the first excited state.

# %% [markdown]
# To go further, try to start with a state $\sqrt{\alpha} \ket{\psi_0} + \sqrt{\frac{1-\alpha}{N-1}} \sum_{k=1}^N \ket{\psi_k}.$
