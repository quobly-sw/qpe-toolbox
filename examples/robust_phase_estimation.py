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
# # Robust Phase Estimation
#
# This example introduces the Robust Phase Estimation algorithm; this QPE version requires only a single ancilla/phase qubit. Our implementation is inspired by [J. Günther et al., Phase Estimation with Partially Randomized Time Evolution](https://doi.org/10.1103/ynxb-p2xq).
#
# In this notebook we explain the idea of the algorithm and apply it to simple models: the Heisenberg model with $4$ spins, the H$_2$ molecule in the minimal basis.
#
# We study the Trotter and statistical errors, and check that the RPE algorithm satisfies Heisenberg scaling, i.e. the ability to measure the energy with precision $\varepsilon$ in time $\mathcal{O}(1/\varepsilon)$.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import quimb.tensor as qtn
from pyscf import gto
from tqdm import notebook as tqdm

import qpe_toolbox.estimation as qpe
from qpe_toolbox import EXACT
from qpe_toolbox.hamiltonian import (
    chemistry_hamiltonian,
    do_dmrg,
    heisenberg_hamiltonian,
)

# %%
plt.rcParams.update({"font.size": 12})

# %% [markdown]
# ## Hadamard Test
# The Robust Phase Estimation algorithm relies on the Hadamard test procedure, which we introduce below. Our presentation takes inspiration from [Lin Lin's lecture notes](https://math.berkeley.edu/~linlin/qasc/) and the ["Hadamard test" Wikipedia page](https://en.wikipedia.org/wiki/Hadamard_test).
#
# The goal of the Hadamard test is to compute $\bra{\psi} U \ket{\psi}$ where $U$ is a unitary operator. Since $U$ is generally not Hermitian, it is not an observable; therefore the real and imaginary parts of $\bra{\psi} U \ket{\psi}$ must be measured separately.
#
# The idea is to build a random variable whose expectation value gives the real (resp. imaginary) part of $\bra{\psi} U \ket{\psi}$.
# Consider the following circuit:
#
# <img src="./figures/gunther2025_fig3.png" align="center">
#
# *Taken from Günther et al. arXiv:2503.05647*.
#
# The Hadamard test uses a single auxiliary qubit initially in state $\ket{0}$ and a physical register with $n$ qubits initialized in state $\ket{\psi}$.
# We start by applying the Hadamard gate $H$ to the auxiliary qubit to put it in a superposition state. Then we apply a controlled-$U$ gate to the physical register conditioned on the auxiliary qubit, followed by a rotation (PHASE) gate  $R(\theta)$ and finally another Hadamard gate on the control qubit.
#
# At the end of the circuit we measure the control qubit and define a random variable $\textbf{Z}_\theta$: if the result of the measurement is $\ket{0}$, we output $1$, if the result is $\ket{1}$, we output $-1$. The expectation value of $\textbf{Z}_\theta$ satisfies:
#
# $$ \mathbb{E}\textbf{Z}_\theta = P(0) - P(1) = \mathrm{Re} \left(e^{i\theta} \bra{\psi}U\ket{\psi}\right). $$
#
# We use two special choices of $\theta$:
#
# $$ \theta = 0 \qquad \implies \qquad \mathbb{E}\textbf{Z}_\theta = \mathrm{Re} \left(\bra{\psi}U\ket{\psi}\right). $$
# $$ \theta = -\frac{\pi}{2} \qquad \implies \qquad \mathbb{E}\textbf{Z}_\theta = \mathrm{Im} \left(\bra{\psi}U\ket{\psi}\right). $$
#
#
# Let $\textbf{X}$ and $\textbf{Y}$ be the random variables corresponding to $\theta=0, -\pi/2$ respectively.
# Define $\textbf{Z} = \textbf{X} + i \textbf{Y}.$ Then we get
#
# $$ \mathbb{E}\textbf{Z} = \bra{\psi}U\ket{\psi} $$
#
# Let us first illustrate a simple instance of the Hadamard test with $U = |0\rangle \langle 0| + e^{i\alpha} |1\rangle \langle 1|$ and $|\psi\rangle = |1\rangle$.

# %%
alpha = np.pi / 6
print(f"cos(alpha) = {np.cos(alpha):.4g} and sin(alpha) = {np.sin(alpha):.4g}")

# Hadamard test with theta=0
circ = qtn.Circuit(2)
circ.apply_gate("X", 1)
circ.apply_gate("H", 0)
circ.apply_gate("CPHASE", alpha, 0, 1)
circ.apply_gate("H", 0)

probs = circ.compute_marginal(where=[0])
print(f"Re(<psi|U|psi>) = {probs[0] - probs[1]:.4g}")

# Hadamard test with theta=-pi/2
circ = qtn.Circuit(2)
circ.apply_gate("X", 1)
circ.apply_gate("H", 0)
circ.apply_gate("CPHASE", alpha, 0, 1)
circ.apply_gate("PHASE", -np.pi / 2, 0)
circ.apply_gate("H", 0)

probs = circ.compute_marginal(where=[0])
print(f"Im(<psi|U|psi>) = {probs[0] - probs[1]:.4g}")

# %% [markdown]
# Now take the Heisenberg Hamiltonian with $4$ spins

# %%
n_qubits = 4
H = heisenberg_hamiltonian(n_qubits)
E0, psi0 = do_dmrg(H)

# %% [markdown]
# We run the Hadamard test on the time evolution operator $U = e^{-iHt}$ with the physical register in state $\ket{\psi} = \ket{\psi_0}$. Then
#
# $$ \mathbb{E}\textbf{Z} = e^{-i E_0 t}. $$
#
# The function `run_hadamard_test` runs the Hadamard test and returns $\mathrm{Re}~e^{i\theta} \bra{\psi}U\ket{\psi}$.
#
# Below we estimate $E_0$ by running the Hadamard test for time evolution over a random time $t$.
#
# We first consider exact time evolution.

# %%
rng = np.random.default_rng(42)
t = rng.random()
data_reg = list(range(1, n_qubits + 1))
U = H.get_U_exact(t, data_reg, controls=(0,))

n_shots = EXACT  # exact computation (no sampling)

X = qpe.run_hadamard_test(psi0, U, 0, n_shots)
Y = qpe.run_hadamard_test(psi0, U, -np.pi / 2, n_shots)
Z = X + 1j * Y

print(f"error = {abs(np.angle(Z) / t + E0):.2g}")

# %% [markdown]
# The previous relation defines a function $g: t \to \mathbb{E}\textbf{Z}(t) $.
#
#
# At this stage let us emphasize two points:
#  * In general:
#
# $$ \mathbb{E}\textbf{Z}(t) = \sum_k c_k e^{i E_k t}, \qquad \text{where} \qquad H\ket{\psi_k} = E_k \ket{\psi_k},~c_k = |\langle \psi | \psi_k \rangle|^2.  $$
# In the following, we consider the simplest case $c_0=1$ ($\psi$ is the ground state).
#
#  *  With a QPU emulator like $\texttt{quimb}$, the probabilities $P(0)$, $P(1)$ can be computed exactly. On a real quantum device, these probabilities are estimated from repeated measurements (shots).
# With a finite number of shots $N_{\rm shots}$, we can estimate $g(t)$ by taking the statistical mean over $N_{\rm shots}$ samples:
#
# $$ \bar{\bf Z}(t) = \frac{1}{N_{\rm shots}} \sum_{n=1}^{N_{\rm shots}} {\bf Z}^{(n)} (t). $$
# If $N_{\rm shots}$ shots are used, the statistical error scales as
#
#  $$\mathcal{O}\left(\frac{1}{\sqrt{N_{\rm shots}}}\right).$$
#
#
#
#
#
#

# %%
shots_list = np.array([10, 50, 100, 200, 400, 500, 800, 1000, 1500, 2000])
errors = []
durations = []
rng = np.random.default_rng(42)

for n_shots in tqdm.tqdm(shots_list):
    st = time.time()
    X = qpe.run_hadamard_test(psi0, U, 0, n_shots, rng=rng)
    Y = qpe.run_hadamard_test(psi0, U, -np.pi / 2, n_shots, rng=rng)
    et = time.time() - st

    Z = X + 1j * Y
    error = abs(np.angle(Z) / t + E0)

    errors.append(error)
    durations.append(et)

# %% [markdown]
# The statistical error decreases as $1/\sqrt{N_{\rm shots}}$ while the computation time increases linearly with $N_{\rm shots}$.

# %%
log_prefactor = np.log(errors) + np.log(shots_list) / 2
mean = np.exp(log_prefactor.mean())
std = np.exp(log_prefactor.std())
fig, (ax_e, ax_t) = plt.subplots(nrows=2)
fig.subplots_adjust(hspace=0.4)
ax_e.loglog(
    shots_list,
    mean / np.sqrt(shots_list),
    "--",
    label="$\\propto 1/\\sqrt{N_{\\rm shots}}$",
)
ax_e.fill_between(
    shots_list,
    mean / std / np.sqrt(shots_list),
    mean * std / np.sqrt(shots_list),
    alpha=0.2,
)
ax_e.loglog(shots_list, errors, "-o")

ax_t.plot(shots_list, durations, "-o")
ax_e.legend(loc="lower left")
ax_e.set_xlabel("number of shots")
ax_t.set_xlabel("number of shots")
ax_e.set_ylabel("error (units of J)")
ax_t.set_ylabel("duration (seconds)");

# %% [markdown]
# ## Robust Phase Estimation Algorithm
#
# ### Introduction
#
# In the previous section, we defined a function $g(t) = \mathbb{E}[\mathbf{Z}(t)] = \bra{\psi} U(t) \ket{\psi}$ is computed using the Hadamard test. In this section, we explain how to process this function within the Quantum Phase Estimation algorithm.
#
# Quote from Günther et al. [PRX Quantum 7, 020332](https://doi.org/10.1103/ynxb-p2xq):
# > "If we think of g(t) as a time signal, then the phase estimation routine will constitute a signal processing transformation to compute the lowest frequency of $g(t)$ (corresponding to the energy $E_0$), provided that we have some guarantee on the overlap of $\ket{\psi}$ with the ground state; we assume a lower bound
#   $c_0 \geq \eta$. With appropriate signal processing methods, one can find the value of $E_0$ with accuracy $\varepsilon$ using $M$ circuits with time evolution for
#  times $t_1, . . . , t_M$. This can be done such that the maximal time evolution $t_{\rm max} = \mathrm{max}\{t_1, . . . , t_M\}$ and the total time over all circuit runs $t_{\rm tot} = t_1 + t_2 + · · · + t_M$  both scale as $\varepsilon^{-1}$. This Heisenberg scaling is known to be optimal."
#
# ### Description of the Algorithm
# To get precision $\varepsilon$, the idea of the robust phase estimation algorithm is to consider $M$ different circuits where $M = \lceil \log_2 \varepsilon^{-1} \rceil$ and estimate
#
# $$ g(2^m) = \mathbb{E}[\mathbf{Z}(2^{m})] = \exp(-i 2^{m} E_0) $$
#
# for $m=0,1,..,M-1$. At each step the algorithm computes an estimate $\theta_m$ of $E_0$; the objective is that $\theta_m$ be the best $m$-bit approximation of $E_0$, so that each iteration adds one bit of precision.
#
# **Algorithm 1 (arXiv p.24 / PRX p.20)**
#
# 1. The algorithm is initialized with $\theta_{-1}=0$.
#
# 2. For each $m$:
#
#     - Take $N_{\rm shots}$ samples to compute the average:
#
#     $$ \bar{\bf Z}(2^{m}) = \frac{1}{N_{\rm shots}} \sum_{n=1}^{N_{\rm shots}} {\bf Z}^{(n)} (2^m) $$
#
#
#     - From the outcome we compute $\phi_m = - \arg(\bar{Z}(2^m))$
#
#     - By definition $\phi_m \in (-\pi, \pi]:~\phi_m~$  is an approximation of $2^mE_0$ modulo $2\pi$
#
#     - Given a previous guess $\theta_{m-1}$ for the ground state energy $E_0$, the new energy estimate $\theta_m$ is given by
#
#     $$ \theta_m = 2^{-m} (2\pi k + \phi_m), $$
#     where $k$ is an integer between $0$ and $2^m - 1$ chosen such that $\theta_m $ minimizes the distance
#
#      $$ d(\theta_m, \theta_{m-1}) = \min_{q\in\mathbb{Z}} | \theta_m - \theta_{m-1} + 2\pi q|, $$
#       under the condition $-\pi < \theta_m \leq \pi$.
#
# The algorithm ensures that at each step, $\theta_m$ is indeed the best $m$-bit approximation of $E_0$.
# The following lemma guarantees convergence:
#
# **Lemma B.1. (arXiv p.25 / PRX p.21)**:
# if $d(\phi_m,2^{m}E)<\frac{\pi}3$ for $m=0,1,...,M -1$ then $\theta_m$ is such that $d(\theta_m,E) \leq 2^{-m}\frac{\pi}3$

# %% [markdown]
# To build an intuition, let us plot the distance $d(\theta, \phi)$ as a function of $\theta$ for a given $\phi$ ($ \phi = 3\pi/2$ in the example) and vice-versa (the distance is symmetric by definition).

# %%
thetas = np.linspace(-2 * np.pi, 2 * np.pi, 600)
plt.xticks(
    np.pi * np.arange(-2, 3), [r"$-2\pi$", r"$-\pi$", "$0$", r"$\pi$", r"$2\pi$"]
)
plt.yticks([0, np.pi / 2, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
plt.xlabel(r"$\theta$")

plt.plot(
    thetas, qpe.angular_distance(thetas, 3 * np.pi / 2), label=r"$d(3\pi/2, \theta)$"
)
plt.plot(
    thetas,
    qpe.angular_distance(3 * np.pi / 2, thetas),
    "--",
    label=r"$d(\theta, 3\pi/2)$",
)
plt.legend();

# %% [markdown]
# Let us now illustrate the first two steps of the algorithm for concreteness. We start with $N_{\rm shots}=2$ and consider exact time evolution.

# %%
n_shots = 2
rng = np.random.default_rng(42)

# m = 0
phi_0 = qpe.rpe_get_hadamard_output(H, psi0, 0, EXACT, n_shots, rng=rng)
theta_0 = phi_0

m = 1
phi_1 = qpe.rpe_get_hadamard_output(H, psi0, m, EXACT, n_shots, rng=rng)
# candidate energies, wrapped into (-pi, pi]
possible_phases = (phi_1 + 2 * np.pi * np.arange(2**m)) / 2**m
possible_phases = (possible_phases + np.pi) % (2 * np.pi) - np.pi

# %% [markdown]
# Let us visualize how the different possible phases compare to $\theta_0$

# %%
plt.plot([-np.pi, np.pi], [1, 1], marker="|", markersize=10, color="k")
plt.plot(theta_0, 1, "*", markersize=20, color="r", label=r"$\theta_0$")
plt.plot(possible_phases, [1, 1], "o", markersize=15, label=r"possible $\theta_1$")

plt.text(-1.1 * np.pi, 0.99, r"$-\pi$", fontsize=16)
plt.text(np.pi, 0.99, r"$\pi$", fontsize=16)
plt.text(1.05 * theta_0, 0.99, r"$\theta_0$", fontsize=16, color="r")
plt.axis("off")
plt.legend();

# %% [markdown]
# We compute $\theta_1$ as the closest possible phase to $\theta_0$ and check that the error decreases between the first and second iteration:

# %%
theta_1 = qpe.rpe_update_theta(phi_1, theta_0, m)
print(f"Exact energy {E0 = :.4f}")
print(f"{theta_0 = :.4f},   error = {abs(E0 - theta_0):.4f}")
print(f"{theta_1 = :.4f},   error = {abs(E0 - theta_1):.4f}")

# %% [markdown]
# ### Statistical Precision
#
# We will now run the algorithm and see the influence of statistical noise. The `robust_phase_estimation` function returns the full list of $\theta_m$, $m=0,...,M-1$.
#
# We start with $N_{\rm shots}=1$.

# %%
epsilon = 0.02
M = int(np.ceil(np.log2(1 / epsilon)))
print(f"Target precision epsilon={epsilon}: requires M={M} iterations\n")

n_shots = 1

rng = np.random.default_rng(42)
theta_list = qpe.robust_phase_estimation(
    H, psi0, M, EXACT, n_shots, verbosity=1, rng=rng
)

# %%
plt.semilogy(
    [qpe.angular_distance(theta, E0) for theta in theta_list],
    "-o",
    label=f"$N_{{\\rm shots}}={n_shots}$",
)
plt.semilogy(np.pi / 3 / 2 ** np.arange(M), "k--", label="$2^{-m}~\\pi/3$")
plt.legend()
plt.xlabel("iteration $m$")
plt.ylabel("$d(\\theta_m, E)$");

# %% [markdown]
# We see that for such small systems, with exact time evolution, a single shot gives an estimate that converges. We now increase the number of shots to improve the precision.

# %%
n_shot_list = [1, 2, 3, 4]

rng = np.random.default_rng(42)
for n_shots in n_shot_list:
    theta_list = qpe.robust_phase_estimation(H, psi0, M, EXACT, n_shots, rng=rng)
    plt.semilogy(
        [qpe.angular_distance(theta, E0) for theta in theta_list],
        "-o",
        label=f"$N_{{\\rm shots}}={n_shots}$",
    )

plt.semilogy(np.pi / 3 / 2 ** np.arange(M), "k--", label="$2^{-m}~\\pi/3$")
plt.legend()
plt.xlabel("iteration $m$")
plt.ylabel("$d(\\theta_m, E)$")
plt.title(rf"$\epsilon={epsilon},\; M={M}$");

# %% [markdown]
# ### RPE with Trotter Approximation
#
# We now apply the same algorithm but replace the exact time evolution operator by a second order Trotter approximation.
#
# The `n_steps` argument in the `robust_phase_estimation` function sets the number of Trotter steps for $m=0$. The number of steps is multiplied by $2$ at each iteration to keep the Trotter timestep constant.
#
# The computation will now take longer since the number of gates for the time evolution grows like $2^m$.

# %%
# %%time
print(f"epsilon={epsilon}, M={M}")

n_steps = 1
n_shots = 4
thetas_ttr_list = []

rng = np.random.default_rng(42)
thetas_ttr = qpe.robust_phase_estimation(
    H, psi0, M, n_steps, n_shots, verbosity=1, rng=rng
)

thetas_ttr_list.append(thetas_ttr)

# %%
plt.semilogy(
    [qpe.angular_distance(theta, E0) for theta in thetas_ttr_list[0]],
    "-o",
    label=f"$n_{{\\rm steps}}={n_steps}$",
)
plt.semilogy(np.pi / 3 / 2 ** np.arange(M), "k--", label="$2^{-m}~\\pi/3$")
plt.legend()
plt.title(r"$N_{\rm shots}=4$")
plt.xlabel("iteration $m$")
plt.ylabel("error");

# %% [markdown]
# Let us increase the number of Trotter steps. This will take a few minutes

# %%
# %%time

n_steps = 2
rng = np.random.default_rng(42)
thetas_ttr = qpe.robust_phase_estimation(
    H, psi0, M, n_steps, n_shots, verbosity=1, rng=rng
)
thetas_ttr_list.append(thetas_ttr)

# %%
for i, n_steps in enumerate([1, 2]):
    plt.semilogy(
        [qpe.angular_distance(theta, E0) for theta in thetas_ttr_list[i]],
        "-o",
        label=f"$n_{{\\rm steps}}={n_steps}$",
    )

plt.semilogy(np.pi / 3 / 2 ** np.arange(M), "k--", label="$2^{-m}~\\pi/3$")
plt.legend()
plt.text(4.5, 0.3, "$N_{\\rm shots}=4$")
plt.xlabel("iteration $m$")
plt.ylabel("error");

# %% [markdown]
# ## Heisenberg Scaling
#
# The experimental time is proportional to $N_{\rm shots} \cdot \sum_{m=0}^{M-1}  2^m$, i.e. it scales like $2^M$. The RPE algorithm reaches a precision $\varepsilon$ in $M = \lceil \log_2 \varepsilon^{-1} \rceil$ iterations.
# Hence it achieves Heisenberg scaling: reaching a precision $\varepsilon$ in time $\mathcal{O}(2^M) = \mathcal{O}(1/\varepsilon)$.
#
# Let us illustrate that below: we run the RPE algorithm for various $\varepsilon$ and plot experimental time versus energy error. Here we use exact time evolution for simplicity: in this case the experimental time is exactly $N_{\rm shots} \cdot \sum_{m=0}^{M-1}  2^m$.

# %%
epsilon_values = 0.1 / 2 ** np.arange(11)
n_shots = 5

cost_list = []
res_list = []
rng = np.random.default_rng(42)
for epsilon in epsilon_values:
    M = int(np.ceil(np.log2(1 / epsilon)))
    cost_list.append(sum([n_shots * 2**m for m in range(M)]))
    theta_list = qpe.robust_phase_estimation(H, psi0, M, EXACT, n_shots, rng=rng)
    res_list.append(theta_list[-1])

# %%
xfit = np.array([0.2, 0.1 / 2**15])
plt.loglog(xfit**-2, xfit, "k:", label=r"$t_{tot}=1/\epsilon^2$")
plt.loglog(xfit**-1, xfit, "b:", label=r"$t_{tot}=1/\epsilon$")
plt.loglog(cost_list, epsilon_values, "rd", label=r"target $\epsilon$")
plt.loglog(cost_list, [abs(en - E0) for en in res_list], "-o", label="RPE")

plt.xlabel("Experimental time $t_{tot}$ (number of shots x repetitions)")
plt.ylabel(r"Energy error $\epsilon$")
plt.title("Heisenberg scaling of RPE with exact time evolution")
plt.xlim(10, 2e5)
plt.legend(title="Heisenberg Hamiltonian\n4 spins, $S=1/2$, $J=1$");

# %% [markdown]
# ## Quantum Chemistry Example: Diatomic Hydrogen

# %% [markdown]
# Let us now consider a molecule: we take $H_2$ in the minimal atomic orbital basis STO-3G. The Hamiltonian in qubit form is obtained via a Jordan-Wigner transformation.
# Unlike the previous spin Hamiltonian, the molecular Hamiltonian is non-local: it couples distant qubits.

# %%
mol = gto.M(
    atom=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))],
    basis="STO-3G",
)

H_H2 = chemistry_hamiltonian(
    mol, hf_mode="rhf", encoding="original", do_fci=True, do_ccsd=False
)
E0_H2, psi0_H2 = do_dmrg(H_H2)
print(f"E_DMRG : {E0_H2 + H_H2.e_const:.10f}")

# %% [markdown]
# ### Exact Time Evolution

# %% [markdown]
# The system is small enough for exact exponentiation of the Hamiltonian matrix, and exact time evolution.

# %%
epsilon = 0.02
M = int(np.ceil(np.log2(1 / epsilon)))
n_shot_list = [2, 3, 1000]

# n_shot stays small, result depends a lot on seed
rng = np.random.default_rng(2)
for n_shots in n_shot_list:
    theta_list = qpe.robust_phase_estimation(H_H2, psi0_H2, M, EXACT, n_shots, rng=rng)
    distances = [qpe.angular_distance(theta, E0_H2) for theta in theta_list]
    plt.semilogy(distances, "-o", label=f"$N_{{\\rm shots}}={n_shots}$")

plt.semilogy(np.pi / 3 / 2 ** np.arange(M), "k--", label="$2^{-m}~\\pi/3$")
plt.legend()
plt.title(r"$H_2$ STO-3G - exact time evolution")
plt.xlabel("iteration $m$")
plt.ylabel("$d(\\theta_m, E)$");

# %% [markdown]
# ### Trotterized Time Evolution

# %% [markdown]
# The Trotter decomposition of the molecular Hamiltonian is longer, and simulations become harder. The following run will take a minute...

# %%
# %%time
n_shots = 2
n_steps = 1

rng = np.random.default_rng(42)
thetas_trotter_H2 = qpe.robust_phase_estimation(
    H_H2, psi0_H2, M, n_steps, n_shots, verbosity=1, rng=rng
)
distances_trotter_H2 = [
    qpe.angular_distance(theta, E0_H2) for theta in thetas_trotter_H2
]

# %% [markdown]
# ... and fails to get the desired precision

# %%
plt.semilogy(np.pi / 3 / 2 ** np.arange(M), "k--", label="$2^{-m}~\\pi/3$")
plt.semilogy(distances_trotter_H2, "-o", label=f"$n_{{\\rm steps}}={n_steps}$")
plt.legend()
plt.title(f"$N_{{\\rm shots}}={n_shots}$")
plt.xlabel("iteration $m$")
plt.ylabel("error");

# %% [markdown]
# If you want to reach the expected precision, what would you try to increase first? $N_{\rm shots}$ or $n_{\rm steps}$?

# %% [markdown]
# ### Chemical Accuracy?
#
# The standard for chemical accuracy is $\varepsilon = 10^{-3}$ Ha. Hartrees are the default unit in `pyscf`. We can directly compute the number of iterations required for chemical accuracy:

# %%
epsilon = 0.001
M = int(np.ceil(np.log2(1 / epsilon)))
print(f"Chemical accuracy eps={epsilon} requires M={M} iterations")

# %% [markdown]
# Reaching chemical accuracy requires at least ten iterations, and a sufficient number of shots and Trotter steps.
# If you want to go further, you can first estimate the runtime for $M=10$ and a given number of shots and Trotter steps,
# then with some patience try to run the simulation.

# %%

# %%
