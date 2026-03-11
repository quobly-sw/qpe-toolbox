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
# # Variational Circuit Preparation

# %% [markdown]
# ## Approximate ground states via tensor-network circuit optimization
#
# We show how to represent an approximate ground state of an Hamiltonian expressed as Matrix-Product-Operator (MPO) as (i) a Matrix-Product-State (MPS), and as (ii) the wavefunction associated with a quantum circuit of finite depth. This reproduces (to some extent) the work of R. Haghshenas et al., *Variational Power of Quantum Circuit Tensor Networks*, [Phys. Rev. X 12, 011047](https://link.aps.org/doi/10.1103/PhysRevX.12.011047) (2022). This can be used to prepare an initial state for Quantum Phase Estimation ([arxiv:2409.11748](https://arxiv.org/abs/2409.11748)).
#
# We also used some syntax and advice from the [Tensor Network Training of Quantum Circuits](https://quimb.readthedocs.io/en/latest/examples/ex_tn_train_circuit.html) example in [$\texttt{quimb}$](https://github.com/jcmgray/quimb)'s documentation.

# %%
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JAX_ENABLE_X64"] = "True"

import autoray
import matplotlib.pyplot as plt
import numpy as np
import quimb.tensor as qtn

from qpe_toolbox.circuit import ansatz_circuit
from qpe_toolbox.estimation import qpe_sample, set_search_window
from qpe_toolbox.hamiltonian import heisenberg_hamiltonian

# %%
opt = "auto-hq"

# %% [markdown]
# ## State preparation

# %% [markdown]
# ### The DMRG algorithm to find an MPS ground state
#
# We first define our model Hamiltonian (Heisenberg chain) on $n=8$ qubits. We know that its ground state can be approximated as a MPS. We use first the DMRG algorithm as a black box to have a reference energy (using a very large bond dimension 100 to have essentially the exact GS energy)

# %%
n = 8
H = heisenberg_hamiltonian(n)
H_MPO = H.to_mpo()
dmrg = qtn.DMRG2(H_MPO, bond_dims=[10, 20, 40, 100], cutoffs=1e-13)
dmrg.solve(tol=1e-6, verbosity=1)
E_exact = np.real(dmrg.energy)
print("Stored reference energy ", E_exact)
H_MPO.draw(
    show_tags=True, show_inds=True, edge_scale=1, layout="kamada_kawai", edge_color=True
)

# %% [markdown]
# ### MPS search via tensor-network differentiation
#
# Let us now write a custom optimizer to find the ground state MPS, effectively performing the same task as in DMRG. The goal is simply to warm up for the circuit optimization that comes later.
#
# When one optimizes the energy $H$ with respect to a variational wave function, we need first to be able to evaluate $\braket{\psi|H|\psi}$ as a tensor contraction

# %%
psi = qtn.MPS_rand_state(n, bond_dim=2)
psiH = psi.H
psi.align_(H_MPO, psiH)
E = psiH & H_MPO & psi
print(E.contract(all, optimize=opt))
E.draw()


# %% [markdown]
# Now we can define a loss function, the energy, to be optimized w.r.t each tensor that makes the MPS $\ket{\psi}$. We use $\texttt{jax}$ as automatic differentiation tools to evaluate the gradients.


# %%
def loss(psi, H_MPO):
    psiH = psi.H
    c = psiH & psi
    psi.align_(H_MPO, psiH)
    E = psiH & H_MPO & psi
    return autoray.do("real", E.contract(all, optimize=opt)) / autoray.do(
        "real", c.contract(all, optimize=opt)
    )


def myoptimizer(chi, H_MPO):
    psi = qtn.MPS_rand_state(n, bond_dim=chi)
    return qtn.TNOptimizer(
        psi,  # the tensor network we want to optimize
        loss,  # the function we want to minimize
        loss_constants={"H_MPO": H_MPO},
        # tags=['U3'],              # only optimize U3 tensors
        autodiff_backend="jax",  # use 'autograd' for non-compiled optimization
        optimizer="L-BFGS-B",  # the optimization algorithm
    )


# %% [markdown]
# Let us optimize for a first bond dimension $\chi=4$. This is already extremely fast and accurate.

# %%
tn_opt = myoptimizer(4, H_MPO)
psi_opt = tn_opt.optimize(n=200)
print("Optimized energy ", tn_opt.loss)
print("Exact energy ", E_exact)
# tn_opt.plot(hlines={'analytic':E_exact})

# %% [markdown]
# Let us be more quantitative by relating the energy estimation to the number of variational parameters $d=2((n-2)\chi^2+2\chi)$ of an MPS, for various $\chi$

# %%
chi_a = [2, 4, 10, 20]
nbp = len(chi_a)
d_a = np.zeros(nbp)
E_a = np.zeros(nbp)
for i in range(nbp):
    tn_opt = myoptimizer(chi_a[i], H_MPO)
    psi_opt = tn_opt.optimize(n=200)
    d_a[i] = tn_opt.d
    E_a[i] = tn_opt.loss

# %%
plt.loglog(d_a, E_a - E_exact, "-o")
plt.xlabel("number of params")
plt.ylabel("energy error");

# %% [markdown]
# ### Finding a quantum circuit directly
#
# MPS are good candidates for the initial state of the QPE algorithm because they can be prepared efficiently in a quantum computer. One possibility would be to find variationally the circuit that best approximates an MPS (see e.g. [this recent proposal](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.132.040404)). Here, we take a shortcut, and directly find the circuit whose associated wavefunction minimizes the energy, in the spirit of R. Haghshenas et al.,  [*Variational Power of Quantum Circuit Tensor Networks*](https://link.aps.org/doi/10.1103/PhysRevX.12.011047). One practical advantage is that we may have less parameters to optimize using parametrized circuits compared to dense-tensors-based MPS.
#
# We consider finite-depth quantum circuits made of parametrized $U_3$ single qubit rotations and $R_{ZZ}$ two qubit gates.

# %% [markdown]
# Let us visualize the circuit wavefunction, and the tensor network contraction associated with the estimation of the energy.

# %%
depth = 4
circ = ansatz_circuit(n, depth)
psi = circ.psi
psi.draw(color=["U3", "RZZ"], show_inds=True)
psiH = psi.H
psi.align_(H_MPO, psiH)
E = psiH & H_MPO & psi
print(len(E.tensors))
E.draw(color=["U3", "RZZ"], show_inds=True)
Es = E.full_simplify()
print(len(Es.tensors))
Es.draw(color=["U3", "RZZ"], show_inds=True)


# %% [markdown]
# We can then define our optimizer on the angles that parametrize the gates.


# %%
def loss_circ(circ, H_MPO):
    psi = circ.psi
    psiH = psi.H
    c = psiH & psi
    psi.align_(H_MPO, psiH)
    E = psiH & H_MPO & psi  # .full_simplify()
    return autoray.do("real", E.contract(all, optimize=opt)) / autoray.do(
        "real", c.contract(all, optimize=opt)
    )


def my_circ_optimizer(circ):
    return qtn.TNOptimizer(
        circ,  # the tensor network we want to optimize
        loss_circ,  # the function we want to minimize
        loss_constants={
            "H_MPO": H_MPO
        },  # supply U to the loss function as a constant TN
        autodiff_backend="jax",  # use 'autograd' for non-compiled optimization
        optimizer="L-BFGS-B",
    )


# %% [markdown]
# Let us make a first test. Since the contraction is a bit more involved than for straightforward MPS optimization, the simulation becomes a bit more tedious.

# %%
depth = 4
circ = ansatz_circuit(n, depth)
# tn_opt_circ = mycircoptimizer(depth,H_MPO,circ)
my_circ_optimizer_ = my_circ_optimizer(circ)
circ_opt = my_circ_optimizer_.optimize(n=100)
print("Optimized energy ", my_circ_optimizer_.loss)
print("Exact energy ", E_exact)
my_circ_optimizer_.plot();

# %% [markdown]
# It can be also interesting to optimize a circuit of large depth using the optimized parameters from a circuit of smaller depth. However as shown below, we don't observe an improvement. So we will not use that in the following.

# %%
# optimize a shallow circuit
circ = ansatz_circuit(n, 2)
my_circ_optimizer_ = my_circ_optimizer(circ)
circ_opt = my_circ_optimizer_.optimize(n=100)
# initialize a deeper circuit with previously optimized parameters
circ = ansatz_circuit(n, 4, random_coeff=1e-4)
circ.set_params(circ_opt.get_params())
# optimize the deeper circuit
my_circ_optimizer_ = my_circ_optimizer(circ)
circ_opt = my_circ_optimizer_.optimize(n=100)
print("Optimized energy ", my_circ_optimizer_.loss)
print("Exact energy ", E_exact)

# %% [markdown]
# Let us analyze how things improve increasing the depth

# %%
depth_max = 4
depth_b = range(1, depth_max + 1)
nbp = depth_max
E_b = np.zeros(nbp)
d_b = np.zeros(nbp)
# E = np.zeros(depth_max)
for i in range(nbp):
    circ = ansatz_circuit(n, depth_b[i])
    my_circ_optimizer_ = my_circ_optimizer(circ)
    circ_opt = my_circ_optimizer_.optimize(n=100)
    d_b[i] = my_circ_optimizer_.d
    E_b[i] = my_circ_optimizer_.loss
    print("Depth ", depth_b[i])
    print("Current energy ", E_b[i])


# %%
plt.loglog(d_b, E_b - E_exact, "-s", color="tab:orange", label="circuit")
plt.xlabel("number of params")
plt.ylabel("energy error")
plt.legend();

# %%
psi_exact = dmrg.state
overlap = (psi_exact.H & circ_opt.psi).contract()
print("fidelity ", abs(overlap) ** 2)

# %% [markdown]
# ## Quantum Phase Estimation
#
# Let us know use a guess state the circuit optimization to initialize the QPE algorithm. For simplicity we will take a Hamiltonian defined on $n=4$ qubits.

# %%
n = 4
H = heisenberg_hamiltonian(n)
H_MPO = H.to_mpo()

dmrg = qtn.DMRG2(H_MPO, bond_dims=[10, 20, 40, 100], cutoffs=1e-13)
dmrg.solve(tol=1e-6, verbosity=1)
E_exact = np.real(dmrg.energy)
psi_exact = dmrg.state

print("\nStored reference energy ", E_exact)

# %% [markdown]
# We take the shallowest circuit

# %%
depth = 1
circ = ansatz_circuit(n, depth)
my_circ_optimizer_ = my_circ_optimizer(circ)
circ_opt = my_circ_optimizer_.optimize(n=100)

print("Optimized energy ", my_circ_optimizer_.loss)
print("Exact energy ", E_exact)

# %% [markdown]
# With this depth, we should be able to reach around 30% fidelity. If it isn't the case, try to rerun the optimization

# %%
overlap = (psi_exact.H & circ_opt.psi).contract()
fidelity = abs(overlap) ** 2
print("fidelity ", fidelity)

# %% [markdown]
# Running QPE on a state with 30% overlap should increase the fidelity.
# Let us initialize a circuit with a physical register and a phase register. We take $m=3$ phase qubits.

# %%
m = 3  # number of phase qubits

# circuit from optimized gates
circ_qpe = qtn.CircuitMPS(m + n)

for gate in circ_opt.gates:
    label, params, qubits = gate.label, gate.params, gate.qubits
    qubits = tuple(qb + m for qb in qubits)
    circ_qpe.apply_gate(label, *params, *qubits)

# %% [markdown]
# Set QPE parameters: first define the size of the search interval $\Delta$ which sets the evolution time $t$

# %%
E_target = my_circ_optimizer_.loss
size_interval = 3
E_const, Emax, evolution_time, global_phase = set_search_window(
    H, E_target, size_interval
)

# %% [markdown]
# and the Trotter parameters

# %%
trotter_order = 2
n_steps = 4
dt = evolution_time / n_steps

# %% [markdown]
# then run textbook QPE (this can take a minute or two). Since $n$ is small, for faster execution you can always set $dt=0$ to perform exact time evolution instead of a Trotterization.

# %%
# %%time
traces, res = qpe_sample(
    H,
    circ_qpe,
    evolution_time,
    dt,  # Trotter timestep. 0 for exact time evolution
    global_phase,
    trotter_order=trotter_order,
)

# %% [markdown]
# NB: the energy minimum can be $< E_{\rm exact}$
#  when $E_{\rm target} - \Delta/2 < E_{\rm exact}$. We only consider the bitstrings with probability $> 4/\pi^2 F$ where $F = |\langle{\psi}|\psi_{\rm exact}\rangle|^2$. The $4/\pi^2$ factor gives a lower bound on the QPE success probability depending on the initial overlap (see e.g. [Measurement section of QPE Wikipedia](https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm) for a derivation).
#
# NB: in practice one will not have access to the overlap. An approximation of the fidelity is sufficient. The following quantity can be used as a proxy, see [arxiv:2306.02620](https://arxiv.org/abs/2306.02620):
#
# $$ F \approx \exp\Big(- \frac{(E-E_0)^2}{2\sigma^2}\Big ),$$
# where $\sigma = \langle H^2 \rangle - E^2$ is the energy variance on guess state $\ket{\psi}$, and $E_0$ is an estimate of the exact ground state energy that doesn't need to be very accurate.

# %%
k_probs_list = sorted(enumerate(np.ravel(res)), key=lambda x: x[1], reverse=True)

kmin = 2 ** (m - 1)
prob_min = 0
emin = E_target
energies = []

for k, prob in k_probs_list:
    energy = E_target + size_interval * (1 / 2 - k / 2**m)
    energies.append(energy)
    if energy < emin and prob > fidelity * 4 / np.pi**2:
        emin = energy
        kmin = k
        prob_min = prob

print(f"Most probable energy = {energies[0]:.5f} w prob = {k_probs_list[0][1]:.5f}")
print(f"First 5 energies = {np.round(energies[:5], 5)}")
print(f'Minimal "significant" energy = {emin:.5f} with probability {prob_min:.5f}')
print(
    f"error = {abs(E_exact - emin):.5f} (theoretical error bound = {size_interval / 2**m:.5f})"
)

# %% [markdown]
# When the GS energy is measured, the physical register contains the GS wavefunction.
# We thus project the phase register of the final circuit wavefunction on the previously found bitstring that
# gives the minimal energy while satisfying the probability criterion

# %%
psi_final = traces["circuit"].psi


# We change psi_final.backend from 'jax' to
# 'numpy' to avoid a ValueError when calling
# psi_final. measure. See the following issue
# https://github.com/jcmgray/quimb/issues/340
# Problem should be fixed by quimb's commit fec27eb
def to_backend(x):
    return np.asarray(x)


psi_final.apply_to_arrays(to_backend)

bin_kmin = f"{kmin:b}".zfill(m)
for i in range(m):
    psi_final.measure_(0, outcome=int(bin_kmin[i]), remove=True)

# %% [markdown]
# The fidelity should improve

# %%
overlap_qpe = psi_final.overlap(psi_exact)
fidelity_qpe = abs(overlap_qpe) ** 2
print(f"Fidelity before QPE = {fidelity:.5f}")
print(f"Fidelity after QPE = {fidelity_qpe:.5f}")

# %% [markdown]
# We observe that even with a moderate overlap, Quantum Phase Estimation projects on the ground state. As an exercise, you can try to reproduce the following figure:
#
# <img src="./figures/fig_overlaps_circ_opt.svg" align="center">
#
# it shows the infidelity as a function of circuit depth (orange curve) or bond dimension (blue curve) for a guess state prepared with either circuit optimization or DMRG. The colored stars then show how the fidelity quickly improves (error drops to $\sim 10^{-5}$) when running QPE with even a few phase qubits upon the trial state from circuit optimization.

# %%
