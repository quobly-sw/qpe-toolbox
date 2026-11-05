import numpy as np
import scipy as sp
from circuits import (
    build_lcu_hadamard_test_circuit,
    build_lcu_qpe_circuit,
)
from config import EstimateResult, QPEConfig, RPEConfig
from lcu import LCU

from qpe_toolbox.estimation.qft import iqft_swapped
from qpe_toolbox.tensor import kron_mps

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - scipy is a hard dependency
    linear_sum_assignment = None


def _fold_state(state, nbits):
    """Fold a bitstring integer into the range [0, 2^{nbits-1}]."""
    return min(state, (1 << nbits) - state)


def _estimate_from_distribution(counts_or_probs, nbit_qpe, n_shots=None, verbosity=0):
    total = n_shots if n_shots and n_shots > 0 else sum(counts_or_probs.values())
    folded_counts = {}
    for state_int, count in counts_or_probs.items():
        folded = _fold_state(state_int, nbit_qpe)
        folded_counts[folded] = folded_counts.get(folded, 0.0) + count
    folded_probs = {k: v / total for k, v in folded_counts.items()}
    max_folded_state = max(folded_probs, key=folded_probs.get)
    theta = max_folded_state / 2**nbit_qpe
    return EstimateResult(theta, folded_probs, max_folded_state)


def run_qpe_lcu(lcu: LCU, config: QPEConfig, mps_init, *, rng=None):
    """
    Run QPE using the LCU walk operator. Returns estimated phase theta.
    """
    circ, regs = build_lcu_qpe_circuit(
        lcu,
        mps_init,
        config.nbit_qpe,
        max_bond=config.max_bond,
    )

    circ.apply_gates(iqft_swapped(regs.qpe))

    if config.n_shots == 0:
        probs = circ.compute_marginal(where=regs.qpe)
        probs = np.ravel(probs)
        prob_dict = dict(enumerate(probs))
        result = _estimate_from_distribution(
            prob_dict, config.nbit_qpe, n_shots=0, verbosity=config.verbosity
        )
    else:
        samples = circ.sample(config.n_shots, seed=rng)
        counts = {}
        for bitstring in samples:
            qpe_bits = bitstring[: config.nbit_qpe]
            state_int = int(qpe_bits, 2)
            counts[state_int] = counts.get(state_int, 0) + 1
        result = _estimate_from_distribution(
            counts, config.nbit_qpe, n_shots=config.n_shots, verbosity=config.verbosity
        )

    if config.verbosity >= 1:
        print("binary" + " " * 6 + "\t ket" + " " * 4 + "\t phase  \t prob")
        sorted_folded = sorted(
            result.folded_probabilities.items(), key=lambda x: x[1], reverse=True
        )[:5]
        for folded_state, prob in sorted_folded:
            print(
                f"{format(folded_state, f'0{config.nbit_qpe}b'):<12}",
                f"{'|' + str(folded_state) + '>':<8}",
                f"{folded_state / 2**config.nbit_qpe:<6.8f}",
                f"{prob:<6.8f}",
                sep=" \t ",
            )

    return result.theta


def _rpe_update_theta(phi_m, theta_ref, m):
    """Original single-phase RPE update."""
    n_plus = round((2**m * theta_ref - phi_m) / (2 * np.pi))
    theta_plus = (phi_m + 2 * np.pi * n_plus) / 2**m

    n_minus = round((2**m * theta_ref - (2 * np.pi - phi_m)) / (2 * np.pi))
    theta_minus = (2 * np.pi - phi_m + 2 * np.pi * n_minus) / 2**m

    candidates = []
    for cand in (theta_plus, theta_minus):
        if 0 <= cand <= np.pi:
            candidates.append(cand)
        elif cand < 0 and cand > -1e-12:
            candidates.append(0.0)
        elif cand > np.pi and cand < np.pi + 1e-12:
            candidates.append(np.pi)

    return min(candidates, key=lambda c: abs(c - theta_ref))


def _rpe_update_theta_M(phi_m, theta_ref, M):
    """
    Generalized single-phase update with arbitrary amplification factor M.
    This is useful for the RMPE setting where M_l is a product of integers
    or a real runtime factor, rather than necessarily 2**m.
    """
    n_plus = round((M * theta_ref - phi_m) / (2 * np.pi))
    theta_plus = (phi_m + 2 * np.pi * n_plus) / M

    n_minus = round((M * theta_ref - (2 * np.pi - phi_m)) / (2 * np.pi))
    theta_minus = (2 * np.pi - phi_m + 2 * np.pi * n_minus) / M

    candidates = []
    for cand in (theta_plus, theta_minus):
        if 0 <= cand <= np.pi:
            candidates.append(cand)
        elif cand < 0 and cand > -1e-12:
            candidates.append(0.0)
        elif cand > np.pi and cand < np.pi + 1e-12:
            candidates.append(np.pi)

    if not candidates:
        candidates = [theta_plus, theta_minus]

    return min(candidates, key=lambda c: abs(c - theta_ref))


def _rpe_update_thetas(phi_ms, theta_refs, m=None, M=None):
    """
    Multiple-eigenvalue RPE update.

    Parameters
    ----------
    phi_ms : sequence of float
        Measured phases in [0, 2*pi), one per detected eigenvalue/interval.
    theta_refs : sequence of float
        Previous phase estimates in [0, pi].
    m : int, optional
        Stage index. If M is None, then M = 2**m.
    M : float, optional
        Amplification factor M_l. Overrides m.
    one_to_one : bool, default=False
        If True and len(phi_ms) == len(theta_refs), solve a linear
        assignment so that each measured phase is matched to a distinct
        previous estimate. Otherwise, each phi is matched independently
        to its closest previous estimate.

    Returns
    -------
    theta_new : list of float
        Updated estimates, same length as phi_ms.
    """
    if M is None:
        if m is None:
            raise ValueError("Either m or M must be provided.")
        M = 2**m

    phi_ms = np.atleast_1d(phi_ms).astype(float)
    theta_refs = np.atleast_1d(theta_refs).astype(float)

    # One-to-one matching, useful after the S eigenvalues are resolved.
    cost = np.empty((len(phi_ms), len(theta_refs)))
    for i, phi in enumerate(phi_ms):
        for j, ref in enumerate(theta_refs):
            cand = _rpe_update_theta_M(phi, ref, M)
            cost[i, j] = abs(cand - ref)

    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost)
    else:
        # Simple greedy fallback for small S.
        pairs = [
            (cost[i, j], i, j)
            for i in range(cost.shape[0])
            for j in range(cost.shape[1])
        ]
        pairs.sort()
        used_i, used_j = set(), set()
        row_ind, col_ind = [], []
        for _, i, j in pairs:
            if i not in used_i and j not in used_j:
                used_i.add(i)
                used_j.add(j)
                row_ind.append(i)
                col_ind.append(j)
        row_ind = np.array(row_ind)
        col_ind = np.array(col_ind)

    out = [None] * len(phi_ms)
    for i, j in zip(row_ind, col_ind, strict=True):
        out[i] = _rpe_update_theta_M(phi_ms[i], theta_refs[j], M)
    return out


def _get_hadamard_output(
    lcu, psi0, order, n_shots, *, max_bond=0, cutoff=1e-12, rng=None
):
    """
    Compute the arccos of the Hadamard test expectation value for theta = 0.
    """
    theta = 0.0
    # Build the Hadamard test circuit with theta = 0 (as in the original wrapper)
    circ = build_lcu_hadamard_test_circuit(
        lcu, psi0, order, theta, max_bond=max_bond, cutoff=cutoff
    )
    aux_ind = 0

    # Evaluate the expectation value <Z> on the auxiliary qubit
    if n_shots == 0:
        # Exact mode: compute marginal probability and take difference
        probs = circ.compute_marginal(where=[aux_ind])
        X_m = probs[0] - probs[1]
    else:
        # Sampling mode: estimate <Z> from measurement outcomes
        popcount = 0
        for bitstring in circ.sample(n_shots, seed=rng):
            popcount += int(bitstring[aux_ind])
        X_m = (n_shots - 2 * popcount) / n_shots

    # Clip to avoid numerical issues outside [-1, 1]
    return np.clip(X_m, -1.0, 1.0)


def run_rpe_lcu(lcu: LCU, config: RPEConfig, mps_init, *, rng=None):
    """
    Run robust phase estimation using the LCU walk operator.
    Returns array of phase estimates after each iteration.
    """
    mps_prep = lcu.prepare_mps
    phi0 = kron_mps(mps_prep, mps_init)

    if rng is None:
        rng = np.random.default_rng()

    theta_values = np.zeros(config.n_repetitions)
    for m in range(config.n_repetitions):
        phi_m = _get_hadamard_output(
            lcu,
            phi0,
            2**m,
            config.n_shots,
            max_bond=config.max_bond,
            cutoff=config.cutoff,
            rng=rng,
        )
        phi_m = np.arccos(phi_m)
        if m == 0:
            theta_values[m] = phi_m
        else:
            theta_values[m] = _rpe_update_theta(phi_m, theta_values[m - 1], m)
    return theta_values


def run_rmpe_lcu(lcu: LCU, config: RPEConfig, mps_init, *, k_state=2, rng=None):
    """
    Run robust phase estimation using the LCU walk operator.
    Returns array of phase estimates after each iteration.
    """
    print(k_state)
    k_round = 8 * k_state

    mps_prep = lcu.prepare_mps
    phi0 = kron_mps(mps_prep, mps_init)

    if rng is None:
        rng = np.random.default_rng()

    theta = np.zeros((config.n_repetitions, k_state))
    for m in range(config.n_repetitions):
        phi = np.zeros(k_round + 1)
        for k in range(k_round + 1):
            phi[k] = _get_hadamard_output(
                lcu,
                phi0,
                k * 2**m,
                config.n_shots,
                max_bond=config.max_bond,
                cutoff=config.cutoff,
                rng=rng,
            )

        ind = sp.linalg.hankel(np.arange(k_round + 2))[
            : k_round // 2 + 1, : k_round // 2 + 1
        ]
        u, *_ = np.linalg.svd(phi[ind], full_matrices=True)
        u0 = u[0 : k_round // 2, 0 : 2 * k_state]
        u0 = np.linalg.pinv(u0)
        u1 = u[-k_round // 2 :, 0 : 2 * k_state]
        phi, *_ = np.linalg.eig(u0 @ u1)
        phi = np.angle(phi)
        phi = phi[0::2]  # [x for x in phi if x > 0.0]

        if m == 0:
            theta[0, :] = phi
        else:
            theta[m] = _rpe_update_thetas(phi_ms=phi, theta_refs=theta[m - 1], m=m)
    return theta
