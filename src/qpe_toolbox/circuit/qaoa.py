# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

import copy
import itertools as it
import time
import warnings

import networkx as nx
import numpy as np
import optuna
import quimb as qu
import quimb.tensor as qtn

_ZZ = qu.pauli("Z") & qu.pauli("Z")  # edge operator (classical Ising energy)


def brute_force_maxcut(graph_matrix, terms):
    """Compute the Max-Cut value of a graph by brute-force enumeration.

    This function enumerates all possible non-trivial bipartitions of the
    vertex set and computes the corresponding cut value. It is intended
    for small graphs due to its exponential complexity.

    Parameters
    ----------
    graph_matrix : array_like, shape (N, N)
        Adjacency matrix of the graph. An edge between vertices ``i`` and
        ``j`` is assumed to exist if ``G[i, j] == 1`` or ``G[j, i] == 1``.

    terms : dict
        Dictionary mapping edges to their weights. Keys must be tuples
        ``(i, j)`` with ``i < j``, and values are the corresponding edge
        weights.

    Returns
    -------
    maxcut : float
        Maximum cut value over all bipartitions.

    winners : list of tuple
        List of bipartitions (each given as a tuple of vertex indices)
        achieving the maximum cut value. Each tuple represents one side
        of the bipartition.

    Notes
    -----
    - Bipartitions differing only by exchange of the two sets are treated
      as equivalent; only subsets of size ``1`` to ``N-1`` are enumerated.
    - The computational complexity scales as :math:`O(2^N)` and is only
      practical for small graphs.

    """
    n_qubits = np.shape(graph_matrix)[0]
    possible_bipartitions = []
    for r in range(1, n_qubits):
        possible_bipartitions += list(it.combinations(range(n_qubits), r))

    # Calculate for each bipartition the number of cuts
    cuts = np.zeros(len(possible_bipartitions))
    for s, x in enumerate(possible_bipartitions):
        for j in x:
            for k in range(n_qubits):
                if k not in x and (graph_matrix[j, k] == 1 or graph_matrix[k, j] == 1):
                    if j < k:
                        cuts[s] += terms[j, k]
                    else:
                        cuts[s] += terms[k, j]
    # Evalutate maxcut and search for the corresponding bipartitions
    maxcut = cuts.max()
    winners = [possible_bipartitions[s] for s in (cuts == maxcut).nonzero()[0]]
    return maxcut, winners


def generate_community_graph(N, *, N_comm=4, rng=None):
    """Generate a random graph with community (block) structure.

    The graph is generated using a stochastic block model, where vertices
    are divided into communities of random sizes and intra-community edges
    are more likely than inter-community edges.

    Parameters
    ----------
    N : int
        Total number of vertices in the graph.

    N_comm : int, optional
        Target number of communities. The actual number of non-trivial
        communities may be smaller if some generated sizes are <= 1.
        Default is 4.

    rng : :numpy-random:`numpy.random.Generator <generator>`, optional
        Random number generator used to sample community sizes.
        If ``None``, a default generator is created.

    Returns
    -------
    G : :class:`networkx.Graph`
        A graph generated according to the stochastic block model.

    Notes
    -----
    - Community sizes are randomly generated and add up to ``N``.
    - The probability matrix is chosen such that:
        - Intra-community edge probability is 0.5.
        - Inter-community edge probability is ``0.5 / (N_comm - 1)``.

    """
    if rng is None:
        rng = np.random.default_rng()
    size_comm = []
    N_av, n = N, 0
    for i in range(N_comm - 1):
        N_av = N_av - n
        n = int((1 / (N_comm - i)) * N_av * rng.random())
        size_comm.append(n)
    size_comm.append(N - sum(size_comm))
    size_comm = [int(i) for i in size_comm if i > 1]

    p = (0.5 / (N_comm - 1)) * np.ones((len(size_comm), len(size_comm)))
    for i in range(len(size_comm)):
        p[i, i] = 0.5

    G = nx.stochastic_block_model(size_comm, p, seed=rng)
    if G.number_of_edges() == 0:
        warnings.warn(
            "Generated community graph has no edges. "
            "Consider passing a different rng or increasing N.",
            UserWarning,
            stacklevel=2,
        )
    return G


def qaoa_energy(x, terms, opt):
    """Evaluate the QAOA energy for a given parameter vector.

    This function constructs a QAOA circuit for the Max-Cut Hamiltonian
    and computes the expectation value of the cost Hamiltonian using
    local expectation values.

    Parameters
    ----------
    x : array_like
        QAOA parameter vector of length ``2p``, where ``p`` is the number
        of QAOA layers. The first ``p`` entries correspond to the angles
        ``gamma``, and the last ``p`` entries correspond to the angles
        ``beta``.

    terms : dict
        Dictionary mapping edges ``(i, j)`` to their weights in the
        cost Hamiltonian.

    opt : :cotengra-api:`ReusableHyperOptimizer`
        Optimizer object for tensor network contraction ordering. Should be
        an instance of ``cotengra``'s ``ReusableHyperOptimizer`` (or compatible
        optimizer). Reusing this object across multiple calls improves
        performance by reusing previously computed contraction strategies.

    Returns
    -------
    energy : float
        Negative expectation value of the cost Hamiltonian evaluated
        for the given QAOA parameters.

    Notes
    -----
    - Expectation values are computed using
      ``circ.local_expectation`` with the JAX backend.
    - The returned value is real; any small imaginary part arising from
      numerical errors is discarded.

    """
    p = np.size(x) // 2
    gammas = x[:p]
    betas = x[p:]
    circ = qtn.circ_qaoa(terms, p, gammas, betas)

    ens = [
        circ.local_expectation(weight * _ZZ, edge, optimize=opt, backend="jax")
        for edge, weight in terms.items()
    ]

    return -sum(ens).real


def study_optimization_time_costs(
    hamilt_terms, hyperopt, bounds, *, batch_size=5, num_iter=20, verbosity=0, seed=None
):
    """Measure time costs for batched parameter suggestions and evaluations
    in an ``optuna`` optimization loop.

    This function runs a ``CMA-ES`` optimization study using ``optuna`` and records the
    time taken for three steps in each iteration:
    1. Asking the study for new parameter suggestions (``ask_time``)
    2. Evaluating the cost function (here, the QAOA energy) (``cost_time``)
    3. Updating the study with the results (``tell_time``)

    Parameters
    ----------
    hamilt_terms : dict
        Dictionary representing the Hamiltonian to optimize, typically mapping
        edges or terms to weights, compatible with the ``energy`` function.

    hyperopt : :cotengra-api:`ReusableHyperOptimizer`
        Optimizer object used in the ``energy`` function to control tensor network
        contraction ordering. Reusing this object across multiple calls improves
        performance.

    bounds : sequence of tuple of float
        Bounds for each parameter in the optimization. Each tuple should be
        ``(lower_bound, upper_bound)``.

    batch_size : int, optional
        Number of parameter sets evaluated per iteration. Default is 5.

    num_iter : int, optional
        Number of optimization iterations. Default is 20.

    verbosity : int, optional
        Controls printing of intermediate results. If ``>= 1``, prints the
        lowest energy in each batch. Default is 0.

    Returns
    -------
    ask_time : list of float
        List of times (in seconds) spent asking the study for new parameter suggestions
        in each iteration.

    tell_time : list of float
        List of times (in seconds) spent updating the study with evaluated energies
        in each iteration.

    cost_time : list of float
        List of times (in seconds per Hamiltonian term per parameter set) spent
        computing the energies in each iteration.

    study : :optuna-api:`study.Study`
        The ``optuna`` study object after all iterations. Can be used to query
        best parameters, best value, or continue optimization.

    Notes
    -----
    - ``cost_time`` is normalized by the number of Hamiltonian terms and batch size
      to give an average per term per parameter set.

    """
    ask_time, cost_time, tell_time = [], [], []
    study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(seed=seed))

    for _ in range(num_iter):
        t_0 = time.time()
        trial_numbers = []
        x_batch = []
        for _ in range(batch_size):
            trial = study.ask()
            trial_numbers.append(trial.number)
            list_suggested_params = [
                trial.suggest_float(str(i), b[0], b[1]) for i, b in enumerate(bounds)
            ]
            x_batch.append(list_suggested_params)
        t_e = time.time()
        ask_time.append(t_e - t_0)

        t_0 = time.time()
        energies = [
            float(qaoa_energy(x=x, terms=hamilt_terms, opt=hyperopt)) for x in x_batch
        ]
        if verbosity >= 1:
            print(f"Lowest energy in the batch: {min(energies):.2f}")
        t_e = time.time()
        cost_time.append((t_e - t_0) / (len(hamilt_terms) * batch_size))

        t_0 = time.time()
        for trial_number, energy in zip(trial_numbers, energies, strict=True):
            study.tell(trial_number, energy)
        t_e = time.time()
        tell_time.append(t_e - t_0)

    return ask_time, tell_time, cost_time, study


def compute_qaoa_contraction_costs(
    graph_dict,
    hyperopt,
    *,
    circuit_depths=(2, 3, 4),
    verbosity=0,
    description="None",
):
    """Compute contraction widths and costs for QAOA circuits on multiple graphs.

    Generates QAOA circuits of varying depth for each graph and estimates the
    tensor network contraction width (W) and log-scaled contraction cost (C)
    using rehearsal contractions.

    Parameters
    ----------
    graph_dict : dict
        Dictionary mapping graph identifiers to entries. Each entry must have at
        least a key ``"terms"`` listing the edges as pairs of vertices.

    hyperopt : :cotengra-api:`ReusableHyperOptimizer`
        Optimizer object used for tensor network contraction rehearsal. Reusing this
        object across multiple graphs improves performance.

    circuit_depths : sequence of int, optional
        List of QAOA circuit depths (number of layers) to analyze. Default is (2, 3, 4).

    verbosity : int, optional
        Level of output. If >= 1, prints progress for each graph and each depth.
        Default is 0.

    description : str, optional
        Text description of this hyperoptimizer run (e.g., "greedy", "random").
        Default is "None".

    Returns
    -------
    dict
        Updated copy of ``graph_dict`` with QAOA contraction metrics. For each
        graph, a new numbered entry is added under ``"hyperoptimizers"``,
        containing contraction width ``W``, cost ``C`` for each depth ``p``,
        and the given description. For example::

            "0": {
                "terms": [[0, 1], [0, 2]],
                "N": 3,
                "E": 2,
                "hyperoptimizers": {
                    "1": {
                        "p=2": {"W": 2.3509, "C": 2.5964},
                        "p=3": {"W": 2.3772, "C": 2.9702},
                        "description": "greedy"
                    }
                }
            }

    Notes
    -----
    - For each graph, a random QAOA parameter initialization is used (``gammas`` and ``betas``).
    - Contraction rehearsal is performed using :quimb-api:`Circuit.local_expectation_rehearse` for each
      term in the Hamiltonian.
    - The average width ``W`` is computed over all local contraction trees.
    - The total contraction cost ``C`` is computed using a numerically stable log-sum-exp
      over all local contraction costs.

    """
    result = copy.deepcopy(graph_dict)

    for key_entry, entry in graph_dict.items():
        if verbosity >= 1:
            print(f"Graph {int(key_entry) + 1}")

        if "hyperoptimizers" not in result[key_entry]:
            result[key_entry]["hyperoptimizers"] = {}

        existing_keys = [int(k) for k in result[key_entry]["hyperoptimizers"]]
        next_key = str(max(existing_keys) + 1 if existing_keys else 1)

        result[key_entry]["hyperoptimizers"][next_key] = {"description": description}

        for depth in circuit_depths:
            terms = {(edge[0], edge[1]): 1.0 for edge in entry["terms"]}

            gammas = qu.randn(depth)
            betas = qu.randn(depth)
            circ = qtn.circ_qaoa(terms, depth, gammas, betas)

            local_exp_rehs = [
                circ.local_expectation_rehearse(weight * _ZZ, edge, optimize=hyperopt)
                for edge, weight in terms.items()
            ]

            average_weights = np.mean([rehs["W"] for rehs in local_exp_rehs])
            all_costs = np.array([rehs["C"] for rehs in local_exp_rehs])
            max_cost = np.max(all_costs)
            total_cost = max_cost + np.log10(np.sum(10 ** (all_costs - max_cost)))

            result[key_entry]["hyperoptimizers"][next_key][f"p={depth}"] = {
                "W": average_weights,
                "C": total_cost,
            }

            if verbosity >= 1:
                print(
                    f"Depth {depth}: W_avg={np.round(average_weights, 4)}, C_tot={np.round(total_cost, 4)}"
                )
        if verbosity >= 1:
            print("\n")

    return result
