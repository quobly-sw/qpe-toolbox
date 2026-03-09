#!/usr/bin/env python3

import os

os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cotengra as ctg
import networkx as nx
import numpy as np
import quimb as qu

from qpe_toolbox.circuit.qaoa import (
    brute_force_maxcut,
    compute_qaoa_contraction_costs,
    generate_community_graph,
    study_optimization_time_costs,
)


def test_study_optimisation():
    n_qubits = 6
    graph = generate_community_graph(n_qubits, rng=np.random.default_rng(42))
    terms = dict.fromkeys(graph.edges, 1)
    p = 3

    hopt = ctg.ReusableHyperOptimizer(
        max_repeats=128,
        methods=["greedy"],
        optlib="random",
        minimize="write",
    )

    eps = 1e-6
    bounds = [(0.0 + eps, qu.pi / 2 - eps)] * p + [
        (-qu.pi / 4 + eps, qu.pi / 4 - eps)
    ] * p
    _, _, _, study = study_optimization_time_costs(
        hamilt_terms=terms,
        hyperopt=hopt,
        bounds=bounds,
        batch_size=2,
        num_iter=20,
        verbosity=0,
        seed=42,
    )

    fevals = [float(trial.value) for trial in study.trials]
    max_energy, _ = brute_force_maxcut(nx.to_numpy_array(graph), terms)

    assert np.abs(min(fevals) / max_energy) > 0.25


def test_contraction_costs():
    node_numbers = [3, 4]
    rng = np.random.default_rng(42)

    graph_dicts = {}
    indx = 0
    seeds = [42, 43]
    for n in node_numbers:
        for seed in seeds:
            edge_creation_prob = rng.uniform(0.25, 0.75)
            # sample an Erdos-Renyi graph
            graph = nx.fast_gnp_random_graph(n, edge_creation_prob, seed=seed)
            assert len(graph.edges) > 0

            # attach homogeneous couplings
            terms = [[int(i), int(j)] for i, j in graph.edges]
            graph_dicts[str(indx)] = {"terms": terms, "N": n, "E": len(terms)}
            graph_dicts[str(indx)]["type"] = "ER"
            graph_dicts[str(indx)]["param"] = float(edge_creation_prob)

            indx += 1

    hopt = ctg.ReusableHyperOptimizer(
        max_repeats=128, methods=["greedy"], optlib="random", minimize="write"
    )

    res = compute_qaoa_contraction_costs(
        graph_dicts, hopt, circuit_depths=(2, 3), description="Economic hyperopt."
    )
    assert isinstance(res, dict)


if __name__ == "__main__":
    test_study_optimisation()
    test_contraction_costs()
