#!/usr/bin/env python3

import json
import os
import tempfile

os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cotengra as ctg
import networkx as nx
import numpy as np
import quimb as qu

from qpe_toolbox.circuit.qaoa import (
    brute_force_MaxCut,
    find_W_and_C_QAOA,
    generate_community_graph,
    study_optimization_time_costs,
)


def test_qaoa():
    num_qubits = 6
    G = generate_community_graph(num_qubits, rng=np.random.default_rng(42))
    terms = dict.fromkeys(G.edges, 1)
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
    max_energy, _ = brute_force_MaxCut(nx.to_numpy_array(G), terms)

    assert np.abs(min(fevals) / max_energy) > 0.25


def test_W_and_C():
    with tempfile.TemporaryDirectory() as tmp_dir:
        orig = os.getcwd()
        os.chdir(tmp_dir)
        try:
            _run_W_and_C()
        finally:
            os.chdir(orig)


def _run_W_and_C():
    list_N = [3, 4]
    n_realizations = 2
    rng = np.random.default_rng(42)

    dict_Gs = {}
    indx = 0
    for i in range(len(list_N)):
        for _ in range(n_realizations):
            prob = rng.uniform(0.25, 0.75)
            G = nx.fast_gnp_random_graph(
                n=list_N[i], p=prob
            )  # sample an Erdos-Renyi graph
            if len(G.edges) > 0:
                terms = [[int(i), int(j)] for i, j in G.edges]
                dict_Gs[str(indx)] = {
                    "terms": terms,  # attach homogeneous couplings
                    "N": int(list_N[i]),
                    "E": len(terms),
                }
                dict_Gs[str(indx)]["type"] = "ER"
                dict_Gs[str(indx)]["param"] = float(prob)

                indx += 1

    filename1 = "my_new_graphs_QAOA"
    with open(filename1 + ".json", "w") as f:
        json.dump(dict_Gs, f, indent=4)

    hopt = ctg.ReusableHyperOptimizer(
        max_repeats=128,
        methods=["greedy"],
        optlib="random",
        minimize="write",
    )

    filename2 = "test_W_and_C"
    find_W_and_C_QAOA(
        graph_dict_name="my_new_graphs_QAOA",
        results_name=filename2,
        hyperopt=hopt,
        circuit_depths=(2, 3),
        verbosity=1,
        description="Economic hyperopt.",
    )

    assert os.path.exists(filename1 + ".json")
    assert os.path.exists(filename2 + ".json")


if __name__ == "__main__":
    test_qaoa()
    test_W_and_C()
