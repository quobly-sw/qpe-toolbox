# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------
import json
import time
import warnings

import numpy as np
from quimb.tensor.circuit import parse_to_gate

from qpe_toolbox.circuit import count_gates
from qpe_toolbox.circuit.serialize_circuits import (
    serialize_from_quimb_Circuit,
    serialize_from_quimb_gates,
)

from .qft import iqft_swapped


def qpe_energy(
    hamiltonian,
    initial_circ,
    n_steps,
    E_target,
    size_interval,
    *,
    trotter_order=1,
    write_gates=False,
    optimize="auto-hq",
    verbosity=0,
):
    """
    Perform quantum phase estimation (QPE) to estimate the energy of a Hamiltonian.

    The algorithm encodes the phase corresponding to the Hamiltonian evolution
    into a phase register and samples it to extract the energy eigenvalue.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object from the QPE-Toolbox ``Hamiltonian`` class.
    initial_circ : :quimb-api:`Circuit` or :quimb-api:`CircuitMPS`
        Initial circuit preparing the trial state in the data register.
    n_steps : int or str
        Number of time steps for Trotterized evolution, or "exact" for exact evolution.
    E_target : float
        Central target energy for the search window.
    size_interval : float
        Width of the energy search interval.
    trotter_order : int, default ``1``
        Order of the Trotter decomposition.
    write_gates : bool, default ``False``
        If True, writes the gate sequence to a text file.
    optimize : str, default ``"auto-hq"``
        Optimization strategy when computing marginals with tensor networks.
    verbosity : int, default ``0``
        Verbosity level. If >= 1, print result summary. If >= 2, print
        additional debug information.

    Returns
    -------
    traces : dict
        Dictionary with computation information, including timing, bond dimensions,
        gate counts, and highest probability phase values.
    energy : float
        Estimated energy eigenvalue from the phase estimation.

    Notes
    -----
    - The phase register is automatically determined as ``initial_circ.N - hamiltonian.n_qubits``.
    - The estimated energy is computed as

      .. math::

         E = E_\\mathrm{max} - \\frac{2 \\pi \\theta}{t_\\mathrm{evol}} - E_\\mathrm{const}

      where :math:`\\theta` corresponds to the phase of the most probable state.
    - Supports both Trotterized and exact evolution.

    """
    E_const, Emax, evolution_time, global_phase = set_search_window(
        hamiltonian, E_target, size_interval
    )

    # First stage: phase encoding
    n_phase_bits = initial_circ.N - hamiltonian.n_qubits

    dt = "exact" if n_steps == "exact" else evolution_time / n_steps
    traces, probs = qpe_sample(
        hamiltonian,
        initial_circ,
        evolution_time,
        dt,
        global_phase,
        trotter_order=trotter_order,
        write_gates=write_gates,
        optimize=optimize,
        verbosity=verbosity - 1,
    )

    traces["prob"] = float(np.max(probs))  # float here is for JSON
    thetas_probs_list = np.ravel(probs).astype(float)
    thetas_probs_list = sorted(
        enumerate(thetas_probs_list), key=lambda x: x[1], reverse=True
    )
    traces["first_thetas"] = thetas_probs_list[:5]

    if verbosity >= 1:
        for x in thetas_probs_list[:5]:
            print(
                f"{x[0]:b}".zfill(n_phase_bits),
                f"|{x[0]}>",
                f"{x[0] / 2**n_phase_bits:<{n_phase_bits + 2}}",
                f"{x[1]:<6.4f}",
                flush=True,
            )

    max_prob_state_int = np.argmax(probs)
    theta = max_prob_state_int / 2**n_phase_bits

    energy = Emax - 2 * np.pi * theta / evolution_time
    energy -= E_const

    return traces, energy


def qpe_sample(
    hamiltonian,
    initial_circ,
    evolution_time,
    dt,
    global_phase,
    *,
    trotter_order=1,
    write_gates=False,
    rehearse=False,
    run_simulation=True,
    optimize="auto-hq",
    verbosity=0,
):
    """
    Apply quantum phase estimation to a given initial circuit and sample the output.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object from the QPE-Toolbox ``Hamiltonian`` class.
    initial_circ : :quimb-api:`Circuit` or :quimb-api:`CircuitMPS`
        Circuit preparing the trial state.
    evolution_time : float
        Total evolution time for the controlled-U operations.
    dt : float or str
        Trotter step size; if "exact", evolution is exact.
    global_phase : float
        Global phase added to the controlled-U operations.
    trotter_order : int, default ``1``
        Order of Trotter decomposition for time evolution.
    write_gates : bool, default ``False``
        If True, saves the gates to a text file.
    rehearse : bool, default ``False``
        If True, precomputes marginals without measurement.
    run_simulation : bool, default ``True``
        Whether to perform full tensor network simulation or just track gates.
    optimize : str, default ``"auto-hq"``
        Optimization strategy for tensor network marginal computation.
    verbosity : int, default ``0``
        If ``> 0``, prints timing and progress information.

    Returns
    -------
    traces : dict
        Dictionary containing bond dimensions, timing, and gate counts.
    result : array or list
        Either the probability tensor of phase qubits (if ``run_simulation`` is True) or a list of gate instructions.

    Notes
    -----
    - Phase estimation is performed using a Hadamard wall followed by controlled-U operations.
    - IQFT is applied on the phase register to extract probabilities.
    - When ``run_simulation=False``, the function produces a gate list instead of simulating the circuit.

    """
    n_phase_bits = initial_circ.N - hamiltonian.n_qubits
    st = time.time()

    phase_reg = list(range(n_phase_bits))

    traces, circ = qpe_first_stage(
        hamiltonian,
        initial_circ,
        evolution_time,
        dt,
        global_phase,
        trotter_order=trotter_order,
        run_simulation=run_simulation,
        verbosity=verbosity,
    )

    for gate_id in iqft_swapped(phase_reg):
        if run_simulation:
            circ.apply_gate(*gate_id, gate_round=traces["gate_round"])
        else:
            circ.append(parse_to_gate(*gate_id, gate_round=traces["gate_round"]))
        traces["gate_round"] += 1
    traces["ctimes"].append(time.time() - st)
    traces["gates_count"] = count_gates(circ)

    if write_gates:
        if dt == "exact":
            raise ValueError("Cannot write gates for exact time evolution")
        n_steps = int(evolution_time / dt)
        filename = f"QPE_ttr{trotter_order}{n_steps}steps_{hamiltonian.n_qubits}qubits_{n_phase_bits}phbits"
        if run_simulation:
            gate_dict = serialize_from_quimb_Circuit(circ)
        else:
            gate_dict = serialize_from_quimb_gates(initial_circ.N, circ)
        with open(filename + ".json", "w") as outfile:
            json.dump(gate_dict, outfile)

    if run_simulation:
        traces["circuit"] = circ.copy()
        if verbosity > 0:
            print("Start computing marginal on the phase register...")
            print(
                f"Elapsed {traces['ctimes'][-1]:.2f}s, bond dim {traces['bond_dims'][-1]}"
            )
        res = circ.compute_marginal(
            where=phase_reg, rehearse=rehearse, optimize=optimize
        )
        traces["ctimes"].append(time.time() - st)
        if verbosity >= 1:
            print(f"Done. Total time {traces['ctimes'][-1]:.2f}s")
        return traces, res
    return traces, circ


def qpe_first_stage(
    hamiltonian,
    initial_circ,
    evolution_time,
    dt,
    global_phase,
    *,
    trotter_order=1,
    run_simulation=True,
    verbosity=0,
):
    """
    Perform the first stage of the quantum phase estimation algorithm.

    This includes:

    * Applying a Hadamard wall on the phase register.
    * Controlled-U operations with the Hamiltonian evolution.
    * Optional Trotterization for approximate time evolution.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object from the QPE-Toolbox ``Hamiltonian`` class.
    initial_circ : :quimb-api:`Circuit` or :quimb-api:`CircuitMPS`
        Initial state of the system.
    evolution_time : float
        Total evolution time.
    dt : float or str
        Time step for Trotter decomposition; "exact" for exact evolution.
    global_phase : float
        Global phase applied to controlled-U operations.
    trotter_order : int, default ``1``
        Trotter order for time evolution.
    run_simulation : bool, default ``True``
        Whether to perform full tensor network simulation or just track gates.
    verbosity : int, default ``0``
        Verbosity level. If >= 1, print progress and bond dimension information.

    Returns
    -------
    traces : dict
        Contains bond dimensions, computation times, and optionally other metadata.
    circ_or_gates : :quimb-api:`Circuit` or list
        Updated circuit if ``run_simulation`` is True; otherwise, list of gate instructions.

    Notes
    -----
    - The phase register size is inferred from ``initial_circ.N - hamiltonian.n_qubits``.
    - Warnings are raised if the Trotter step size exceeds the required evolution time.

    """
    # input validation
    if dt == 0:
        dt = "exact"
    if not ((dt == "exact") or (np.isscalar(dt) and np.isreal(dt) and dt > 0)):
        raise ValueError("Can only evolve for positive dt")

    n_phase_bits = initial_circ.N - hamiltonian.n_qubits
    st = time.time()
    ctimes = []
    circ = initial_circ.copy()
    bd_list = [circ.psi.max_bond()]
    gates_list = []

    data_reg = [n_phase_bits + i for i in range(hamiltonian.n_qubits)]
    phase_reg = list(range(n_phase_bits))

    c_round = 0
    # Hadamard wall
    for k in range(n_phase_bits):
        if run_simulation:
            circ.apply_gate("H", phase_reg[k], gate_round=c_round)
        else:
            gates_list.append(parse_to_gate("H", phase_reg[k], gate_round=c_round))
    c_round += 1
    bd_list.append(circ.psi.max_bond())
    ctimes.append(time.time() - st)

    if verbosity >= 1:
        print(f"Start C-Us, elapsed {ctimes[-1]:.2f} s, bond dim {bd_list[-1]}")

    # Controlled-U
    for k in range(n_phase_bits):
        # |q> = q_0 * 2**(m-1) + ... + q_{m-1-k} * 2**k + ... + q_{m-1}

        if run_simulation:
            circ.apply_gate(
                "PHASE", global_phase * 2**k, phase_reg[k], gate_round=c_round
            )
        else:
            gates_list.append(
                parse_to_gate(
                    "PHASE", global_phase * 2**k, phase_reg[k], gate_round=c_round
                )
            )
        c_round += 1

        if dt == "exact":
            U_gate = hamiltonian.get_U_exact(
                evolution_time * 2**k, data_reg, controls=(phase_reg[k],)
            )
            circ.apply_gate(U_gate)
        else:
            if dt > evolution_time * 2**k:
                warnings.warn(
                    f"k={k}, dt={dt:.3f} > t*2**k={evolution_time * 2**k:.3f} -> dt set to t*2**k",
                    stacklevel=2,
                )
            n_steps = int(evolution_time * 2**k / dt + 1 / 2)
            trotter_slice = hamiltonian.get_trotter_step(dt, data_reg, trotter_order)
            for _ in range(n_steps):
                for gate_id in trotter_slice:
                    if run_simulation:
                        circ.apply_gate(
                            *gate_id, controls=(phase_reg[k],), gate_round=c_round
                        )
                    else:
                        gates_list.append(
                            parse_to_gate(
                                *gate_id, controls=(phase_reg[k],), gate_round=c_round
                            )
                        )
                    c_round += 1

        bd_list.append(circ.psi.max_bond())
        ctimes.append(time.time() - st)
        if verbosity >= 1:
            print(
                f"Done w/ {k}-th C-U, elapsed {ctimes[-1]:.2f} s, bond dim {bd_list[-1]}"
            )

    traces = {"ctimes": ctimes, "bond_dims": bd_list, "gate_round": c_round}
    if run_simulation:
        return traces, circ
    return traces, gates_list


def set_search_window(hamiltonian, E_target, size_interval):
    """
    Set up the energy search window for phase estimation.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object from the QPE-Toolbox ``Hamiltonian`` class.
    E_target : float
        Central target energy around which to search.
    size_interval : float
        Width of the energy interval (must be > 0).

    Returns
    -------
    E_const : float
        Constant energy offset of the Hamiltonian (``hamiltonian.e_const`` or 0.0).
    Emax : float
        Upper edge of the energy interval for phase encoding.
    evolution_time : float
        Total evolution time corresponding to the search interval.
    global_phase : float
        Phase corresponding to ``Emax * evolution_time``.

    Notes
    -----
    - Evolution time is chosen as ``2 * pi / size_interval`` to map the interval to [0, 2π].
    - ``global_phase`` is added to ensure the phase encoding is centered around the target energy.

    """
    if not (size_interval > 0):
        raise ValueError("Invalid size_interval")
    E_const = getattr(hamiltonian, "e_const", 0.0)
    Emax = E_target - E_const + size_interval / 2
    evolution_time = 2 * np.pi / size_interval
    global_phase = Emax * evolution_time

    return E_const, Emax, evolution_time, global_phase
