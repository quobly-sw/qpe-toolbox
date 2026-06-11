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

from qpe_toolbox import EXACT
from qpe_toolbox.circuit import count_gates
from qpe_toolbox.circuit.serialize_circuits import (
    serialize_from_quimb_Circuit,
    serialize_from_quimb_gates,
)

from .qpe_circuit import qpe_circuit, qpe_first_stage_circuit, qpe_gates


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
    n_steps : int or qpe_toolbox.EXACT
        Number of time steps for Trotterized evolution, or ``EXACT`` for exact
        evolution.
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

    dt = EXACT if n_steps is EXACT else evolution_time / n_steps
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
    dt : float or qpe_toolbox.EXACT
        Trotter step size; if ``EXACT``, evolution is exact.
    global_phase : float
        Global phase added to the controlled-U operations.
    trotter_order : int, default ``1``
        Order of Trotter decomposition for time evolution.
    write_gates : bool, default ``False``
        If True, saves the gates to a text file.
    rehearse : bool, default ``False``
        If True, precomputes marginals without measurement.
    optimize : str, default ``"auto-hq"``
        Optimization strategy for tensor network marginal computation.
    verbosity : int, default ``0``
        If ``> 0``, prints timing and progress information.

    Returns
    -------
    traces : dict
        Dictionary containing bond dimensions, timing, and gate counts.
    probs : array
        Probability tensor of the phase qubits.

    Notes
    -----
    - Phase estimation is performed using a Hadamard wall followed by controlled-U operations.
    - IQFT is applied on the phase register to extract probabilities.
    - To obtain the gate list without simulating the circuit (resource
      analysis), use ``qpe_gate_list``.
    """
    n_phase_bits = initial_circ.N - hamiltonian.n_qubits
    phase_reg = list(range(n_phase_bits))
    st = time.time()

    unitaries = _evolution_powers(
        hamiltonian, evolution_time, dt, n_phase_bits, trotter_order
    )
    global_phases = [global_phase * 2**k for k in range(n_phase_bits)]
    traces, circ = qpe_circuit(
        initial_circ, unitaries, global_phases=global_phases, verbosity=verbosity
    )
    traces["gates_count"] = count_gates(circ)

    if write_gates:
        filename = _qpe_gates_filename(
            hamiltonian, evolution_time, dt, trotter_order, n_phase_bits
        )
        gate_dict = serialize_from_quimb_Circuit(circ)
        with open(filename + ".json", "w") as outfile:
            json.dump(gate_dict, outfile)

    traces["circuit"] = circ.copy()
    if verbosity > 0:
        print("Start computing marginal on the phase register...")
        print(
            f"Elapsed {traces['ctimes'][-1]:.2f}s, bond dim {traces['bond_dims'][-1]}"
        )
    res = circ.compute_marginal(where=phase_reg, rehearse=rehearse, optimize=optimize)
    traces["ctimes"].append(time.time() - st)
    if verbosity >= 1:
        print(f"Done. Total time {traces['ctimes'][-1]:.2f}s")
    return traces, res


def qpe_gate_list(
    hamiltonian,
    n_phase_bits,
    evolution_time,
    dt,
    global_phase,
    *,
    trotter_order=1,
    write_gates=False,
):
    """
    Build the QPE gate list for a Hamiltonian evolution without simulating it.

    This produces the same gate sequence as ``qpe_sample`` but skips the tensor
    network simulation; use it for resource analysis and circuit serialization.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object from the QPE-Toolbox ``Hamiltonian`` class.
    n_phase_bits : int
        Number of phase estimation qubits.
    evolution_time : float
        Total evolution time for the controlled-U operations.
    dt : float or qpe_toolbox.EXACT
        Trotter step size; if ``EXACT``, evolution is exact.
    global_phase : float
        Global phase added to the controlled-U operations.
    trotter_order : int, default ``1``
        Order of Trotter decomposition for time evolution.
    write_gates : bool, default ``False``
        If True, saves the gates to a JSON file.

    Returns
    -------
    traces : dict
        Dictionary containing the gate counts.
    gates_list : list of :quimb-api:`Gate`
        Gate sequence of the full QPE circuit.
    """
    unitaries = _evolution_powers(
        hamiltonian, evolution_time, dt, n_phase_bits, trotter_order
    )
    global_phases = [global_phase * 2**k for k in range(n_phase_bits)]
    gates_list = list(qpe_gates(unitaries, global_phases=global_phases))
    traces = {"gates_count": count_gates(gates_list)}

    if write_gates:
        filename = _qpe_gates_filename(
            hamiltonian, evolution_time, dt, trotter_order, n_phase_bits
        )
        gate_dict = serialize_from_quimb_gates(
            n_phase_bits + hamiltonian.n_qubits, gates_list
        )
        with open(filename + ".json", "w") as outfile:
            json.dump(gate_dict, outfile)

    return traces, gates_list


def qpe_first_stage(
    hamiltonian,
    initial_circ,
    evolution_time,
    dt,
    global_phase,
    *,
    trotter_order=1,
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
    dt : float or qpe_toolbox.EXACT
        Time step for Trotter decomposition; ``EXACT`` for exact evolution.
    global_phase : float
        Global phase applied to controlled-U operations.
    trotter_order : int, default ``1``
        Trotter order for time evolution.
    verbosity : int, default ``0``
        Verbosity level. If >= 1, print progress and bond dimension information.

    Returns
    -------
    traces : dict
        Contains bond dimensions, computation times, and other metadata.
    circ : :quimb-api:`Circuit` or :quimb-api:`CircuitMPS`
        Updated circuit with the first stage applied.

    Notes
    -----
    - The phase register size is inferred from ``initial_circ.N - hamiltonian.n_qubits``.
    - Warnings are raised if the Trotter step size exceeds the required evolution time.
    """
    n_phase_bits = initial_circ.N - hamiltonian.n_qubits
    unitaries = _evolution_powers(
        hamiltonian, evolution_time, dt, n_phase_bits, trotter_order
    )
    global_phases = [global_phase * 2**k for k in range(n_phase_bits)]
    return qpe_first_stage_circuit(
        initial_circ, unitaries, global_phases=global_phases, verbosity=verbosity
    )


def exact_evolution_powers(hamiltonian, evolution_time, n_phase_bits):
    """
    Build the exact evolution unitaries :math:`U(t \\, 2^k)` for the QPE sequence.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object from the QPE-Toolbox ``Hamiltonian`` class.
    evolution_time : float
        Total evolution time ``t``.
    n_phase_bits : int
        Number of phase estimation qubits.

    Returns
    -------
    unitaries : list of list of :quimb-api:`Gate`
        ``unitaries[k]`` holds the single dense gate implementing
        :math:`U(t \\, 2^k) = e^{-i H t 2^k}` on data-register-local qubits,
        without controls, as expected by ``qpe_circuit`` and ``qpe_gates``.
    """
    data_reg = list(range(hamiltonian.n_qubits))
    return [
        [hamiltonian.get_U_exact(evolution_time * 2**k, data_reg)]
        for k in range(n_phase_bits)
    ]


def trotter_evolution_gates(hamiltonian, evolution_time, dt, *, trotter_order=1):
    """
    Build the gate sequence of one Trotterized evolution :math:`U(t)`.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object from the QPE-Toolbox ``Hamiltonian`` class.
    evolution_time : float
        Evolution time ``t``.
    dt : float
        Trotter step size, must be real > 0.
    trotter_order : int, default ``1``
        Order of the Trotter decomposition.

    Returns
    -------
    generator of :quimb-api:`Gate`
        Lazily yields the gates of the Trotterized :math:`U(t)` on
        data-register-local qubits, without controls, as expected by
        ``qpe_circuit``, ``qpe_gates`` and the Hadamard test. The generator
        is one-shot: it can be consumed only once.

    Notes
    -----
    - The number of Trotter steps is ``round(evolution_time / dt)``; a warning
      is raised when ``dt`` exceeds ``evolution_time``.
    """
    if not (np.isscalar(dt) and np.isreal(dt) and dt > 0):
        raise ValueError(f"dt must be real > 0, got {dt}")
    if dt > evolution_time:
        warnings.warn(
            f"dt={dt:.3f} > evolution_time={evolution_time:.3f}",
            stacklevel=2,
        )
    data_reg = list(range(hamiltonian.n_qubits))
    trotter_slice = hamiltonian.get_trotter_step(dt, data_reg, trotter_order)
    n_steps = int(evolution_time / dt + 1 / 2)
    return _repeat_gates(trotter_slice, n_steps)


def trotter_evolution_powers(
    hamiltonian, evolution_time, dt, n_phase_bits, *, trotter_order=1
):
    """
    Build the Trotterized evolution unitaries :math:`U(t \\, 2^k)` for the QPE sequence.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object from the QPE-Toolbox ``Hamiltonian`` class.
    evolution_time : float
        Total evolution time ``t``.
    dt : float
        Trotter step size, must be real > 0.
    n_phase_bits : int
        Number of phase estimation qubits.
    trotter_order : int, default ``1``
        Order of the Trotter decomposition.

    Returns
    -------
    unitaries : list of generator of :quimb-api:`Gate`
        ``unitaries[k]`` lazily yields the gates of the Trotterized
        :math:`U(t \\, 2^k)` on data-register-local qubits, without controls,
        as expected by ``qpe_circuit`` and ``qpe_gates``. Each generator is
        one-shot: it can be consumed only once.

    Notes
    -----
    - A warning is raised when ``dt`` exceeds the evolution time of a power.
    """
    return [
        trotter_evolution_gates(
            hamiltonian, evolution_time * 2**k, dt, trotter_order=trotter_order
        )
        for k in range(n_phase_bits)
    ]


def _repeat_gates(gate_ids, n_steps):
    """Lazily yield the gates of ``n_steps`` repetitions of a gate instruction list."""
    for _ in range(n_steps):
        for gate_id in gate_ids:
            yield parse_to_gate(*gate_id)


def _evolution_powers(hamiltonian, evolution_time, dt, n_phase_bits, trotter_order):
    """Dispatch between exact and Trotterized evolution unitaries."""
    if dt is EXACT:
        return exact_evolution_powers(hamiltonian, evolution_time, n_phase_bits)
    if not (np.isscalar(dt) and np.isreal(dt) and dt > 0):
        raise ValueError(f"dt must be EXACT or real > 0, got {dt}")
    return trotter_evolution_powers(
        hamiltonian, evolution_time, dt, n_phase_bits, trotter_order=trotter_order
    )


def _qpe_gates_filename(hamiltonian, evolution_time, dt, trotter_order, n_phase_bits):
    """Build the file name for QPE gate serialization."""
    if dt is EXACT:
        raise ValueError("Cannot write gates for exact time evolution")
    n_steps = int(evolution_time / dt)
    return f"QPE_ttr{trotter_order}{n_steps}steps_{hamiltonian.n_qubits}qubits_{n_phase_bits}phbits"


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
        raise ValueError(f"Invalid size_interval: {size_interval}")
    E_const = getattr(hamiltonian, "e_const", 0.0)
    Emax = E_target - E_const + size_interval / 2
    evolution_time = 2 * np.pi / size_interval
    global_phase = Emax * evolution_time

    return E_const, Emax, evolution_time, global_phase
