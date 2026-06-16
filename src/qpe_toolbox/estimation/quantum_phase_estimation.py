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

import numpy as np
from quimb.tensor.circuit import parse_to_gate

from qpe_toolbox import EXACT
from qpe_toolbox.circuit import count_gates
from qpe_toolbox.circuit.serialize_circuits import serialize_from_quimb_gates

from .qpe_circuit import qpe_circuit, qpe_first_stage_circuit, qpe_gates


def qpe_energy(
    hamiltonian,
    initial_circ,
    n_trotter_steps,
    E_target,
    size_interval,
    *,
    trotter_order=1,
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
    n_trotter_steps : int or qpe_toolbox.EXACT
        Number of Trotter steps for the ``U(t)`` evolution, or ``EXACT`` for
        exact evolution.
    E_target : float
        Central target energy for the search window.
    size_interval : float
        Width of the energy search interval.
    trotter_order : int, default ``1``
        Order of the Trotter decomposition.
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

    traces, probs = qpe_sample(
        hamiltonian,
        initial_circ,
        evolution_time,
        n_trotter_steps,
        global_phase,
        trotter_order=trotter_order,
        optimize=optimize,
        verbosity=verbosity - 1,
    )

    probs_flat = np.ravel(probs).astype(float)
    first_thetas = [(i, probs_flat[i]) for i in probs_flat.argsort()[::-1][:5]]
    highest_prob_state, highest_prob = first_thetas[0]
    traces["first_thetas"] = first_thetas
    traces["prob"] = float(highest_prob)  # float here is for JSON

    if verbosity >= 1:
        for m, p in first_thetas:
            print(
                f"{m:b}".zfill(n_phase_bits),
                f"|{m}>",
                f"{m / 2**n_phase_bits:<{n_phase_bits + 2}}",
                f"{p:<6.4f}",
            )

    theta = highest_prob_state / 2**n_phase_bits
    energy = Emax - 2 * np.pi * theta / evolution_time
    energy -= E_const

    return traces, energy


def qpe_sample(
    hamiltonian,
    initial_circ,
    evolution_time,
    n_trotter_steps,
    global_phase,
    *,
    trotter_order=1,
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
    n_trotter_steps : int or qpe_toolbox.EXACT
        Number of Trotter steps for the ``U(t)`` evolution; if ``EXACT``,
        evolution is exact.
    global_phase : float
        Global phase added to the controlled-U operations.
    trotter_order : int, default ``1``
        Order of Trotter decomposition for time evolution.
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
        hamiltonian, evolution_time, n_trotter_steps, n_phase_bits, trotter_order
    )
    traces, circ = qpe_circuit(
        initial_circ, unitaries, global_phase=global_phase, verbosity=verbosity
    )
    traces["gates_count"] = count_gates(circ)

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
    n_trotter_steps,
    global_phase,
    *,
    trotter_order=1,
    savefile=None,
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
    n_trotter_steps : int or qpe_toolbox.EXACT
        Number of Trotter steps for the ``U(t)`` evolution; if ``EXACT``,
        evolution is exact.
    global_phase : float
        Global phase added to the controlled-U operations.
    trotter_order : int, default ``1``
        Order of Trotter decomposition for time evolution.
    savefile : str or os.PathLike or None, default ``None``
        If not ``None``, path to which the serialized gates are written as JSON.
        Not supported for exact time evolution (``n_trotter_steps=EXACT``).

    Returns
    -------
    traces : dict
        Dictionary containing the gate counts.
    gates_list : list of :quimb-api:`Gate`
        Gate sequence of the full QPE circuit.

    Raises
    ------
    ValueError
        If ``savefile`` is given together with exact time evolution
        (``n_trotter_steps=EXACT``).
    """
    unitaries = _evolution_powers(
        hamiltonian, evolution_time, n_trotter_steps, n_phase_bits, trotter_order
    )
    gates_list = list(qpe_gates(unitaries, global_phase=global_phase))
    traces = {"gates_count": count_gates(gates_list)}

    if savefile is not None:
        if n_trotter_steps is EXACT:
            raise ValueError("Cannot write gates for exact time evolution")
        gate_dict = serialize_from_quimb_gates(
            n_phase_bits + hamiltonian.n_qubits, gates_list
        )
        with open(savefile, "w") as outfile:
            json.dump(gate_dict, outfile)

    return traces, gates_list


def qpe_first_stage(
    hamiltonian,
    initial_circ,
    evolution_time,
    n_trotter_steps,
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
    n_trotter_steps : int or qpe_toolbox.EXACT
        Number of Trotter steps for the ``U(t)`` evolution; ``EXACT`` for exact
        evolution.
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
    """
    n_phase_bits = initial_circ.N - hamiltonian.n_qubits
    unitaries = _evolution_powers(
        hamiltonian, evolution_time, n_trotter_steps, n_phase_bits, trotter_order
    )
    return qpe_first_stage_circuit(
        initial_circ, unitaries, global_phase=global_phase, verbosity=verbosity
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


def trotter_evolution_gates(
    hamiltonian, evolution_time, n_trotter_steps, *, trotter_order=1
):
    """
    Build the gate sequence of one Trotterized evolution :math:`U(t)`.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object from the QPE-Toolbox ``Hamiltonian`` class.
    evolution_time : float
        Evolution time ``t``.
    n_trotter_steps : int
        Number of Trotter steps, a positive integer. The Trotter step size is
        ``dt = evolution_time / n_trotter_steps``.
    trotter_order : int, default ``1``
        Order of the Trotter decomposition.

    Returns
    -------
    generator of :quimb-api:`Gate`
        Lazily yields the gates of the Trotterized :math:`U(t)` on
        data-register-local qubits, without controls, as expected by
        ``qpe_circuit``, ``qpe_gates`` and the Hadamard test. The generator
        is one-shot: it can be consumed only once.
    """
    trotter_slice = _trotter_slice(
        hamiltonian, evolution_time, n_trotter_steps, trotter_order
    )
    return _repeat_gates(trotter_slice, n_trotter_steps)


def trotter_evolution_powers(
    hamiltonian, evolution_time, n_trotter_steps, n_phase_bits, *, trotter_order=1
):
    """
    Build the Trotterized evolution unitaries :math:`U(t \\, 2^k)` for the QPE sequence.

    The Trotter step size ``dt = evolution_time / n_trotter_steps`` is kept
    constant across powers: power ``k`` uses ``n_trotter_steps * 2**k`` steps.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object from the QPE-Toolbox ``Hamiltonian`` class.
    evolution_time : float
        Total evolution time ``t``.
    n_trotter_steps : int
        Number of Trotter steps for the ``U(t)`` evolution, a positive integer.
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
    """
    # dt = evolution_time / n_trotter_steps is constant across powers, so the
    # Trotter step is built once and reused for every power.
    trotter_slice = _trotter_slice(
        hamiltonian, evolution_time, n_trotter_steps, trotter_order
    )
    return [
        _repeat_gates(trotter_slice, n_trotter_steps * 2**k)
        for k in range(n_phase_bits)
    ]


def _trotter_slice(hamiltonian, evolution_time, n_trotter_steps, trotter_order):
    """Build one Trotter step's gate-id list for ``dt = evolution_time / n_trotter_steps``."""
    if not (isinstance(n_trotter_steps, (int, np.integer)) and n_trotter_steps > 0):
        raise ValueError(
            f"n_trotter_steps must be a positive integer, got {n_trotter_steps}"
        )
    dt = evolution_time / n_trotter_steps
    data_reg = list(range(hamiltonian.n_qubits))
    return hamiltonian.get_trotter_step(dt, data_reg, trotter_order)


def _repeat_gates(gate_ids, n_trotter_steps):
    """Lazily yield the gates of ``n_trotter_steps`` repetitions of an instruction list."""
    for _ in range(n_trotter_steps):
        for gate_id in gate_ids:
            yield parse_to_gate(*gate_id)


def _evolution_powers(
    hamiltonian, evolution_time, n_trotter_steps, n_phase_bits, trotter_order
):
    """Dispatch between exact and Trotterized evolution unitaries."""
    if n_trotter_steps is EXACT:
        return exact_evolution_powers(hamiltonian, evolution_time, n_phase_bits)
    return trotter_evolution_powers(
        hamiltonian,
        evolution_time,
        n_trotter_steps,
        n_phase_bits,
        trotter_order=trotter_order,
    )


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
