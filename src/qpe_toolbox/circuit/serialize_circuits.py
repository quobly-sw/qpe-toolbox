# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

import io
import re

import numpy as np
import quimb.tensor as qtn
from qiskit import ClassicalRegister, QuantumCircuit
from quimb.tensor.circuit import parse_openqasm2_file


def apply_gate_qiskit(qc, label, qubits, params):
    """Apply a quantum gate to a ``qiskit`` :qiskit-api:`QuantumCircuit` using a string label.

    This function dispatches gate application based on:

        - the gate label (case-insensitive),

        - the number of target qubits (single- or two-qubit),

        - and whether the gate is parameterized.

    It is inspired in the :quimb-api:`Circuit.apply_gate` method from `quimb`'s :quimb-api:`Circuit`.

    The alias ``"cnot"`` is automatically mapped to ``"cx"``.

    Parameters
    ----------
    qc : :qiskit-api:`QuantumCircuit`
        The quantum circuit to which the gate is applied.

    label : str
        Gate label identifying the quantum operation. The comparison is
        case-insensitive. For example, ``"RX"``, ``"rx"``, and ``"Rx"``
        are treated identically. The label ``"cnot"`` is mapped internally
        to ``"cx"``.

    qubits : list[int]
        List of qubit indices the gate acts on. Supported values are:

            - length 1 for single-qubit gates,

            - length 2 for two-qubit gates.

        Any other length raises a ``ValueError``.

    params : list[float]
        List of gate parameters. Use an empty list for non-parametrized
        gates. For parameterized gates, the number of parameters must match
        the gate definition (e.g. one parameter for ``rx``, three for ``u3``).

    Raises
    ------
    ValueError
        If the number of qubits is not supported (i.e. not 1 or 2).

    AttributeError
        If the gate label does not correspond to a qiskit gate.

    Notes
    -----
    - Qubit index bounds are not checked and must be valid for the circuit.

    Examples
    --------
    Apply a single-qubit rotation:

    >>> apply_gate_qiskit(qc, "RX", [0], [np.pi / 2])

    Apply a controlled-NOT gate:

    >>> apply_gate_qiskit(qc, "cnot", [0, 1], [])

    Apply a two-qubit parameterized gate:

    >>> apply_gate_qiskit(qc, "rxx", [0, 1], [0.3])

    """

    label = label.lower()
    if label == "cnot":
        label = "cx"

    # Check the number of qubits
    n_qubits = len(qubits)
    if len(qubits) not in (1, 2):
        raise ValueError(f"Unsupported number of qubits: {n_qubits}")
    getattr(qc, label)(*params, *qubits)


def deserialize_to_qiskit_QuantumCircuit(
    full_gate_dict, *, max_depth=np.inf, measure=False
):
    """Deserialize a gate dictionary into a ``qiskit`` :qiskit-api:`QuantumCircuit`.

    This function reconstructs a :qiskit-api:`QuantumCircuit` from a serialized gate
    representation, where gates are annotated with qubit indices, parameters,
    and a discrete circuit ``round`` (layer index).

    Gates are applied in the order they appear in ``full_gate_dict["gates"]``,
    subject to an optional depth cutoff.

    Parameters
    ----------
    full_gate_dict : dict
        Dictionary encoding the quantum circuit. It must contain:

        - ``"n_qubits"`` : int
            Total number of qubits in the circuit.

        - ``"gates"`` : list[dict]
            List of gate specifications. Each gate dictionary must contain:

            - ``"name"`` : str
                Gate label (passed to :func:`apply_gate_qiskit`).

            - ``"qubits"`` : list[int]
                Target qubit indices.

            - ``"params"`` : list[float]
                Gate parameters (empty for non-parameterized gates).

            - ``"round"`` : int
                Layer index at which the gate is applied.

    max_depth : int or inf, optional
        Maximum circuit depth (number of rounds) to load. Only gates with
        ``gate["round"] < max_depth`` are applied. If ``inf`` (default), the full
        circuit is reconstructed.

    measure : bool, optional
        If ``True``, a classical register of size ``n_qubits`` is added and
        all qubits are measured at the end of the circuit. Default is ``False``.

    Returns
    -------
    qc : :qiskit-api:`QuantumCircuit`
        The reconstructed ``qiskit`` quantum circuit.

    Raises
    ------
    KeyError
        If required keys are missing from ``full_gate_dict`` or its gate
        entries.

    ValueError
        If invalid gate specifications are encountered during deserialization
        (propagated from :func:`apply_gate_qiskit`).

    Notes
    -----
    - Gate ordering within the same round is preserved as given in the input.
    - No validation is performed on qubit index bounds.
    - Parameter consistency is delegated to :func:`apply_gate_qiskit` and ``qiskit``.

    """
    N = int(full_gate_dict["n_qubits"])
    qc = QuantumCircuit(N)
    for gate in full_gate_dict["gates"]:
        if gate["round"] < max_depth:
            apply_gate_qiskit(qc, gate["name"], gate["qubits"], gate["params"])

    if measure:
        qc.add_register(ClassicalRegister(N))
        qc.measure(range(N), range(N))

    return qc


def serialize_from_quimb_Circuit(qc):
    """Serialize a ``quimb`` circuit into a ``JSON``-compatible dictionary.

    This function converts a :quimb-api:`Circuit` object into
    a plain ``python`` dictionary containing only ``JSON``-serializable types.
    The resulting dictionary can be safely stored, transmitted, or
    deserialized into other circuit representations (e.g. ``qiskit``).

    Parameters
    ----------
    qc : :quimb-api:`Circuit`
        The ``quimb`` circuit to be serialized.

    Returns
    -------
    dict
        Dictionary representation of the circuit with the following keys:

        - ``"n_qubits"`` : int
          Number of qubits in the circuit.

        - ``"gates"`` : list of dict
          Ordered list of gate specifications. Each gate dictionary
          contains:

            - ``"name"`` : str
              Gate label as used by ``quimb``.

            - ``"qubits"`` : list of int
              Target qubit indices.

            - ``"controls"``: list
                Control qubits' indices.

            - ``"params"`` : list of float
              Gate parameters (rounded to 4 decimal places).

            - ``"round"`` : int
              Circuit round (layer index) in which the gate appears.

    Notes
    -----
    - All numerical values are explicitly cast to built-in ``python`` types
      (``int`` and ``float``) to ensure ``JSON`` compatibility.
    - The ordering of gates in ``qc.gates`` is preserved, allowing
      faithful reconstruction of the circuit.
    - The output format is designed to be compatible with downstream
      deserialization into other frameworks (e.g. ``qiskit``).

    See Also
    --------
    deserialize_to_quimb_Circuit :
        Reconstruct a ``quimb`` :quimb-api:`Circuit` from the serialized dictionary.

    deserialize_to_qiskit_QuantumCircuit :
        Reconstruct a ``qiskit`` :qiskit-api:`QuantumCircuit` from the serialized dictionary.

    """
    return serialize_from_quimb_gates(qc.N, qc.gates)


def serialize_from_quimb_gates(n_qubits, gates_list):
    """Serialize a list of ``quimb`` gates into a ``JSON``-compatible dictionary.

    This function converts a list of :quimb-api:`Gate` objects into
    a plain ``python`` dictionary containing only ``JSON``-serializable types.
    The resulting dictionary can be safely stored, transmitted, or
    deserialized into other circuit representations (e.g. ``qiskit``).

    Parameters
    ----------
    n_qubits : int
        Total number of qubits in the circuit.
    gates_list : list
        The list of ``quimb`` gates to be serialized.

    Returns
    -------
    dict
        Dictionary representation of the circuit with the following keys:

        - ``"n_qubits"`` : int
          Number of qubits in the circuit.

        - ``"gates"`` : list of dict
          Ordered list of gate specifications. Each gate dictionary
          contains:

            - ``"name"`` : str
              Gate label as used by ``quimb``.

            - ``"qubits"`` : list of int
              Target qubit indices.

            - ``"controls"``: list
                Control qubits' indices.

            - ``"params"`` : list of float
              Gate parameters (rounded to 4 decimal places).

            - ``"round"`` : int
              Circuit round (layer index) in which the gate appears.

    Notes
    -----
    - All numerical values are explicitly cast to built-in ``python`` types
      (``int`` and ``float``) to ensure ``JSON`` compatibility.
    - The ordering of gates is preserved, allowing
      faithful reconstruction of the circuit.
    - The output format is designed to be compatible with downstream
      deserialization into other frameworks (e.g. ``qiskit``).

    See Also
    --------
    deserialize_to_quimb_Circuit :
        Reconstruct a ``quimb`` :quimb-api:`Circuit` from the serialized dictionary.

    deserialize_to_qiskit_QuantumCircuit :
        Reconstruct a ``qiskit`` :qiskit-api:`QuantumCircuit` from the serialized dictionary.

    """
    return {
        "n_qubits": n_qubits,
        "gates": [
            {
                "name": gate.label,
                "qubits": [int(q) for q in gate.qubits],
                "params": [float(p) for p in gate.params],
                "controls": [int(q) for q in gate.controls]
                if gate.controls is not None
                else [],
                "round": int(gate.round) if gate.round is not None else None,
            }
            for gate in gates_list
        ],
    }


def deserialize_to_quimb_Circuit(
    full_gate_dict, *, max_depth=np.inf, contract=False, **gate_opts
):
    """Deserialize a gate dictionary into a :quimb-api:`Circuit` up to a given depth.

    This function reconstructs a circuit from a serialized
    representation of a quantum circuit (i.e. the one that can be saved as JSON).
    Only gates whose ``round`` index is strictly smaller than ``max_depth`` are included,
    allowing for partial reconstruction of the circuit.

    Parameters
    ----------
    full_gate_dict : dict
        Serialized circuit description. Must contain the following keys:

        - ``"n_qubits"`` : int or str
            Total number of qubits in the circuit.

        - ``"gates"`` : list of dict
            List of gate specifications. Each gate dictionary must contain:

            - ``"name"`` : str
                Gate identifier understood by :quimb-api:`Circuit.apply_gate`.

            - ``"params"`` : list
                Gate parameters (will be cast to ``float``).

            - ``"qubits"`` : list
                Qubit indices the gate acts on.

            - ``"controls"``: list
                Control qubits' indices.

            - ``"round"`` : int or str
                Layer / round index of the gate.

    max_depth : int or inf, optional.
        Maximum circuit depth to deserialize, i.e. if ``round >= depth`` the gate is
        ignored. Default is inf, the full circuit is deserialized.

    contract : bool, optional
        Whether to contract the quimb Circuit. Default is False.

    gate_opts : Supplied to the gate function, options here will override the default gate_opts

    Returns
    -------
    qc : :quimb-api:`Circuit`
        A ``quimb`` circuit instance containing all gates up to the specified depth.

    Notes
    -----
    - If the circuit is deserialized for plotting purposes, set ``contract=False``.
    - This function assumes the serialized gate names and parameters are
      compatible with :quimb-api:`Circuit.apply_gate`.
    - Useful when assessing the complexity of a circuit and its
      contraction layer-by-layer.

    """
    qc = qtn.Circuit(N=int(full_gate_dict["n_qubits"]))

    for gate in full_gate_dict["gates"]:
        if gate["round"] < max_depth:
            qc.apply_gate(
                gate_id=gate["name"],
                params=[float(p) for p in gate["params"]],
                qubits=[int(q) for q in gate["qubits"]],
                controls=[int(q) for q in gate.get("controls", [])],
                gate_round=int(gate["round"]),
                contract=contract,
                **gate_opts,
            )

    return qc


def deserialize_to_quimb_CircuitMPS(
    full_gate_dict,
    max_bond,
    cutoff,
    *,
    max_depth=np.inf,
    perm=False,
    psi0=None,
):
    """Deserialize a gate dictionary into a :quimb-api:`CircuitMPS` or
    :quimb-api:`CircuitPermMPS` up to a given depth from a serialized
    representation of a quantum circuit (i.e., the same format as saved in JSON).
    Only gates whose `round` index is strictly smaller than ``max_depth`` are applied,
    allowing for partial reconstruction of the circuit.

    Parameters
    ----------
    full_gate_dict : dict
        Serialized circuit description. Must contain the following keys:

        - ``"n_qubits"`` : int or str
            Total number of qubits in the circuit.
        - ``"gates"`` : list
            List of gate specifications. Each gate dictionary must contain:

            - ``"name"`` : str
                Gate identifier understood by
                :quimb-api:`CircuitMPS.apply_gate`
            - ``"params"`` : list
                Gate parameters (will be cast to float).
            - ``"qubits"`` : list
                Qubit indices the gate acts on.
            - ``"controls"``: list
                Control qubits' indices.
            - ``"round"`` : int or str
                Layer/depth index of the gate.

    max_bond : int
        Maximum bond dimension of the MPS.

    cutoff : float
        Truncation cutoff for singular values when applying gates to the MPS.

    max_depth : int or inf, optional.
        Maximum circuit depth to deserialize, i.e. if ``round >= max_depth`` the gate
        is ignored. Default is inf, the full circuit is deserialized.

    perm : bool, optional
        If ``True``, use :quimb-api:`CircuitPermMPS` instead of
        :quimb-api:`CircuitMPS`. Default is False.

    Returns
    -------
    cmps : :quimb-api:`CircuitMPS` or \
           :quimb-api:`CircuitPermMPS`
        An MPS representation of the reconstructed circuit containing
        all gates contracted up to the specified depth.

    """
    N = int(full_gate_dict["n_qubits"])
    if perm:
        cmps = qtn.CircuitPermMPS(
            N=N,
            psi0=psi0,
            max_bond=max_bond,
            cutoff=cutoff,
        )
    else:
        cmps = qtn.CircuitMPS(
            N=N,
            psi0=psi0,
            max_bond=max_bond,
            cutoff=cutoff,
        )

    for gate in full_gate_dict["gates"]:
        if gate["round"] < max_depth:
            cmps.apply_gate(
                gate_id=gate["name"],
                params=np.asarray(gate["params"], dtype=float),
                qubits=[int(q) for q in gate["qubits"]],
                controls=[int(q) for q in gate["controls"]],
            )
    return cmps


def dump_quimb_Circuit_to_qasm(circ, savefile_base, *, save_rounds=True):
    """
    Export a ``quimb`` circuit to an ``OpenQASM 2.0`` file.

    This function serializes a :quimb-api:`Circuit` into a
    QASM 2.0-compatible text file. Optionally, the circuit round
    (layer index) of each gate is stored in a separate sidecar file.

    Parameters
    ----------
    circ : :quimb-api:`Circuit`
        The ``quimb`` circuit to export.

    savefile_base : str
        Base filename (without extension) for the output files.
        The function writes:

            - ``<savefile_base>.qasm``
            - ``<savefile_base>_rounds.txt`` (if ``save_rounds=True``)

    save_rounds : bool, optional
        If ``True`` (default), write the circuit round of each gate
        to a separate text file, with one integer per line.

    Raises
    ------
    ValueError
        If a gate label is encountered that cannot be mapped to a
        supported OpenQASM 2.0 instruction.

    Notes
    -----
    - Gate labels are converted to lowercase for QASM compatibility.
    - The gate label ``"cnot"`` is automatically mapped to ``"cx"``,
      which is the canonical OpenQASM name.
    - Gate parameters are converted to native Python ``float`` and
      formatted with limited precision to ensure portability.
    - Circuit round information is *not* part of the QASM standard and
      is therefore stored separately.

    """
    qasm_lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{circ.N}];",
    ]

    with (
        open(f"{savefile_base}_rounds.txt", "w") if save_rounds else io.StringIO()
    ) as writer:
        for gate in circ.gates:
            label = gate.label.lower()
            qubits = gate.qubits
            params = gate.params
            if label == "cnot":  # 'cnot' not recognized by qiskit
                label = "cx"

            # Convert to native types for JSON/QASM safety
            param_str = ",".join(f"{float(p):.8g}" for p in params)
            qubit_str = ",".join(f"q[{int(q)}]" for q in qubits)

            if gate.label.upper() in qtn.circuit.ALL_PARAM_GATES:
                qasm_lines.append(f"{label.lower()}({param_str}) {qubit_str};")

            elif (
                gate.label.upper() in qtn.circuit.ONE_QUBIT_GATES
                or gate.label.upper() in qtn.circuit.TWO_QUBIT_GATES
            ):
                qasm_lines.append(f"{label.lower()} {qubit_str};")

            else:
                raise ValueError(f"Unsupported gate: {gate.label}")

            # Write rounds if enabled
            writer.write(f"{gate.round}\n")

    # Write QASM file
    with open(f"{savefile_base}.qasm", "w") as out:
        out.write("\n".join(qasm_lines))


def load_qasm_to_quimb_Circuit(
    filename,
    *,
    with_rounds=False,
    max_depth=np.inf,
    gate_contract=False,
    min_layout=False,
):
    """This function parses a QASM 2.0 file and reconstructs a
    :quimb-api:`Circuit`. Optionally, it can restore
    circuit round (layer) information from a sidecar file and
    load only a truncated circuit depth.

    Parameters
    ----------
    filename : str
        Base filename (without extension) of the QASM file to load.

    with_rounds : bool, optional
        If ``True``, load circuit round information from
        ``filename_rounds.txt`` and assign gates using
        ``gate_round``. Default is ``False``.

    max_depth : int or inf, optional
        Maximum circuit depth to load. Gates with ``gate_round >= max_depth`` are
        ignored. Default is inf, the full circuit is loaded.

    gate_contract : bool, optional
        Whether to immediately contract gates into the tensor
        network when applying them. Passed directly to
        ``circ.apply_gate``. Default is ``False``.

    min_layout : bool, optional
        If ``True``, infer the number of qubits from the maximum qubit
        index appearing in the gates.
        If ``False`` (default), read the register size from the QASM
        header.

    Returns
    -------
    circ : :quimb-api:`Circuit`
        The reconstructed quimb circuit.

    Notes
    -----
    - The QASM file is parsed using ``parse_openqasm2_file``.
    - When ``with_rounds=False``, circuit round information is ignored
      and gates are loaded in sequential order.
    - When ``with_rounds=True``, a sidecar file
      ``<filename>_rounds.txt`` must exist and contain one integer per
      gate.
    - This function assumes that the QASM file uses gate labels
      compatible with quimb.

    """
    # Load the gates
    parsed_qasm = parse_openqasm2_file(filename + ".qasm")
    gates = parsed_qasm["gates"]

    # Find the size of the circuit
    if min_layout:
        # Method I: simpler but assumes that gates act on all qubits
        N = 1 + max(q for gate in gates for q in gate["qubits"])
    else:
        # Method II: read the number of qubits from the header
        with open(filename + ".qasm") as f:
            next(f)
            next(f)
            line = f.readline()
            N = int(re.findall(r"q\[(\d+)\]", line)[0])

    if with_rounds:
        # Build a circuit with the `gate_round` information
        circ = qtn.Circuit(N=N)
        with open(filename + "_rounds.txt") as f:
            rounds = [int(line.strip()) for line in f]

        for i, gate in enumerate(gates):
            if rounds[i] < max_depth:  # load only rounds below depth
                circ.apply_gate(
                    gate_id=gate.label,
                    params=[float(p) for p in gate.params],
                    qubits=[int(q) for q in gate.qubits],
                    gate_round=int(rounds[i]),
                    contract=gate_contract,  # other options in quimb.tensor.tensor_1d.gate_TN_1D
                )
        return circ
    # Build a circuit without the `gate_round` information
    return qtn.Circuit.from_gates(N=N, gates=gates, gate_contract=False)
