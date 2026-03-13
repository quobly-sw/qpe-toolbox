# --------------------------------------------------------------------------------------
# This file is part of qpe-toolbox.
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See LICENSE.txt and NOTICE.txt in the
# project root.
#
# --------------------------------------------------------------------------------------

import numpy as np
import quimb.tensor as qtn

##### Kronecker products ######################################################


def kron_mpos(mpo1, mpo2):
    """
    Construct the Kronecker (tensor) product of two MPOs.

    This returns an MPO representing :math:`\\mathrm{MPO}_1 \\otimes \\mathrm{MPO}_2`,
    with tensors arranged in the left, right, up, down index ordering

    The function supports both single-site and multi-site MPOs and handles
    boundary tensor reshaping explicitly.

    Parameters
    ----------
    mpo1 : :quimb-api:`MatrixProductOperator`
        First MPO operand.
    mpo2 : :quimb-api:`MatrixProductOperator`
        Second MPO operand.

    Returns
    -------
    :quimb-api:`MatrixProductOperator`
        The Kronecker product MPO acting on the concatenated Hilbert space.

    Raises
    ------
    ValueError
        If the tensor shapes of either MPO are not compatible with the expected
        MPO boundary conventions.

    """
    # make sure MPOs tensors are in the order left, right, up, down
    mpo1.permute_arrays("lrud")
    mpo2.permute_arrays("lrud")

    arrays = []

    sites1 = list(mpo1.gen_sites_present())
    if mpo1.L > 1:
        for idx in sites1[:-1]:
            tensor = mpo1[idx]
            arrays.append(tensor.data)
        arrays.append(mpo1.tensors[-1].data[:, np.newaxis, :, :])
    elif (mpo1.tensors[sites1[0]].shape) == (2, 2):
        arrays.append(mpo1.tensors[sites1[0]].data[np.newaxis, ...])
    else:
        raise ValueError("Weird shape mpo1")

    sites2 = list(mpo2.gen_sites_present())
    if mpo2.L > 1:
        arrays.append(mpo2.tensors[0].data[np.newaxis, ...])
        for idx in sites2[1:]:
            tensor = mpo2[idx]
            arrays.append(tensor.data)
    elif (mpo2.tensors[sites2[0]].shape) == (2, 2):
        arrays.append(mpo2.tensors[sites2[0]].data[np.newaxis, ...])
    else:
        raise ValueError("Weird shape mpo2")

    return qtn.MatrixProductOperator(arrays)


def kron_mps(mps1, mps2, *, verbosity=0):
    """
    Construct the Kronecker (tensor) product of two MPS objects.

    This returns an MPS representing :math:`\\mathrm{MPS}_1 \\otimes \\mathrm{MPS}_2`,
    with tensors arranged in the left, right, physical index ordering.

    Parameters
    ----------
    mps1 : :quimb-api:`MatrixProductState`
        First MPS operand.
    mps2 : :quimb-api:`MatrixProductState`
        Second MPS operand.
    verbosity : int, default ``0``
        If ``> 0``, print the shapes of the resulting tensors.

    Returns
    -------
    :quimb-api:`MatrixProductState`
        The Kronecker product MPS on the combined physical register.

    Raises
    ------
    ValueError
        If the tensor shapes of either MPS are not compatible with the expected
        MPS boundary conventions.

    """
    # make sure MPS tensors are in the order left, right, physical
    mps1.permute_arrays("lrp")
    mps2.permute_arrays("lrp")

    arrays = []

    sites1 = list(mps1.gen_sites_present())
    if mps1.L > 1:
        for idx in sites1[:-1]:
            tensor = mps1[idx]
            arrays.append(tensor.data)
        arrays.append(mps1.tensors[-1].data[:, np.newaxis, :])
    elif mps1.tensors[sites1[0]].shape == (2,):
        arrays.append(mps1.tensors[sites1[0]].data[np.newaxis, ...])
    else:
        raise ValueError("Weird shape mps1")

    sites2 = list(mps2.gen_sites_present())
    if mps2.L > 1:
        arrays.append(mps2.tensors[sites2[0]].data[np.newaxis, :, :])
        for idx in sites2[1:]:
            tensor = mps2[idx]
            arrays.append(tensor.data)
    elif mps2.tensors[sites2[0]].shape == (2,):
        arrays.append(mps2.tensors[sites2[0]].data[np.newaxis, ...])
    else:
        raise ValueError("Weird shape mps2")

    if verbosity > 0:
        print([np.shape(array) for array in arrays])

    return qtn.MatrixProductState(arrays, shape="lrp")


######### Apply MPO to circuits ###############################################


def apply_gate_from_mpo(circ, mpo, *, compress=False, cutoff=1e-10, max_bond=0):
    """
    Apply an MPO-defined gate to a circuit state and return a new CircuitMPS.

    The MPO is applied to the circuit wavefunction ``circ.psi``. Optional
    compression can be performed during and/or after application.

    Parameters
    ----------
    circ : :quimb-api:`Circuit` or :quimb-api:`CircuitMPS`
        Input circuit whose state will be acted on.
    mpo : :quimb-api:`MatrixProductOperator`
        MPO representing the quantum gate or evolution operator.
    compress : bool, default ``False``
        Whether to compress the resulting MPS during application.
    cutoff : float, default ``1e-10``
        Singular value cutoff used during compression.
    max_bond : int, default ``0``
        Maximum allowed bond dimension. ``0`` means no explicit limit.

    Returns
    -------
    :quimb-api:`CircuitMPS`
        New circuit with the updated MPS state.

    Notes
    -----
    This function does not modify the input circuit in place.

    """
    psi = mpo.apply(circ.psi, compress=compress, cutoff=cutoff, max_bond=max_bond)
    return qtn.CircuitMPS(psi0=psi, cutoff=cutoff, max_bond=max_bond)


# Controls
#
## When the ancilla register is not implemented
#
### add extra control qubits


def add_creg_mpo(mpo, mpo_reg, creg, cket):
    """
    Add multiple control qubits to an MPO.

    The resulting MPO represents a controlled operation acting on the original
    MPO, conditioned on the control register being in a specified computational
    basis state.

    Parameters
    ----------
    mpo : :quimb-api:`MatrixProductOperator`
        Base MPO representing the target operation.
    mpo_reg : list[int]
        Indices of the physical register acted on by ``mpo``.
    creg : list[int]
        Control qubit indices, indexed with respect to the MPO ordering.
    cket : str or int
        Control state. Can be a bitstring (e.g. ``"11"``) or an integer encoding
        the computational basis state.

    Returns
    -------
    :quimb-api:`MatrixProductOperator`
        MPO with added control qubits.

    Raises
    ------
    ValueError
        If an unsupported control state is requested.
    TypeError
        If ``cket`` is neither ``str`` nor ``int``.

    Notes
    -----
    This function is not fully tested and should be used with care for
    multi-control configurations.

    """
    # make sure indices of each tensor in the MPO are in the order left, right, up, down
    mpo.permute_arrays("lrud")

    if len(creg) == 1:
        if cket in ["1", 1]:
            location = "after" if (mpo_reg[-1] < creg[0]) else "before"
            return add_cqubit_mpo(mpo, location)
        raise ValueError(f"{cket} on one control bit not implemented")

    m_c = len(creg)
    if isinstance(cket, str):
        ctrl_mps = qtn.MPS_computational_state(cket)
    elif isinstance(cket, (int, np.integer)):
        ctrl_mps = qtn.MPS_computational_state(f"{cket:0{m_c}b}")
    else:
        raise TypeError("cket must be int or str")
    projector = ctrl_mps.partial_trace_to_mpo(keep=list(range(m_c)))
    Id_creg = qtn.MPO_identity(m_c)
    Id_mpo = qtn.MPO_identity(mpo.L)
    if mpo_reg[-1] < creg[0]:
        return kron_mpos(Id_mpo, Id_creg - projector) + kron_mpos(mpo, projector)
    return kron_mpos(Id_creg - projector, Id_mpo) + kron_mpos(projector, mpo)


### add a single control qubit


def add_cqubit_mpo(mpo, location):
    """
    Add a single control qubit to an MPO.

    The control qubit is added either before or after the existing MPO,
    depending on register ordering.

    Parameters
    ----------
    mpo : :quimb-api:`MatrixProductOperator`
        Base MPO representing the target operation.
    location : {"before", "after"}
        Whether to add the control qubit before or after the MPO qubits.

    Returns
    -------
    :quimb-api:`MatrixProductOperator`
        MPO augmented with a single control qubit.

    Raises
    ------
    ValueError
        If ``location`` is not one of ``"before"`` or ``"after"``.

    """
    # make sure indices of each tensor in the MPO are in the order left, right, up, down
    mpo.permute_arrays("lrud")

    sites = list(mpo.gen_sites_present())
    if location == "before":
        arrays = [np.array([[[0, 0], [0, 1]]], dtype=mpo.dtype)]
        arrays.append(np.array([mpo[sites[0]].data]))
        for idx in sites[1:]:
            t = mpo[idx]
            arrays.append(t.data)

        start = np.array([[[1, 0], [0, 0]]], dtype=mpo.dtype)
        end = np.eye(2, dtype=mpo.dtype)[np.newaxis]

    elif location == "after":
        arrays = [mpo[idx].data for idx in sites[:-1]]
        t = mpo[sites[-1]].data
        arrays.append(t[:, np.newaxis, :, :])
        arrays.append(np.array([[[0, 0], [0, 1]]], dtype=mpo.dtype))

        start = np.eye(2, dtype=mpo.dtype)[np.newaxis]
        end = np.array([[[1, 0], [0, 0]]], dtype=mpo.dtype)
    else:
        raise ValueError("Invalid location")

    res = qtn.MatrixProductOperator(arrays)
    mid = [np.eye(2, dtype=mpo.dtype)[np.newaxis, np.newaxis]] * (mpo.L - 1)
    aux = qtn.MatrixProductOperator([start, *mid, end])
    return res + aux


## when the ancilla register is already implemented

### control on one qubit being in a given value


def controlled_mpo(mpo, phys_reg, aux_reg, k_ctrl, *, ctrl=1):
    """
    Construct an MPO controlled on an auxiliary qubit being in a given state.

    The MPO is assumed to be of the form ``Id ⊗ U``, where ``Id`` acts on the
    auxiliary register and ``U`` acts on the physical register.

    Parameters
    ----------
    mpo : :quimb-api:`MatrixProductOperator`
        Input MPO in the form ``Id ⊗ U``.
    phys_reg : list[int]
        Indices of the physical register qubits.
    aux_reg : list[int]
        Indices of the auxiliary (control) register qubits.
    k_ctrl : int
        Index of the control qubit relative to ``aux_reg``.
    ctrl : int, default ``1``
        Control value (``0`` or ``1``) conditioning the operation.

    Returns
    -------
    :quimb-api:`MatrixProductOperator`
        Controlled MPO.

    Raises
    ------
    ValueError
        If the register ordering assumption is violated.

    Notes
    -----
    This implementation assumes that all auxiliary-register tensors initially
    correspond to identity operators.

    """
    # make sure indices of each tensor in the MPO are in the order left, right, up, down
    mpo.permute_arrays("lrud")

    sites = list(mpo.gen_sites_present())
    if phys_reg[0] < aux_reg[-1]:
        raise ValueError("only implemented for min(phys_reg) > max(aux_reg)")
    if mpo[sites[aux_reg[-1]]].data.shape != (1, 1, 2, 2):
        raise ValueError("Invalid MPO tensor shape")
    if not np.allclose(mpo[sites[aux_reg[-1]]].data, np.eye(2), atol=1e-12):
        raise ValueError("Invalid last MPO tensor")

    projectors = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=mpo.dtype)
    arrays1 = [mpo[s].data for s in sites]

    # due to quimb data structure, need to access tensor.data to avoid aliasing
    mpo2 = qtn.MPO_identity(len(phys_reg + aux_reg), dtype=mpo.dtype)
    arrays2 = [mpo2[s].data for s in sites]

    # first site has only 3 legs
    sh = (1,) * (k_ctrl != aux_reg[0]) + (1, 2, 2)
    arrays1[sites[k_ctrl]] = projectors[ctrl].reshape(sh)
    arrays2[sites[k_ctrl]] = projectors[(ctrl + 1) % 2].reshape(sh)

    mpo1 = qtn.MatrixProductOperator(arrays1)
    mpo2 = qtn.MatrixProductOperator(arrays2)
    return mpo1 + mpo2
