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

##### State preparation #######################################################


def state_preparation_mpo(state_mps):
    r"""
    Build the MPO mapping the all-zero state to a target MPS.

    The MPO is the outer product :math:`2^n |\psi\rangle\langle 0|` on :math:`n`
    qubits. The :math:`2^n` factor normalizes the transpilation overlap computed by
    :func:`~qpe_toolbox.circuit.transpile_mpo_to_circuit`.

    Parameters
    ----------
    state_mps : :quimb-api:`MatrixProductState`
        Target MPS to be reproduced by some circuit Ansatz, with arrays in
        ``'lpr'`` order.

    Returns
    -------
    :quimb-api:`MatrixProductOperator`
        Reference MPO for circuit state preparation.

    Raises
    ------
    ValueError
        If the arrays of ``state_mps`` are not in ``'lpr'`` order.
    """
    n_qubits = state_mps.num_tensors
    sites = [state_mps.site_ind(i) for i in range(n_qubits)]
    bonds = [state_mps.bond(i, i + 1) for i in range(n_qubits - 1)]
    # arrays are read positionally: require quimb's "lpr" layout, returned by DMRG2
    layout_error = (
        "state_mps arrays must be in 'lpr' order, "
        "call state_mps.permute_arrays('lpr') first"
    )
    ket0 = np.array([2.0, 0])  # normalization of the cost by 2**n_qubits

    if state_mps[0].inds != (sites[0], bonds[0]):
        raise ValueError(layout_error)
    arrays = [(state_mps[0].data[:, :, np.newaxis] * ket0).swapaxes(0, 1)]

    for i in range(1, n_qubits - 1):
        if state_mps[i].inds != (bonds[i - 1], sites[i], bonds[i]):
            raise ValueError(layout_error)
        # array has order lpr then get lprp' so transpose to lrpp'
        arrays.append((state_mps[i].data[:, :, :, np.newaxis] * ket0).swapaxes(1, 2))

    if state_mps[-1].inds != (bonds[-1], sites[-1]):
        raise ValueError(layout_error)
    # array has order lp then get lpp' so no transpose
    arrays.append(state_mps[-1].data[:, :, np.newaxis] * ket0)

    return qtn.MatrixProductOperator(arrays=arrays)


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


def controlled_mpo(mpo, k_ctrl, *, ctrl=1):
    """
    Construct an MPO controlled on one of its qubits being in a given state.

    The control qubit must carry the identity in ``mpo``, which is replaced by a
    projector; the rest of ``mpo`` becomes the controlled operation.

    Parameters
    ----------
    mpo : :quimb-api:`MatrixProductOperator`
        Input MPO, acting as the identity on site ``k_ctrl``.
    k_ctrl : int
        Site index of the control qubit.
    ctrl : int, default ``1``
        Control value (``0`` or ``1``) conditioning the operation.

    Returns
    -------
    :quimb-api:`MatrixProductOperator`
        Controlled MPO.

    Raises
    ------
    ValueError
        If ``mpo`` does not factorize as the identity on site ``k_ctrl``.
    """
    # make sure indices of each tensor in the MPO are in the order left, right, up, down
    mpo.permute_arrays("lrud")

    sites = list(mpo.gen_sites_present())
    # the control tensor is replaced by a projector, which is only valid if the MPO
    # factorizes there: trivial bonds on both sides and the identity acting on it
    sh = mpo[sites[k_ctrl]].data.shape
    if sh[:-2] != (1,) * (len(sh) - 2):
        raise ValueError("Invalid MPO tensor shape")
    if not np.allclose(mpo[sites[k_ctrl]].data, np.eye(2), atol=1e-12):
        raise ValueError("Invalid control MPO tensor")

    projectors = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=mpo.dtype)
    arrays1 = [mpo[s].data for s in sites]

    # due to quimb data structure, need to access tensor.data to avoid aliasing
    mpo2 = qtn.MPO_identity(len(sites), dtype=mpo.dtype)
    arrays2 = [mpo2[s].data for s in sites]

    arrays1[sites[k_ctrl]] = projectors[ctrl].reshape(sh)
    arrays2[sites[k_ctrl]] = projectors[(ctrl + 1) % 2].reshape(sh)

    mpo1 = qtn.MatrixProductOperator(arrays1)
    mpo2 = qtn.MatrixProductOperator(arrays2)
    return mpo1 + mpo2
