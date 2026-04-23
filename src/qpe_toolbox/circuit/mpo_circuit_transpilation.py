import re

import numpy as np
from scipy import linalg as spLA

from tqdm import tqdm

import cotengra as ctg
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_core import TensorNetwork, tensor_contract, tensor_split, get_tags

from pyscf import gto
from qpe_toolbox.hamiltonian import Hamiltonian, chemistry_hamiltonian
from qpe_toolbox.circuit.parametrized_circuits import * # do PR to generalize nn_layer so just import that function

list_paulis = ["I", "X", "Y", "Z"]

def exp_Pauli_string_as_MPO(ham_term, n_qubits, *, theta):
    """EXAMPLE
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
    """WHAT IT NEEDS TO EXPLAIN
    receives coupling, string and position. transforms the string to an mpo using Hamilt class
    then defines also an identity mpo on the same number of sites. multiply each one by cos theta 
    and sin theta and adds them
    
    lrud
    """
    
    string_coeff = ham_term[0] # e.g. -0.345
    pauli_string = ham_term[1] # e.g. 'ZYXXZ'
    active_qubits = ham_term[2] # e.g. [0, 2, 3, 6, 9]
    id4 = qu.identity(2).reshape(1, 1, 2, 2)
    
    pauli_string_tensors = []
    pauli_weight = 0
    for qubit in range(n_qubits):
        if qubit in active_qubits:
            pauli_string_tensors.append(
                qu.pauli(pauli_string[pauli_weight]).reshape(1, 1, 2, 2)
            )
            pauli_weight += 1
        else:
            pauli_string_tensors.append(id4)

    # Fix the legs on the edges
    pauli_string_tensors[0] = pauli_string_tensors[0].reshape(1, 2, 2)
    pauli_string_tensors[-1] = pauli_string_tensors[-1].reshape(1, 2, 2)
    string_mpo = qtn.MatrixProductOperator(arrays=pauli_string_tensors)
    id_mpo = string_mpo.identity()

    id_mpo[0] *= np.cos(theta * string_coeff)
    string_mpo[0] *= 1j * np.sin(theta * string_coeff)

    exp_pauli_string_mpo = id_mpo.add_MPO(string_mpo)
    exp_pauli_string_mpo.compress(cutoff=1e-6, max_bond=2)
    
    return exp_pauli_string_mpo


def trotter1_approx_as_MPO(ham_terms, n_qubits, *, dt, cutoff, max_bond, reverse_order=False):
    
    #list_bondims = []

    U_trotter1_mpo = exp_Pauli_string_as_MPO(ham_terms[0], n_qubits, theta=-dt)
    
    if reverse_order:
        trange_counter = tqdm(list(range(1, len(ham_terms))))
    else:
        trange_counter = tqdm(list(reversed(range(1, len(ham_terms)))))
    
    for i in trange_counter:
        new_factor_mpo = exp_Pauli_string_as_MPO(ham_terms[i], n_qubits, theta=-dt)
        U_trotter1_mpo = U_trotter1_mpo.apply(new_factor_mpo, compress=True, cutoff=cutoff, max_bond=max_bond)
        trange_counter.set_description(f'{"": <8}bond dimension: {U_trotter1_mpo.max_bond()}')
        #list_bondims.append(U_trotter1_mpo.max_bond())
        #print(ham_terms[i], ' new chi=', list_bondims[-1])
        
    return U_trotter1_mpo


def trotter2_approx_as_MPO(ham_terms, n_qubits, *, dt, cutoff, max_bond, verbosity = 0):
    
    if verbosity == 1:
        print(f'{"": <4}The 2nd order Trotter approximant consists on two subsequent 1st order approximants:', '\n')
        print(f'{"": <8}Computing the first 1st order approximant for a time step {{0.5}}dt')
        
    layer1_mpo = trotter1_approx_as_MPO(
        ham_terms,
        n_qubits,
        dt = dt / 2,
        cutoff = cutoff,
        max_bond = max_bond,
        )
    
    if verbosity == 1:
        print('\n')
        print(rf'{"": <8}Computing the second 1st order approximant for a time step {{0.5}}dt')
        
    layer2_mpo = trotter1_approx_as_MPO(
        ham_terms,
        n_qubits,
        dt = dt / 2,
        cutoff = cutoff,
        max_bond = max_bond,
        reverse_order=True
        )
    
    if verbosity == 1:
        print('\n')
        
    U_trotter2_mpo = layer1_mpo.apply(layer2_mpo, compress=True, cutoff=cutoff, max_bond=max_bond)
    
    return U_trotter2_mpo


def trotter4_approx_as_MPO(ham_terms, n_qubits, *, dt, cutoff, max_bond, verbosity = 0):

    sym_factor = 1./( 2.-2**(1./3.) )
    
    if verbosity == 1:
        print('The 4th order Trotter approximant consists on three subsequent 2nd order approximants:', '\n')
        print(rf'{"": <4}Computing the first 2nd order approximant for a time step s$_{{sym}}$dt')
        
    layer1_3_mpo = trotter2_approx_as_MPO(
        ham_terms,
        n_qubits,
        dt = dt * sym_factor,
        cutoff = cutoff,
        max_bond = max_bond,
        verbosity=verbosity,
        )
    
    if verbosity == 1:
        print(rf'{"": <4}Computing the second 2nd order approximant for a time step $(1-2s_{{sym}})*dt$')
        
    layer2_mpo = trotter2_approx_as_MPO(
        ham_terms,
        n_qubits,
        dt = dt * (1 - 2 * sym_factor),
        cutoff = cutoff,
        max_bond = max_bond,
        verbosity=verbosity,
        )
    
    if verbosity == 1:
        print(f'{"": <4}Multiplying first and second MPOs', '\n')
    
    U_trotter4_mpo = layer1_3_mpo.apply(layer2_mpo, compress=True, cutoff=cutoff, max_bond=max_bond)
    if verbosity == 1:
        print(f'{"": <4}Intermediate bond dimension:', U_trotter4_mpo.max_bond(), '\n')
        print(f'{"": <4}Multiplying intermediate and third MPOs', '\n')
    U_trotter4_mpo = U_trotter4_mpo.apply(layer2_mpo, compress=True, cutoff=cutoff, max_bond=max_bond)
    if verbosity == 1:
        print(f'{"": <4}Final bond dimension:', U_trotter4_mpo.max_bond())
    
    return U_trotter4_mpo


def trotter_approx_as_MPO(ham_terms, n_qubits, *, dt, order, cutoff, max_bond, verbosity = 0):
    
    #list_bondims = []
    if order == 1:
        U_trotter_mpo = trotter1_approx_as_MPO(
            ham_terms,
            n_qubits,
            dt=dt,
            cutoff=cutoff,
            max_bond=max_bond,
            )
            
    elif order == 2:
        U_trotter_mpo = trotter2_approx_as_MPO(
            ham_terms,
            n_qubits,
            dt=dt,
            cutoff=cutoff,
            max_bond=max_bond,
            verbosity=verbosity,
            )
        
    elif order == 4:
        U_trotter_mpo = trotter4_approx_as_MPO(
            ham_terms,
            n_qubits,
            dt=dt,
            cutoff=cutoff,
            max_bond=max_bond,
            verbosity=verbosity,
            )
    else:
        ValueError(f"order {order} not implemented")
        
    #print(*(U_trotter_mpo[i].shape for i in range(n_qubits)), sep="\n")
    return U_trotter_mpo


def generate_brickwall_circuit_modified(
    n_qubits,
    depth,
    one_qubit_gate_label,
    two_qubit_gate_label,
    *,
    start_ent=False,
    purely_ent=False, # whether or not to do 1-spin rotations
    param_scaling=1.0,
    rng=None,
):
    """
    Generate a brickwall-structured ``quimb`` :quimb-api:`Circuit`.

    This function constructs a quantum circuit composed of alternating
    single-body layers and nearest-neighbor two-body entangling layers
    arranged in a brickwall pattern. Each circuit layer is assigned a
    distinct circuit round.

    Circuit structure (one layer, ``start_ent=False``)::

        q0 ──[]───●───────
                  │
        q1 ──[]───●───●───
                      │
        q2 ──[]───●───●───
                  |
        q3 ──[]───●───●───
                      |
        q4 ──[]───●───●───
                  |
        q5 ──[]───●───────
          <─────────────>
            first layer

    where::

        []     = single-body gate
        ●─●    = nearest-neighbor entangling gate

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.

    depth : int
        Number of circuit layers (identified here with the gate rounds).

    one_qubit_gate_label : str
        Label identifying the single-body gate.

    two_qubit_gate_label : str
        Label identifying the two-body entangling gate.

    start_ent : bool, optional
        If ``True``, each layer starts with the brickwall entangling layer.
        Otherwise (default ``False``), the single-body layer is applied first.

    param_scaling : float, default ``1.0``
        Scaling factor for randomly initialized parameters.

    rng : :numpy-random:`numpy.random.Generator <generator>`, optional
        Random number generator used to generate gate parameters.
        If ``None``, a default generator is created.

    Returns
    -------
    circ : :quimb-api:`Circuit`
        The generated brickwall quantum circuit.

    Raises
    ------
    ValueError
        If ``one_qubit_gate_label`` does not correspond to a valid single-body gate,
        or if ``two_qubit_gate_label`` is not a valid two-body gate.

    Notes
    -----
    - Separate random number generators are used for single-body and
      two-body gate parameters to ensure reproducibility and decoupled
      randomness.
    - The same gate parameters are reused across all gates within a
      given layer.
    """
    if one_qubit_gate_label.upper() not in qtn.circuit.ONE_QUBIT_GATES:
        raise ValueError(f"Expected a single-body gate: {one_qubit_gate_label}")
    if two_qubit_gate_label.upper() not in qtn.circuit.TWO_QUBIT_GATES:
        raise ValueError(f"Expected a two-body gate: {two_qubit_gate_label}")
    if rng is None:
        rng = np.random.default_rng()

    circ = qtn.Circuit(n_qubits)
    for k in range(depth):
        if start_ent:
            for start in range(2):
                two_qubit_nn_layer(
                    circ,
                    start,
                    two_qubit_gate_label,
                    param_scaling=param_scaling,
                    gate_round=k,
                    rng=rng,
                )
            if purely_ent == False:
                one_qubit_layer(circ, one_qubit_gate_label, gate_round=k)
        else:
            if purely_ent == False:
                one_qubit_layer(circ, one_qubit_gate_label, gate_round=k)
            for start in range(2):
                two_qubit_nn_layer(
                    circ,
                    start,
                    two_qubit_gate_label,
                    param_scaling=param_scaling,
                    gate_round=k,
                    rng=rng,
                )

    return circ


def init_cost_tn(unitary_mpo, depth, tol = 1e-1, factorize = False, closed = False):
    """
     by defect, "SU4" are fed unfactorized in the circuits,
     while others like "RZZ" are always split by defect.
     Since we choose "SU4" as a starting point,
     we give the option of splitting
    """
    
    n_qubits = unitary_mpo.num_tensors
    rng = np.random.default_rng()
    TN = TensorNetwork()
    
    #-----------------------------------------
    # Define the Ansatz as a brickwall circuit
    
    bw_circ = generate_brickwall_circuit_modified(
        n_qubits=n_qubits,
        depth=depth,
        one_qubit_gate_label="U1", # it is irrelevant (purely_ent cancels its effect)
        two_qubit_gate_label="SU4",
        start_ent=True,
        purely_ent=True, # whether or not to do 1-spin rotations
        param_scaling=tol, # initialize close to identity
        rng=rng,
    )

    bw_unitary_tn = bw_circ.get_uni()
    
    if factorize:
        n_gates = int(re.search(r"\d+", list(bw_unitary_tn.tensors[-1].tags)[0]).group())
        
        for n in range(1, n_gates+1):
            gate_tens = bw_unitary_tn.select(tags=(f"GATE_{n}"), which="any").tensors
            #bw_unitary_tn.contract_between(tags1=gate_tens[0].tags, tags2=gate_tens[1].tags) # contract (if RZZ)
            # # do the same but splitting!

    TN = TN & bw_unitary_tn
    
    #-----------------------------------
    # Add the MPO unitary in the network 
    
    new_ket_ind = "B" # by defect leave the (B)ra open
    if closed:
        new_ket_ind = "k" # execute the trace by contracting the (k)et
        
    for x, tensor in enumerate(unitary_mpo):
        
        reind_tens = tensor.copy(deep=True)
        old_inds = reind_tens.inds

        if (x == 0 or x == n_qubits - 1):
            inds = (list(old_inds)[0], new_ket_ind + str(x), f"b{x}")
        else:
            inds = (list(old_inds)[0], list(old_inds)[1], new_ket_ind + str(x), f"b{x}")
            
        reind_tens.modify(inds = inds, tags = (f"I{x}", f"Uref{x}", "MPO"))
        
        TN = TN & reind_tens
        
    return TN


def get_environments(n_qubits, x, cost_tn, draw_env = False):
    
    env_tn = cost_tn.select(tags=[f"I{x}"], which="!any")
    
    left_tags = [f"I{x}" for x in range(x)]
    right_tags = [f"I{x}" for x in range(x+1, n_qubits)]
        
    if draw_env:
        
        copy_tn = cost_tn.copy(deep=True)
        copy_tn.select(tags=[f"I{x}"], which="!any").select(tags=left_tags, which="any")#.add_tag(f"L{x}")
        copy_tn.select(tags=[f"I{x}"], which="!any").select(tags=right_tags, which="any")#.add_tag(f"R{x}")
        copy_tn.draw((f"L{x}", f"R{x}", f"Uref{x}"), layout="kamada_kawai", show_inds=False, show_tags=False)
    
    list_env_tn = []
    if x > 0:
        list_env_tn.append(env_tn.select(tags=left_tags, which="any"))
    if x < n_qubits - 1:
        list_env_tn.append(env_tn.select(tags=right_tags, which="any"))

    return list_env_tn


def find_transfer_structure(n_qubits, cost_tn):
    
    dict_L_env = {}
    dict_R_env = {}
    for x in range(n_qubits):
        list_env_tn = get_environments(n_qubits, x, cost_tn)
        if x == 0:
            dict_R_env[f"I{x}"] = list_env_tn[0]
        elif x == n_qubits-1:
            dict_L_env[f"I{x}"] = list_env_tn[0]
        else:
            dict_L_env[f"I{x}"] = list_env_tn[0]
            dict_R_env[f"I{x}"] = list_env_tn[1]
    
    dict_L_transf = {}
    dict_R_transf = {}
    for x in range(1, n_qubits - 1):
        
        # left transfer
        
        transf_tn = dict_L_env[f"I{x + 1}"].select(tags=(f"I{x}"), which="any")
        transf_tags = get_tags(transf_tn)
        filtered_tags = []
        for tag in transf_tags:
            if tag[0] == "G" or tag[0] == "U":
                filtered_tags.append(tag)
        dict_L_transf[f"L{x}_to_L{x + 1}"] = filtered_tags
        
        # right transfer
        
        transf_tn = dict_R_env[f"I{n_qubits - 2 - x}"].select(tags=(f"I{n_qubits - 1 - x}"), which="any")
        transf_tags = get_tags(transf_tn)
        filtered_tags = []
        for tag in transf_tags:
            if tag[0] == "G" or tag[0] == "U":
                filtered_tags.append(tag)
        dict_R_transf[f"R{n_qubits - 1 - x}_to_R{n_qubits - 2 - x}"] = filtered_tags
        
    return dict_L_transf, dict_R_transf


def build_first_sweep(n_qubits, cost_tn, dict_L_transf, dict_R_transf, drop_tags=True):
    """
    Generate the first set of left and right environments (recycling the former)
    and save them in a list of tensor networks
    """

    dict_envs = {}
    
    # the first environments are L{1} and R{n_qubits-1}, which are the edge tensors on the MPO
    L_init = cost_tn.select(tags="Uref0", which="all").tensors[0]
    L_init.add_tag(tag=("L1"))
    R_init = cost_tn.select(tags=f"Uref{n_qubits-1}", which="all").tensors[0]
    R_init.add_tag(tag=(f"R{n_qubits-2}"))
    
    dict_envs[f"L1"] = L_init
    dict_envs[f"R{n_qubits-2}"] = R_init
    L_next = L_init.copy(deep=True)
    R_next = R_init.copy(deep=True)
    
    for counter, transf_tags in enumerate(zip(dict_L_transf.values(), dict_R_transf.values())):
        
        #print(counter, f"L{counter + 2}", f"R{n_qubits - 3 - counter}")
        L_next = L_next.copy(deep=True)
        L_transf_tn = cost_tn.select(tags=transf_tags[0], which="any")
        L_next = L_next & L_transf_tn
        L_next = tensor_contract(*L_next.tensors, drop_tags=drop_tags)
        L_next.add_tag(tag=(f"L{counter + 2}"))
        
        dict_envs[f"L{counter + 2}"] = L_next
        
        R_next = R_next.copy(deep=True)
        R_transf_tn = cost_tn.select(tags=transf_tags[1], which="any")
        R_next = R_next & R_transf_tn
        R_next = tensor_contract(*R_next.tensors, drop_tags=drop_tags)
        R_next.add_tag(tag=(f"R{n_qubits - 3 - counter}"))
        
        dict_envs[f"R{n_qubits - 3 - counter}"] = R_next
        
    return dict_envs


def build_local_cost_tn(n_qubits, x, dict_envs, cost_tn):
    
    gates_to_opt = cost_tn.select(tags=f"I{x}", which="any")
    tags_to_opt = get_tags(gates_to_opt)
    
    filtered_tags = []
    for tag in tags_to_opt: 
        if tag[0] == "G": # exclude "Uref"
            filtered_tags.append(tag)
    
    if x == 0:
        local_cost_tn = gates_to_opt & dict_envs[f"R{x}"]
    elif x == n_qubits - 1:
        local_cost_tn = dict_envs[f"L{x}"] & gates_to_opt
    else:
        local_cost_tn = dict_envs[f"L{x}"] & gates_to_opt & dict_envs[f"R{x}"]

    return local_cost_tn, filtered_tags


def PRC_local_cost_tn(loc_cost_tn, tags, hyperopt):
    """
    Pop-Rehearse-Contract
    
    this will be called from sweep, and will only rehearse on the first sweep;
    can pop more than one gate!
    """

    p_loc_cost_tn = loc_cost_tn.copy(deep=True)
    p_loc_cost_tn.delete(tags=tags)
    prc_contr_loc_cost_tn = p_loc_cost_tn.contract(optimize=hyperopt)
    
    return prc_contr_loc_cost_tn, hyperopt


