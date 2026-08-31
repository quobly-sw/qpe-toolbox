# Basic workflow

To perform Quantum Phase Estimation with the toolbox, take the following steps:

1. The `Hamiltonian` class describes the qubit Hamiltonian. Chose a system:

   1.1. Spin model with e.g. `heisenberg_hamiltonian` or a custom `Hamiltonian` instance.

   1.2. Molecule with `chemistry_hamiltonian`; see the tutorial on {doc}`Chemistry to qubit Hamiltonians <../tutorials/chemistry_to_qubit>`.

2. Prepare an initial state as a Matrix Product State. Two methods are available:

   2.1. Density Matrix Renormalization Group (DMRG) - see the tutorial on {doc}`Chemistry to qubit Hamiltonians <../tutorials/chemistry_to_qubit>`.

   2.2. Parametrized circuit optimization - see the tutorial on {doc}`Variational circuit preparation I <../tutorials/circuit_preparation_opt>` and {doc}`Variational circuit preparation II <../tutorials/circuit_preparation_approximate_local_opt>`.

3. Encode $\hat{H}$ into a unitary via either:

   3.1. Exact time evolution or Trotterization, available as methods of the `Hamiltonian` class - see the
    tutorial on {doc}`Trotterization <../tutorials/trotter_decomposition>`.

   3.2. Block encoding functions from the `lcu_walk_operator` module - see the tutorial on {doc}`Linear Combination of Unitaries <../tutorials/qpe_with_lcu>`.

4. Initialize circuit. From `circuit.initialization`, chose between:

   4.1. `make_circ` to create a Tensor Network representation of the circuit.

   4.2. `make_circMPS` to store the state as an MPS and iteratively apply the gates.

   See the tutorials on {doc}`Building circuits <../tutorials/building_circuits>` and {doc}`MPS performance <../tutorials/performance_mps>`
    for an introduction on circuit simulation with `quimb`.

5. Run QPE: in `estimation` module, chose between

   5.1. Textbook QPE: `phase_estimation` - see {doc}`this tutorial <../tutorials/textbook_qpe>`.

   5.2. Robust Phase Estimation, a version of QPE with a single ancilla and circuit repetitions: `robust_phase_estimation` - see
    the {doc}`corresponding tutorial <../tutorials/robust_phase_estimation>`.
