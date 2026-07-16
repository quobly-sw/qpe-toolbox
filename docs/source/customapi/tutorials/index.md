# Tutorials

Here is a list of notebooks that introduce the basics of the package. They are also available in the `examples` directory as plain `.py` files using the `py:percent` format. See our [installation guide](../installation/index.md) on how to run them on your own.

```{toctree}
:numbered:
:maxdepth: 1

building_circuits.ipynb

chemistry_to_qubit.ipynb

textbook_qpe.ipynb

trotter_decomposition.ipynb

qpe_with_trotter.ipynb

qpe_with_lcu.ipynb

robust_phase_estimation.ipynb

performance_mps.ipynb

hyperoptimization.ipynb

mpo_to_circuit.ipynb

circuit_preparation_global_opt.py

circuit_preparation_local_opt.py

circuit_preparation_approximate_local_opt.py
```


## Details

* {doc}`building_circuits <building_circuits>` explains how to create, plot, record and load quantum circuits in `quimb` and `qiskit`.

* {doc}`chemistry_to_qubit <chemistry_to_qubit>` describes how to build the qubit Hamiltonian and perform the Density Matrix Renormalization Group (DMRG) algorithm for a given molecule.

* {doc}`textbook_qpe <textbook_qpe>` introduces the textbook Quantum Phase Estimation algorithm assuming time evolution is implemented exactly.

* {doc}`trotter_decomposition <trotter_decomposition>` introduces the Trotter-Suzuki decomposition to implement a time evolution operator $U$.

* {doc}`qpe_with_trotter <qpe_with_trotter>` studies the Quantum Phase Estimation algorithm using Trotterization of the evolution operator, and provides resource estimates.

* {doc}`qpe_with_lcu <qpe_with_lcu>` gives an introduction to Block Encoding via Linear Combination of Unitaries.

* {doc}`robust_phase_estimation <robust_phase_estimation>` introduces the Robust Phase Estimation algorithm, based on the Hadamard test circuit.

* {doc}`performance_mps <performance_mps>` compares the performance of `quimb` and `qiskit` when contracting and sampling circuits with Matrix Product States.

* {doc}`hyperoptimization <hyperoptimization>` presents advanced contraction schemes provided by `quimb`.

* {doc}`variational_circuit_preparation <variational_circuit_preparation>` finds an initial guess state with variational circuit optimization.
