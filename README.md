![qpe-toolbox logo](https://github.com/quobly-sw/qpe-toolbox/raw/main/docs/source/_static/qpe-toolbox_logo.png)

[![Doc](https://img.shields.io/badge/Doc-dev-green.svg)](https://quobly-sw.github.io/qpe-toolbox)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/quobly-sw/qpe-toolbox/actions/workflows/pytest_action.yaml/badge.svg)](https://github.com/quobly-sw/qpe-toolbox/actions/workflows/pytest_action.yaml)

[`qpe-toolbox`](https://github.com/quobly-sw/qpe-toolbox) is an open-source python package, which
combines quantum chemistry and tensor networks methods to compile and simulate the quantum circuits of
the Quantum Phase Estimation (QPE) algorithm. The code is hosted on [github](https://github.com/quobly-sw/qpe-toolbox),
and docs are hosted on [GitHub Pages](https://quobly-sw.github.io/qpe-toolbox).

Check [here](https://quobly-sw.github.io/qpe-toolbox/_static/2512_toolbox_deepdive.pdf) for a presentation introducing our motivation and philosophy about the toolbox.

We articulate the toolbox on top of the tensor-network library [`quimb`](https://github.com/jcmgray/quimb) to provide: (1) classical preprocessing strategies focused
on initializing and setting up QPE circuits, (2) an internal circuit-level simulator, and (3) a list of postprocessing
functionalities for retrieving final energies.
For the preprocessing stage, the QPE Toolbox offers fermionic encodings based on [OpenFermion](https://quantumai.google/openfermion) and [PySCF](https://pyscf.org/).

# Installation
## Requirements
Our packaged is built above [`numpy`](https://github.com/numpy/numpy), [`openfermion`](https://github.com/quantumlib/OpenFermion), [`pyscf`](https://github.com/pyscf/pyscf), [`quimb`](https://github.com/jcmgray/quimb), [`scipy`](https://github.com/scipy/scipy) as core dependencies. [`jupyterlab`](https://github.com/jupyterlab/jupyterlab) and [`jupytext`](https://github.com/mwouts/jupytext) are needed to run the examples as notebooks. The complete list of dependencies is in [pyproject.toml](./pyproject.toml).

## Installation from sources
Installing from sources gives access to our [examples](#Examples) which contains detailed explanations on the Quantum Phase Estimation algorithm.
### with pip
```bash
# clone the project
git clone git@github.com:quobly-sw/qpe-toolbox.git && cd qpe-toolbox

# Create a virtual environment
python3 -m venv .venv --prompt qpe-toolbox

# activate it
source .venv/bin/activate

# install the package and its dependencies
pip install -e . --group=dev
```

### with uv
```bash
# clone the project
git clone git@github.com:quobly-sw/qpe-toolbox.git && cd qpe-toolbox

# Create a virtual environment and synchronize it with lock
uv sync --locked

# activate it
source .venv/bin/activate
```



# Contents

The package is divided in four modules:
- `circuit`: creation and manipulation of `quimb` circuits.
- `hamiltonian`: class for defining Hamiltonians and interface with `pyscf` for chemistry.
- `estimation`: perform different flavors of Quantum Phase Estimation.
- `tensor`: manipulation of Matrix Product Operators (MPO) and Matrix Product States (MPS).


# Examples
We provide a list of notebooks that introduce the basics of the package and contain detailed explanations on the Quantum Phase Estimation algorithm. They are available in the `examples` directory as plain `.py` files using the `py:percent` format. We use [Jupytext](https://jupytext.readthedocs.io/en/latest/) to convert them and pair them with a twin `.ipynb` notebook.
To convert a given example and execute it as a notebook, open `jupyterlab`, right-click on the `.py` file and select "Open with > Notebook" or "Jupytext Notebook": this allows you to save the notebook's outputs in your local repository. We also include [the executed notebooks in our documentation](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/index.html).

1. [`building_circuits`](examples/building_circuits.py) explains how to create, plot, record and load quantum circuits in `quimb` and `qiskit`.

2. [`chemistry_to_qubit`](examples/chemistry_to_qubit.py) describes how to build the qubit Hamiltonian and perform the Density Matrix Renormalization Group (DMRG) algorithm for a given molecule.

3. [`textbook_qpe`](examples/textbook_qpe.py) introduces the textbook Quantum Phase Estimation algorithm assuming time evolution is implemented exactly.

4. [`trotter_decomposition`](examples/trotter_decomposition.py) introduces the Trotter-Suzuki decomposition to implement a time evolution operator $U$.

5. [`qpe_with_trotter`](examples/qpe_with_trotter.py) studies the Quantum Phase Estimation algorithm using Trotterization of the evolution operator, and provides resource estimates.

6. [`qpe_with_lcu`](examples/qpe_with_lcu.py) gives an introduction to Block Encoding via Linear Combination of Unitaries.

7. [`robust_phase_estimation`](examples/robust_phase_estimation.py) introduces the Robust Phase Estimation algorithm, based on the Hadamard test circuit.

8. [`performance_mps`](examples/performance_mps.py) compares the performance of `quimb` and `qiskit` when contracting and sampling circuits with Matrix Product States.

9. [`hyperoptimization`](examples/hyperoptimization.py) presents advanced contraction schemes provided by `quimb`.

10. [`variational_circuit_preparation`](examples/variational_circuit_preparation.py) finds an initial guess state with variational circuit optimization.

# Basic workflow

To perform Quantum Phase Estimation with the toolbox, take the following steps:

1. The `Hamiltonian` class describes the qubit Hamiltonian. Choose a system:

   1.1. Spin model with e.g. `heisenberg_hamiltonian` or a custom `Hamiltonian` instance.

   1.2. Molecule with `chemistry_hamiltonian`; see the [`chemistry_to_qubit`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/chemistry_to_qubit.html) example.

2. Prepare an initial state as a Matrix Product State. Two methods are available:

   2.1. Density Matrix Renormalization Group (DMRG) - see the [`chemistry_to_qubit`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/chemistry_to_qubit.html) tutorial.

   2.2. Parametrized circuit optimization - see the tutorial on [`variational_circuit_preparation`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/variational_circuit_preparation.html).

3. Encode the Hamiltonian into a unitary via either:

   3.1. Exact time evolution or Trotterization, available as methods of the `Hamiltonian` class - see the
    tutorial on [`trotter_decomposition`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/trotter_decomposition.html).

   3.2. Block encoding functions from the `estimation` module - see the tutorial on Linear Combination of Unitaries: [`qpe_with_lcu`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/qpe_with_lcu.html).

4. Initialize a circuit with a physical register and a phase register. From the `circuit` module, chose between:

   4.1. `make_circ` to create a Tensor Network representation of the circuit.

   4.2. `make_circMPS` to store the state as an MPS and iteratively apply the gates.

   See the tutorials on [`building_circuits`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/building_circuits.html), [`performance_mps`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/performance_mps.html)
    and [`hyperoptimization`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/hyperoptimization.html) for an introduction on circuit simulation with `quimb`.

5. Run QPE: in the `estimation` module, choose between

   5.1. Textbook QPE: see the corresponding tutorial [`textbook_qpe`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/textbook_qpe.html).

   5.2. Robust Phase Estimation, a version of QPE with a single ancilla and circuit repetitions - see the [`robust_phase_estimation`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/robust_phase_estimation.html) tutorial.

# License

Apache 2.0

# Credits

2026 Foxconn, Quobly

Authors:
- Thibaud Louvet (thibaud.louvet@quobly.io)
- Calvin Ku (calvin.ku@foxconn.com)
- Yu-Cheng Chen (kesson.yc.chen@foxconn.com)
- Carlos Ramos Marimón (carlos.marimon@quobly.io)
- Olivier Gauthé (olivier.gauthe@quobly.io)
- Tristan Meunier (tristan.meunier@quobly.io)
- Min-Hsiu Hsieh (min-hsiu.hsieh@foxconn.com)
- Benoit Vermersch (benoit.vermersch@quobly.io)
