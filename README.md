![qpe-toolbox logo](https://github.com/quobly-sw/qpe-toolbox/raw/main/docs/source/_static/qpe-toolbox_logo.png)

[![Doc](https://img.shields.io/badge/Doc-dev-green.svg)](https://quobly-sw.github.io/qpe-toolbox)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/quobly-sw/qpe-toolbox/actions/workflows/pytest_action.yaml/badge.svg)](https://github.com/quobly-sw/qpe-toolbox/actions/workflows/pytest_action.yaml)
[![PyPI](https://img.shields.io/pypi/v/qpe-toolbox?color=teal)](https://pypi.org/project/qpe-toolbox)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/qpe-toolbox)](https://anaconda.org/conda-forge/qpe-toolbox)

[`qpe-toolbox`](https://github.com/quobly-sw/qpe-toolbox) is an open-source
Python package for compiling and simulating Quantum Phase Estimation (QPE)
circuits, combining quantum chemistry with tensor-network methods.
The code is hosted on [github](https://github.com/quobly-sw/qpe-toolbox),
and docs are available on [GitHub Pages](https://quobly-sw.github.io/qpe-toolbox).

See our [overview presentation](https://quobly-sw.github.io/qpe-toolbox/_static/2512_toolbox_deepdive.pdf)
for the motivation and philosophy behind the toolbox.

Built on the tensor-network library [`quimb`](https://github.com/jcmgray/quimb), it provides:
  - **Classical preprocessing**: quantum chemistry with [PySCF](https://pyscf.org) and fermionic encodings via [OpenFermion](https://quantumai.google/openfermion)
  - **Quantum simulation**: QPE circuit construction and circuit-level simulator with tensor networks
  - **Postprocessing**: energy retrieval from phase measurement outcomes

# Installation
We provide a more detailed installation guide in our [documentation](https://quobly-sw.github.io/qpe-toolbox/customapi/installation/index.html).

## Requirements
Our package is built above [`numpy`](https://github.com/numpy/numpy)/[`scipy`](https://github.com/scipy/scipy)/[`matplotlib`](https://github.com/matplotlib/matplotlib),  [`openfermion`](https://github.com/quantumlib/OpenFermion), [`pyscf`](https://github.com/pyscf/pyscf) and [`quimb`](https://github.com/jcmgray/quimb) as core dependencies. We use [`jax`](https://github.com/jax-ml/jax) for variational circuit optimization. [`jupyterlab`](https://github.com/jupyterlab/jupyterlab) and [`jupytext`](https://github.com/mwouts/jupytext) are needed to run the examples as notebooks. The complete list of dependencies is in [pyproject.toml](https://github.com/quobly-sw/qpe-toolbox/raw/main/pyproject.toml).

`qpe-toolbox` is a pure Python library, however due to [PySCF restrictions](https://pyscf.org/user/install.html), Windows is only supported through the Windows Subsystem for Linux (WSL).

## Installation from pypi
`qpe-toolbox` is available on [PyPI](https://pypi.org/project/qpe-toolbox):
```bash
pip install qpe-toolbox
```

## Installation from conda-forge
`qpe-toolbox` is available on [conda-forge](https://anaconda.org/conda-forge/qpe-toolbox). Install it with `conda`:
```bash
conda install -c conda-forge qpe-toolbox
```
or with `pixi`:
```bash
pixi add qpe-toolbox
```

## Installation from sources
Installing from sources gives access to our [tutorials](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/index.html) which contains detailed explanations on the Quantum Phase Estimation algorithm.

### with pip
```bash
# clone the project
git clone git@github.com:quobly-sw/qpe-toolbox.git && cd qpe-toolbox

# Create a virtual environment
python3 -m venv .venv --prompt qpe-toolbox

# activate it
source .venv/bin/activate

# update pip to support dependency-groups (requires pip >= 25.1)
pip install --upgrade pip

# install the package and its dependencies
pip install -e ".[recommended]" --group dev
```

### with uv
```bash
git clone git@github.com:quobly-sw/qpe-toolbox.git && cd qpe-toolbox
uv sync --locked
source .venv/bin/activate
```

### with pixi
```bash
git clone git@github.com:quobly-sw/qpe-toolbox.git && cd qpe-toolbox
pixi shell --environment dev
```

# Contents

The package is divided in four modules:
- `circuit`: creation and manipulation of `quimb` circuits.
- `hamiltonian`: class for defining Hamiltonians and interface with `pyscf` for chemistry.
- `estimation`: perform different flavors of Quantum Phase Estimation.
- `tensor`: manipulation of Matrix Product Operators (MPO) and Matrix Product States (MPS).


# Examples
We provide a list of notebooks that introduce the basics of the package and contain detailed explanations on the Quantum Phase Estimation algorithm. Refer to [our documentation](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/index.html#details) for an up to date list. They are available in the [`examples`](https://github.com/quobly-sw/qpe-toolbox/blob/main/examples) directory as plain `.py` files using the `py:percent` format. We use [Jupytext](https://jupytext.readthedocs.io/en/latest/) to convert them and pair them with a twin `.ipynb` notebook.
To convert a given example and execute it as a notebook, open `jupyterlab`, right-click on the `.py` file and select "Open with > Notebook" or "Jupytext Notebook": this allows you to save the notebook's outputs in your local repository. We also include [the executed notebooks in our documentation](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/index.html).

# Basic workflow

To perform Quantum Phase Estimation with the toolbox, take the following steps:

1. The `Hamiltonian` class describes the qubit Hamiltonian. Choose a system:

   - Spin model with e.g. `heisenberg_hamiltonian` or a custom `Hamiltonian` instance.
   - Molecule with `chemistry_hamiltonian`; see the [`chemistry_to_qubit`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/chemistry_to_qubit.html) example.

2. Prepare an initial state as a Matrix Product State. Two methods are available:

   - Density Matrix Renormalization Group (DMRG) - see the [`chemistry_to_qubit`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/chemistry_to_qubit.html) tutorial.
   - Parametrized circuit optimization - see the tutorial on [`variational_circuit_preparation`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/variational_circuit_preparation.html).

3. Encode the Hamiltonian into a unitary via either:

   - Exact time evolution or Trotterization, available as methods of the `Hamiltonian` class - see the tutorial on [`trotter_decomposition`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/trotter_decomposition.html).
   - Block encoding functions from the `estimation` module - see the tutorial on Linear Combination of Unitaries: [`qpe_with_lcu`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/qpe_with_lcu.html).

4. Initialize a circuit with a physical register and a phase register. From the `circuit` module, choose between:

   - `make_circ` to create a Tensor Network representation of the circuit.
   - `make_circMPS` to store the state as an MPS and iteratively apply the gates.

   See the tutorials on [`building_circuits`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/building_circuits.html) and [`performance_mps`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/performance_mps.html) for an introduction on circuit simulation with `quimb`.

5. Run QPE: in the `estimation` module, choose between

   - Textbook QPE: see the corresponding tutorial [`textbook_qpe`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/textbook_qpe.html).
   - Robust Phase Estimation, a version of QPE with a single ancilla and circuit repetitions - see the [`robust_phase_estimation`](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/robust_phase_estimation.html) tutorial.

# Contributing

Contributions are welcome and highly appreciated. To get started, check out the [contributing guidelines](https://github.com/quobly-sw/qpe-toolbox/blob/main/CONTRIBUTING.md).

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
