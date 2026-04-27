# Installation Guide

```{warning}
On Windows, QPE Toolbox is supported only through the Windows Subsystem for Linux (WSL) due to dependency constraints.
```

## Requirements
QPE Toolbox is a pure python library and will run on any system as long as its dependencies support it. In practice, this means any 64-bit macOS or Linux distribution should work. Due to [PySCF restrictions](https://pyscf.org/user/install.html), Windows is only supported through the Windows Subsystem for Linux (WSL).

## Installation

**With `pip`**

```bash
pip install "qpe-toolbox[recommended]"
```

**With `uv`**

```bash
uv add "qpe-toolbox[recommended]"
```

**With `conda`**

```bash
conda install -c conda-forge qpe-toolbox
pip install cotengrust kahypar  # add recommended optional dependencies (pypi-only)
```

**With `pixi`**

```bash
pixi add qpe-toolbox
pixi add cotengrust kahypar  # add recommended optional dependencies
```

## Installation from sources
Installing from sources gives access to our [tutorials](https://quobly-sw.github.io/qpe-toolbox/customapi/tutorials/index.html) which contains detailed explanations on the Quantum Phase Estimation algorithm.

### with uv
This is the recommended installation method. We provide a `uv.lock` file and use it to run our examples as part of our test suite, therefore this setup is expected to work on any supported platform.
```bash
# clone the project
git clone git@github.com:quobly-sw/qpe-toolbox.git && cd qpe-toolbox

# launch jupyter to run examples
uv run --locked jupyter lab --notebook-dir=examples/
```

### with pixi
Installing with `pixi` allows to use the MKL as a backend for BLAS/LAPACK, which may provides a speed-up compared to NumPy openblas default.
```bash
# clone the project
git clone git@github.com:quobly-sw/qpe-toolbox.git && cd qpe-toolbox

# launch jupyter to run examples
pixi run --environment=dev jupyter lab --notebook-dir=examples/
```

### with pip
Installing development dependencies with `pip` requires pip ≥ 25.1 (for [PEP 735](https://peps.python.org/pep-0735/) dependency group support). We recommend using `uv` or `pixi` for a simpler setup.

```bash
# clone the project
git clone git@github.com:quobly-sw/qpe-toolbox.git && cd qpe-toolbox

# Create a virtual environment
python3 -m venv .venv --prompt qpe-toolbox

# activate it
source .venv/bin/activate

# update pip to support dependency-groups
pip install --upgrade pip

# install the package and its dependencies
pip install -e ".[recommended]" --group dev

# launch jupyter to run examples
jupyter lab --notebook-dir=examples/
```
