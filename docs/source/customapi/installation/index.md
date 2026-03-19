# Installation Guide

```{warning}
On Windows, QPE Toolbox is supported only through the Windows Subsystem for Linux (WSL) due to dependency constraints.
```

## Requirements
QPE Toolbox is a pure python library and will run on any system as long as its dependencies support it. In practice, this means any 64-bit macOS or Linux distribution should work. Due to [PySCF restrictions](https://pyscf.org/user/install.html), Windows is only supported through the Windows Subsystem for Linux (WSL).

## Installation

**Installing with `pip`**

```bash
pip install qpe-toolbox
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

# install the package and its dependencies
pip install -e .[dev]
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
