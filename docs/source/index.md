# Welcome to QPE Toolbox's documentation!

[![Doc](https://img.shields.io/badge/Doc-dev-green.svg)](https://quobly-sw.github.io/qpe-toolbox)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/quobly-sw/qpe-toolbox/actions/workflows/pytest_action.yaml/badge.svg)](https://github.com/quobly-sw/qpe-toolbox/actions/workflows/pytest_action.yaml)
[![PyPI](https://img.shields.io/pypi/v/qpe-toolbox?color=teal)](https://pypi.org/project/qpe-toolbox)

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

```{figure} _static/qpe-toolbox_pipeline.png
:name: fig:toolbox_global_view
:width: 100%
:align: center
The `qpe-toolbox` pipeline.
```

## Package Modules
::::{grid} 4
:::{grid-item-card} {doc}`circuit <autoapi/qpe_toolbox/circuit/index>`
Creation and manipulation of `quimb` circuits.
:::

:::{grid-item-card} {doc}`hamiltonian <autoapi/qpe_toolbox/hamiltonian/index>`
class for defining Hamiltonians and interface with `pyscf` for chemistry.
:::

:::{grid-item-card} {doc}`estimation <autoapi/qpe_toolbox/estimation/index>`
perform different flavors of Quantum Phase Estimation.

:::

:::{grid-item-card} {doc}`tensor <autoapi/qpe_toolbox/tensor/index>`
manipulation of Matrix Product Operators (MPO) and Matrix Product States (MPS).

:::
::::

## Contents

```{toctree}
:maxdepth: 1
:caption: Guides

Installation Guide <customapi/installation/index>

Basic Workflow <customapi/workflow/index>

Tutorials <customapi/tutorials/index>
```

```{toctree}
:maxdepth: 1
:caption: Development
Changelog <changelog>

GitHub Repository <https://github.com/quobly-sw/qpe-toolbox>

API Reference <autoapi/index>
```
