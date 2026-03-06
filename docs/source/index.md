Welcome to QPE Toolbox's documentation!
=====================

`qpe-toolbox` is an open-source python package, which
combines quantum chemistry and tensor networks methods to compile and simulate the quantum circuits of
the quantum phase estimation (QPE) algorithm.

We articulate the toolbox on top of the tensor-network library [`quimb`](https://github.com/jcmgray/quimb) to provide: (1) classical preprocessing strategies focused
on initializing and setting up QPE circuits, (2) an internal circuit-level simulator, and (3) a list of postprocessing
functionalities for retrieving final energies.
For the preprocessing stage, the QPE Toolbox offers fermionic encodings based on [`openFermion`](https://quantumai.google/openfermion) and [`pyscf`](https://pyscf.org/).

```{figure} _static/qpe-toolbox_pipeline.png
:name: fig:toolbox_global_view
:width: 100%
:align: center
The **`qpe-toolbox`** pipeline.
```

## Package Modules
::::{grid} 4
:::{grid-item-card} Circuit
The {doc}`circuit <autoapi/qpe_toolbox/circuit/index>` module provides a set of functions for creating and manipulating `quimb` circuits.
:::

:::{grid-item-card} Hamiltonian
The {doc}`hamiltonian <autoapi/qpe_toolbox/hamiltonian/index>` module provides the class for defining Hamiltonians, and a interface with `pyscf` for chemistry.
:::

:::{grid-item-card} Estimation
The {doc}`estimation <autoapi/qpe_toolbox/estimation/index>` module provides a set of functions for performing different flavors of Quantum Phase Estimation.

:::

:::{grid-item-card} Tensor
The {doc}`tensor <autoapi/qpe_toolbox/tensor/index>` module provides a set of functions for the manipulation of Matrix Product Operators and Matrix Product States.


:::
::::

## In-depth presentation of the QPE Toolbox

Check {download}`here <_static/2512_toolbox_deepdive.pdf>` for a presentation introducing our motivation and philosophy about the toolbox.


## Source

The code is hosted on [GitHub](https://github.com/quobly-sw/qpe-toolbox), and docs are hosted on [GitHub pages](https://quobly-sw.github.io/qpe-toolbox).

```{toctree}
:maxdepth: 1
:caption: Contents

Installation Guide <customapi/installation/index>

Basic Workflow <customapi/workflow/index>

Tutorials <customapi/tutorials/index>

GitHub Repository <https://github.com/quobly-sw/qpe-toolbox>

API Reference <autoapi/index>
```
