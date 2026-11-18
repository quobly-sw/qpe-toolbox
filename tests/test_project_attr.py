#!/usr/bin/env python3

import importlib.metadata
import pkgutil
import subprocess
import sys

import pytest

import qpe_toolbox


def test_version():
    """Test that version attr is accessible and matches distribution"""
    assert isinstance(qpe_toolbox.__version__, str)
    assert qpe_toolbox.__version__ == importlib.metadata.version("qpe_toolbox")


# Exhaustive on purpose: a new subpackage is a deliberate change and should be
# added here together with its own tests.
SUBPACKAGES = {"circuit", "estimation", "hamiltonian", "tensor"}


def test_package_contains_submodules():
    submodules = {m.name for m in pkgutil.iter_modules(qpe_toolbox.__path__) if m.ispkg}
    assert submodules == SUBPACKAGES


def test_dir_exposes_public_api():
    assert set(dir(qpe_toolbox)) == SUBPACKAGES | {"EXACT", "__version__"}


# Run in a fresh interpreter: pytest imports every test module at collection
# time, which binds the subpackages on the package and would make the accesses
# below plain dict lookups. Nothing may be loaded before the second loop, since
# importing one subpackage also imports the ones it depends on.
_LAZY_SCRIPT = """
import sys

import qpe_toolbox

for name in sys.argv[1:]:
    assert name not in vars(qpe_toolbox), f"{name} eagerly bound"
    assert f"qpe_toolbox.{name}" not in sys.modules, f"{name} eagerly imported"

for name in sys.argv[1:]:
    assert getattr(qpe_toolbox, name).__name__ == f"qpe_toolbox.{name}"
    assert f"qpe_toolbox.{name}" in sys.modules, f"{name} not loaded on access"
"""


def test_subpackages_loaded_lazily():
    result = subprocess.run(
        [sys.executable, "-c", _LAZY_SCRIPT, *sorted(SUBPACKAGES)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_unknown_attribute_raises():
    # Must stay indistinguishable from what a plain module raises, payload
    # included: `name` is what the interpreter uses to suggest a close match.
    with pytest.raises(
        AttributeError, match="has no attribute 'does_not_exist'"
    ) as err:
        _ = qpe_toolbox.does_not_exist
    assert err.value.name == "does_not_exist"


if __name__ == "__main__":
    test_version()
    test_package_contains_submodules()
    test_dir_exposes_public_api()
    test_subpackages_loaded_lazily()
    test_unknown_attribute_raises()
