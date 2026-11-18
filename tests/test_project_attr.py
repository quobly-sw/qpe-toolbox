#!/usr/bin/env python3

import importlib.metadata
import pkgutil

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
    submodules = {m.name for m in pkgutil.iter_modules(qpe_toolbox.__path__)}
    assert submodules == SUBPACKAGES


def test_dir_exposes_public_api():
    assert set(dir(qpe_toolbox)) == SUBPACKAGES | {"EXACT", "__version__"}


def test_subpackages_lazily_accessible():
    for name in sorted(SUBPACKAGES):
        subpackage = getattr(qpe_toolbox, name)
        assert subpackage.__name__ == f"qpe_toolbox.{name}"
        assert name in dir(qpe_toolbox)


def test_unknown_attribute_raises():
    missing = "does_not_exist"
    with pytest.raises(AttributeError):
        getattr(qpe_toolbox, missing)


if __name__ == "__main__":
    test_version()
    test_package_contains_submodules()
    test_dir_exposes_public_api()
    test_subpackages_lazily_accessible()
    test_unknown_attribute_raises()
