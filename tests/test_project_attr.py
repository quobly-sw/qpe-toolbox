#!/usr/bin/env python3

import pkgutil

import pytest

import qpe_toolbox


def test_has_version():
    assert hasattr(qpe_toolbox, "__version__")
    assert type(qpe_toolbox.__version__) is str


def test_package_contains_submodules():
    submodules = {m.name for m in pkgutil.iter_modules(qpe_toolbox.__path__)}
    assert "circuit" in submodules
    assert "estimation" in submodules
    assert "hamiltonian" in submodules
    assert "tensor" in submodules


def test_subpackages_lazily_accessible():
    for name in ("circuit", "estimation", "hamiltonian", "tensor"):
        subpackage = getattr(qpe_toolbox, name)
        assert subpackage.__name__ == f"qpe_toolbox.{name}"
        assert name in dir(qpe_toolbox)


def test_unknown_attribute_raises():
    missing = "does_not_exist"
    with pytest.raises(AttributeError):
        getattr(qpe_toolbox, missing)


if __name__ == "__main__":
    test_has_version()
    test_package_contains_submodules()
    test_subpackages_lazily_accessible()
    test_unknown_attribute_raises()
