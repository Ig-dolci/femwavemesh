"""The former name, kmvmesh, as an alias of femwavemesh."""

import importlib
import sys

import pytest

import femwavemesh


def test_kmvmesh_warns_and_gives_the_same_names() -> None:
    """``import kmvmesh`` warns and exposes femwavemesh's public names."""
    sys.modules.pop("kmvmesh", None)
    with pytest.warns(DeprecationWarning, match="femwavemesh"):
        old = importlib.import_module("kmvmesh")
    assert old.__all__ == femwavemesh.__all__
    assert old.__version__ == femwavemesh.__version__
    for name in femwavemesh.__all__:
        assert getattr(old, name) is getattr(femwavemesh, name)
