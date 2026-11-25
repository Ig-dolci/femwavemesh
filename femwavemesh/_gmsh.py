"""Deferred import of gmsh.

gmsh is a dependency, but its Python module loads a shared library (and, on
Linux, X and OpenGL libraries) that a headless environment may lack. The
import is deferred to the functions that run gmsh, so ``import femwavemesh``
works everywhere and only meshing needs a working gmsh.
"""

from __future__ import annotations

from types import ModuleType


def load(purpose: str) -> ModuleType:
    """Import gmsh.

    Parameters
    ----------
    purpose : str
        What gmsh is needed for, completing "gmsh is required to ...".

    Returns
    -------
    types.ModuleType
        The ``gmsh`` module.

    Raises
    ------
    ImportError
        If gmsh cannot be imported.
    """
    try:
        import gmsh
    except (ImportError, OSError) as error:
        raise ImportError(f"gmsh is required to {purpose}.") from error
    return gmsh
