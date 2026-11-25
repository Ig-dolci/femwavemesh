"""Reading gmsh meshes into vertex and cell arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _gmsh

#: gmsh element types of the cells.
TRIANGLE, TETRAHEDRON = 2, 4


def zero_based(tags: np.ndarray, coordinates: np.ndarray, node_tags: np.ndarray,
               vertices_per_cell: int, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn gmsh's node tags into zero-based vertex indices.

    Parameters
    ----------
    tags : numpy.ndarray
        Node tags, as ``gmsh.model.mesh.getNodes`` returns them.
    coordinates : numpy.ndarray
        Flat node coordinates, three per node.
    node_tags : numpy.ndarray
        Flat node tags of the cells.
    vertices_per_cell : int
        3 for triangles, 4 for tetrahedra.
    dimension : int
        Number of coordinates kept per vertex.

    Returns
    -------
    points : numpy.ndarray
        Vertex coordinates, shape ``(n_vertices, dimension)``.
    cells : numpy.ndarray
        Zero-based vertex indices, shape ``(n_cells, vertices_per_cell)``.
    """
    index = np.zeros(int(tags.max()) + 1, dtype=np.int64)
    index[tags.astype(np.int64)] = np.arange(len(tags))
    points = coordinates.reshape(-1, 3)[:, :dimension]
    cells = index[node_tags.astype(np.int64)].reshape(-1, vertices_per_cell)
    return points, cells


def _read(path: str | Path, dimension: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Read the vertices and the simplices of one dimension.

    Parameters
    ----------
    path : str or pathlib.Path
        A gmsh mesh file.
    dimension : int or None
        2 for the triangles, 3 for the tetrahedra; ``None`` takes the
        dimension of the mesh.

    Returns
    -------
    points : numpy.ndarray
        Vertex coordinates, shape ``(n_vertices, dimension)``.
    cells : numpy.ndarray
        Zero-based vertex indices, shape ``(n_cells, dimension + 1)``.
    """
    gmsh = _gmsh.load("read meshes")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        if dimension is None:
            dimension = gmsh.model.getDimension()
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        _, node_tags = gmsh.model.mesh.getElementsByType(
            TRIANGLE if dimension == 2 else TETRAHEDRON)
    finally:
        gmsh.finalize()
    return zero_based(tags, coordinates, node_tags, dimension + 1, dimension)


def read_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the vertices and cells of a triangular or tetrahedral mesh.

    Parameters
    ----------
    path : str or pathlib.Path
        A gmsh mesh file.

    Returns
    -------
    points : numpy.ndarray
        Vertex coordinates, shape ``(n_vertices, d)``, ``d`` the dimension
        of the mesh (``(z, x)`` for spyro's 2D meshes, ``(x, y, z)`` in 3D).
    cells : numpy.ndarray
        Zero-based vertex indices of the triangles or tetrahedra, shape
        ``(n_cells, d + 1)``.

    Raises
    ------
    ImportError
        If gmsh is not installed.
    """
    return _read(path, None)


def read_triangulation(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the vertices and triangles of a mesh file.

    Parameters
    ----------
    path : str or pathlib.Path
        A gmsh mesh file.

    Returns
    -------
    points : numpy.ndarray
        Vertex coordinates ``(z, x)``, shape ``(n_vertices, 2)``.
    triangles : numpy.ndarray
        Zero-based vertex indices, shape ``(n_triangles, 3)``.

    Raises
    ------
    ImportError
        If gmsh is not installed.
    """
    return _read(path, 2)


def read_tetrahedra(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the vertices and tetrahedra of a mesh file.

    Parameters
    ----------
    path : str or pathlib.Path
        A gmsh mesh file.

    Returns
    -------
    points : numpy.ndarray
        Vertex coordinates ``(x, y, z)``, shape ``(n_vertices, 3)``.
    tetrahedra : numpy.ndarray
        Zero-based vertex indices, shape ``(n_tetrahedra, 4)``.

    Raises
    ------
    ImportError
        If gmsh is not installed.
    """
    return _read(path, 3)
