"""Geometry, topology and interface conformity of a mesh.

The audits read the cells as written (they do not trust the requested
size): entity counts, measures, edge lengths, orientation and, for each
interface, the cells that cross it. A conforming mesh has none.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

import numpy as np

from .counting import count_kmv_dofs, entity_counts
from .domain import FacetedSurface
from .io import read_mesh


def crossing_cells(sides: np.ndarray) -> int:
    """Count the cells with vertices strictly on both sides of an interface.

    Parameters
    ----------
    sides : numpy.ndarray
        Side of each vertex of each cell, ``-1``, ``0`` (on the interface) or
        ``+1``, shape ``(n_cells, vertices_per_cell)``.

    Returns
    -------
    int
        Number of cells that cross the interface.
    """
    return int(np.sum((sides.min(axis=1) < 0) & (sides.max(axis=1) > 0)))


def _check_cells(points: np.ndarray, cells: np.ndarray, measure: np.ndarray) -> None:
    """Reject arrays that are not a valid simplicial mesh.

    Parameters
    ----------
    points : numpy.ndarray
        Vertex coordinates.
    cells : numpy.ndarray
        Vertex indices.
    measure : numpy.ndarray
        Unsigned area or volume of each cell.

    Raises
    ------
    ValueError
        If a coordinate is not finite, there are no cells, a cell is
        degenerate or appears twice.
    """
    if not np.isfinite(points).all() or not len(cells):
        raise ValueError("Non-finite coordinates or no cells.")
    if np.any(measure <= 1e-16):
        raise ValueError("Degenerate cell.")
    if len(np.unique(np.sort(cells, axis=1), axis=0)) != len(cells):
        raise ValueError("Duplicate cell.")


def _check_shapes(points: np.ndarray, cells: np.ndarray,
                  vertices_per_cell: int, dimension: int) -> None:
    """Check the shapes and the vertex indices of a mesh.

    Parameters
    ----------
    points : numpy.ndarray
        Vertex coordinates.
    cells : numpy.ndarray
        Vertex indices.
    vertices_per_cell : int
        Expected vertices of each cell.
    dimension : int
        Expected coordinates of each vertex.

    Raises
    ------
    ValueError
        If the shapes are wrong or an index is out of range.
    """
    if points.ndim != 2 or points.shape[1] != dimension or cells.ndim != 2 \
            or cells.shape[1] != vertices_per_cell:
        raise ValueError(f"Expected points (n, {dimension}) and cells (m, {vertices_per_cell}).")
    if cells.size and (cells.min() < 0 or cells.max() >= len(points)):
        raise ValueError("Unknown vertex.")


def audit_triangles(points: np.ndarray, triangles: np.ndarray,
                    conforming_depths: Sequence[float] = ()) -> dict:
    """Audit a triangulation in spyro's ``(z, x)`` coordinates.

    Parameters
    ----------
    points : numpy.ndarray
        Vertex coordinates ``(z, x)``, shape ``(n_vertices, 2)``.
    triangles : numpy.ndarray
        Vertex indices, shape ``(n_triangles, 3)``.
    conforming_depths : sequence of float, optional
        Depths of horizontal interfaces that no triangle should cross.

    Returns
    -------
    dict
        Entity counts (``vertices``, ``edges``, ``triangles``), area
        (``area_km2``, ``minimum_area_km2``, ``maximum_area_km2``), edge
        lengths (``edge_min_km``, ``edge_mean_km``, ``edge_max_km``,
        ``edge_std_km``), ``bounds_min`` and ``bounds_max``,
        ``signed_area_counts``, ``crossing_triangles`` (per depth, a vertex
        within 1e-9 km of the depth being on it) and
        ``interface_conforming``.

    Raises
    ------
    ValueError
        If the arrays are not a valid triangulation.
    """
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles)
    _check_shapes(points, triangles, 3, 2)
    p = points[triangles]
    oriented = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                      - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    area = np.abs(oriented)
    _check_cells(points, triangles, area)
    lengths = np.concatenate([np.linalg.norm(p[:, b] - p[:, a], axis=1)
                              for a, b in ((0, 1), (1, 2), (2, 0))])
    vertices, edges, cells = entity_counts(triangles)
    crossing = {}
    for depth in conforming_depths:
        z = points[:, 0]
        side = np.where(z > depth + 1e-9, 1, np.where(z < depth - 1e-9, -1, 0))
        crossing[str(depth)] = crossing_cells(side[triangles])
    return dict(vertices=vertices, edges=edges, triangles=cells,
                area_km2=float(area.sum()), minimum_area_km2=float(area.min()),
                maximum_area_km2=float(area.max()),
                edge_min_km=float(lengths.min()), edge_mean_km=float(lengths.mean()),
                edge_max_km=float(lengths.max()), edge_std_km=float(lengths.std()),
                bounds_min=p.reshape(-1, 2).min(0).tolist(),
                bounds_max=p.reshape(-1, 2).max(0).tolist(),
                signed_area_counts=dict(positive=int(np.sum(oriented > 0)),
                                        negative=int(np.sum(oriented < 0))),
                crossing_triangles=crossing,
                interface_conforming=all(n == 0 for n in crossing.values()))


def audit_tetrahedra(points: np.ndarray, tetrahedra: np.ndarray,
                     seafloor: FacetedSurface | None = None) -> dict:
    """Audit a tetrahedralization in ``(x, y, z)`` coordinates.

    The keys are those of the 3D Salt campaigns' audit, so their records
    stay comparable.

    Parameters
    ----------
    points : numpy.ndarray
        Vertex coordinates, shape ``(n_vertices, 3)``.
    tetrahedra : numpy.ndarray
        Vertex indices, shape ``(n_tetrahedra, 4)``.
    seafloor : FacetedSurface, optional
        Interface that no tetrahedron should cross.

    Returns
    -------
    dict
        Entity counts (``vertices``, ``edges``, ``faces``, ``tetrahedra``),
        ``KMV3_dofs_predicted``, volume (``volume_km3``,
        ``minimum_volume_km3``, ``maximum_volume_km3``), ``edge_min_km`` and
        ``edge_max_km``, ``bounds_min`` and ``bounds_max``,
        ``signed_volume_counts``, ``all_cells_affine_tetrahedra``,
        ``interface_conforming`` and
        ``requested_size_is_not_edge_CPW_guarantee``; with a sea floor,
        also ``seafloor_crossing_tetrahedra``.

    Raises
    ------
    ValueError
        If the arrays are not a valid tetrahedralization.
    """
    points = np.asarray(points, dtype=float)
    tetrahedra = np.asarray(tetrahedra)
    _check_shapes(points, tetrahedra, 4, 3)
    p = points[tetrahedra]
    oriented = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                         p[:, 3] - p[:, 0]) / 6
    volume = np.abs(oriented)
    _check_cells(points, tetrahedra, volume)
    lengths = np.linalg.norm(np.concatenate([p[:, j] - p[:, i]
                                             for i, j in combinations(range(4), 2)]), axis=1)
    vertices, edges, faces, cells = entity_counts(tetrahedra)
    record = dict(vertices=vertices, edges=edges, faces=faces, tetrahedra=cells,
                  KMV3_dofs_predicted=count_kmv_dofs(tetrahedra, 3),
                  volume_km3=float(volume.sum()), minimum_volume_km3=float(volume.min()),
                  maximum_volume_km3=float(volume.max()),
                  edge_min_km=float(lengths.min()), edge_max_km=float(lengths.max()),
                  bounds_min=p.reshape(-1, 3).min(0).tolist(),
                  bounds_max=p.reshape(-1, 3).max(0).tolist(),
                  signed_volume_counts=dict(positive=int(np.sum(oriented > 0)),
                                            negative=int(np.sum(oriented < 0))),
                  all_cells_affine_tetrahedra=True, interface_conforming=True,
                  requested_size_is_not_edge_CPW_guarantee=True)
    if seafloor is not None:
        crossing = crossing_cells(seafloor.side(points)[tetrahedra])
        record.update(seafloor_crossing_tetrahedra=crossing,
                      interface_conforming=crossing == 0)
    return record


def audit_mesh(path: str | Path, degree: int,
               conforming_depths: Sequence[float] = (),
               seafloor: FacetedSurface | None = None) -> dict:
    """Metadata of a mesh file: hash, dimension, KMV budget and audit.

    Parameters
    ----------
    path : str or pathlib.Path
        A triangular (spyro ``(z, x)``) or tetrahedral gmsh mesh.
    degree : int
        Degree of the KMV element of the budget.
    conforming_depths : sequence of float, optional
        Horizontal interfaces of a 2D mesh.
    seafloor : FacetedSurface, optional
        Interface of a 3D mesh.

    Returns
    -------
    dict
        ``path``, ``sha256``, ``dimension``, ``kmv_degree``, ``kmv_dofs``
        (scalar space) and ``geometry``, the output of
        :func:`audit_triangles` or :func:`audit_tetrahedra`.
    """
    points, cells = read_mesh(path)
    dimension = points.shape[1]
    geometry = (audit_triangles(points, cells, conforming_depths) if dimension == 2
                else audit_tetrahedra(points, cells, seafloor))
    return {"path": str(path),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "dimension": dimension, "kmv_degree": degree,
            "kmv_dofs": count_kmv_dofs(cells, degree), "geometry": geometry}
