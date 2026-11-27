r"""Geometry, topology, shape quality and interface conformity of a mesh.

The audits read the cells as written (they do not trust the requested
size): entity counts, measures, edge lengths, orientation, shape quality
and, for each interface, the cells that cross it. A conforming mesh has none.

The shape quality is the radius ratio :math:`d\,r/R` of each cell
(:func:`radius_ratios`: 1 for the equilateral triangle and the regular
tetrahedron, 0 for a flat cell) and its angles (:func:`cell_angles`: the
angles of a triangle, the dihedral angles of a tetrahedron). The smallest
inscribed radius bounds the time step of an explicit scheme.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

import numpy as np

from .counting import KMV, count_dofs, count_kmv_dofs, element_family, entity_counts
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


def _corners(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Vertex coordinates of each triangle or tetrahedron.

    Parameters
    ----------
    points : numpy.ndarray
        Vertex coordinates, shape ``(n, 2)`` or ``(n, 3)``.
    cells : numpy.ndarray
        Vertex indices, shape ``(m, 3)`` or ``(m, 4)``.

    Returns
    -------
    numpy.ndarray
        Coordinates, shape ``(m, d + 1, d)``.

    Raises
    ------
    ValueError
        If the shapes do not match or an index is out of range.
    """
    points = np.asarray(points, dtype=float)
    cells = np.asarray(cells)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("Expected points (n, 2) or (n, 3).")
    _check_shapes(points, cells, points.shape[1] + 1, points.shape[1])
    return points[cells]


def _inward_normals(p: np.ndarray) -> np.ndarray:
    """Inward normals of the faces of tetrahedra.

    Parameters
    ----------
    p : numpy.ndarray
        Vertices of the tetrahedra, shape ``(m, 4, 3)``.

    Returns
    -------
    numpy.ndarray
        Normal of the face opposite each vertex, pointing to that vertex,
        of length twice the face's area; shape ``(m, 4, 3)``. A flat
        tetrahedron gets zero normals.
    """
    normals = []
    for k in range(4):
        a, b, c = (i for i in range(4) if i != k)
        n = np.cross(p[:, b] - p[:, a], p[:, c] - p[:, a])
        side = np.sign(np.einsum("ij,ij->i", n, p[:, k] - p[:, a]))
        normals.append(n * side[:, np.newaxis])
    return np.stack(normals, axis=1)


def _radii(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Inscribed radius and radius ratio of triangles or tetrahedra.

    Parameters
    ----------
    p : numpy.ndarray
        Vertices, shape ``(m, 3, 2)`` or ``(m, 4, 3)``.

    Returns
    -------
    inradius : numpy.ndarray
        Radius :math:`r` of the inscribed circle or sphere.
    ratio : numpy.ndarray
        :math:`d\,r/R`, 0 for a flat cell.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if p.shape[1] == 3:
            lengths = np.linalg.norm(p[:, [1, 2, 0]] - p, axis=2)
            e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
            area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
            half = lengths.sum(axis=1) / 2
            # r = A / s and R = abc / (4 A), so 2 r / R = 8 A^2 / (s abc).
            inradius = area / half
            ratio = 8 * area**2 / (half * lengths.prod(axis=1))
        else:
            e1, e2, e3 = (p[:, k] - p[:, 0] for k in (1, 2, 3))
            volume = np.abs(np.einsum("ij,ij->i", e1, np.cross(e2, e3))) / 6
            surface = 0.5 * np.linalg.norm(_inward_normals(p), axis=2).sum(axis=1)
            # R = |N| / (12 V) from the circumcentre; r = 3 V / S, so 3 r / R = 108 V^2 / (S |N|).
            square = [np.einsum("ij,ij->i", e, e)[:, np.newaxis] for e in (e1, e2, e3)]
            centre = (square[0] * np.cross(e2, e3) + square[1] * np.cross(e3, e1)
                      + square[2] * np.cross(e1, e2))
            inradius = 3 * volume / surface
            ratio = 108 * volume**2 / (surface * np.linalg.norm(centre, axis=1))
    return np.nan_to_num(inradius), np.nan_to_num(ratio, posinf=0.0)


def radius_ratios(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    r"""Radius ratio of each triangle or tetrahedron.

    The ratio is :math:`d\,r/R`, with :math:`r` and :math:`R` the radii of
    the inscribed and circumscribed circles (spheres) and :math:`d` the
    dimension: 1 for the equilateral triangle and the regular tetrahedron,
    decreasing to 0 for a flat cell, slivers included.

    Parameters
    ----------
    points : numpy.ndarray
        Vertex coordinates, shape ``(n, 2)`` (spyro's ``(z, x)``) or
        ``(n, 3)``.
    cells : numpy.ndarray
        Vertex indices of the triangles ``(m, 3)`` or tetrahedra ``(m, 4)``.

    Returns
    -------
    numpy.ndarray
        One ratio per cell, in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the shapes do not match or an index is out of range.
    """
    return _radii(_corners(points, cells))[1]


def cell_angles(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Angles of each triangle, or dihedral angles of each tetrahedron.

    Parameters
    ----------
    points : numpy.ndarray
        Vertex coordinates, shape ``(n, 2)`` (spyro's ``(z, x)``) or
        ``(n, 3)``.
    cells : numpy.ndarray
        Vertex indices of the triangles ``(m, 3)`` or tetrahedra ``(m, 4)``.

    Returns
    -------
    numpy.ndarray
        Degrees. For triangles, shape ``(m, 3)``: the angles at the vertices
        0, 1 and 2 (60 each for the equilateral triangle). For tetrahedra,
        shape ``(m, 6)``: the dihedral angles at the edges (0, 1), (0, 2),
        (0, 3), (1, 2), (1, 3) and (2, 3) (70.53 each for the regular
        tetrahedron). A flat tetrahedron has angles 0.

    Raises
    ------
    ValueError
        If the shapes do not match or an index is out of range.
    """
    return _angles(_corners(points, cells))


def _angles(p: np.ndarray) -> np.ndarray:
    """Angles of triangles, or dihedral angles of tetrahedra, from their vertices.

    Parameters
    ----------
    p : numpy.ndarray
        Vertices, shape ``(m, 3, 2)`` or ``(m, 4, 3)``.

    Returns
    -------
    numpy.ndarray
        Degrees, shape ``(m, 3)`` or ``(m, 6)`` (see :func:`cell_angles`).
    """
    if p.shape[1] == 3:
        u = p[:, [1, 2, 0]] - p
        v = p[:, [2, 0, 1]] - p
        sine = np.abs(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
        return np.degrees(np.arctan2(sine, np.einsum("ijk,ijk->ij", u, v)))
    normals = _inward_normals(p)
    flat = ~np.all(np.linalg.norm(normals, axis=2) > 0, axis=1)
    angles = []
    for edge in combinations(range(4), 2):
        k, m = (i for i in range(4) if i not in edge)
        # The faces opposite k and m meet at the edge; their inward normals
        # make the supplement of the dihedral angle.
        angles.append(np.arctan2(np.linalg.norm(np.cross(normals[:, k], normals[:, m]), axis=1),
                                 -np.einsum("ij,ij->i", normals[:, k], normals[:, m])))
    return np.where(flat[:, np.newaxis], 0.0, np.degrees(np.stack(angles, axis=1)))


def _quality(p: np.ndarray, angle_name: str) -> dict:
    """Shape-quality entries of an audit.

    Parameters
    ----------
    p : numpy.ndarray
        Vertices of valid cells, shape ``(m, 3, 2)`` or ``(m, 4, 3)``.
    angle_name : str
        ``"angle"`` for triangles, ``"dihedral"`` for tetrahedra.

    Returns
    -------
    dict
        ``radius_ratio_min``, ``radius_ratio_mean``, ``radius_ratio_p1``
        (1st percentile), ``<angle_name>_min_deg``, ``<angle_name>_max_deg``
        and ``inradius_min_km``.
    """
    inradius, ratio = _radii(p)
    angles = _angles(p)
    return {"radius_ratio_min": float(ratio.min()), "radius_ratio_mean": float(ratio.mean()),
            "radius_ratio_p1": float(np.percentile(ratio, 1)),
            f"{angle_name}_min_deg": float(angles.min()),
            f"{angle_name}_max_deg": float(angles.max()),
            "inradius_min_km": float(inradius.min())}


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
        ``edge_std_km``), shape quality (``radius_ratio_min``,
        ``radius_ratio_mean``, ``radius_ratio_p1`` the 1st percentile,
        ``angle_min_deg``, ``angle_max_deg``, ``inradius_min_km``),
        ``bounds_min`` and ``bounds_max``, ``signed_area_counts``,
        ``crossing_triangles`` (per depth, a vertex within 1e-9 km of the
        depth being on it) and ``interface_conforming``.

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
                **_quality(p, "angle"),
                bounds_min=p.reshape(-1, 2).min(0).tolist(),
                bounds_max=p.reshape(-1, 2).max(0).tolist(),
                signed_area_counts=dict(positive=int(np.sum(oriented > 0)),
                                        negative=int(np.sum(oriented < 0))),
                crossing_triangles=crossing,
                interface_conforming=all(n == 0 for n in crossing.values()))


def audit_tetrahedra(points: np.ndarray, tetrahedra: np.ndarray,
                     seafloor: FacetedSurface | None = None) -> dict:
    """Audit a tetrahedralization in ``(x, y, z)`` coordinates.

    The keys include those of the 3D Salt campaigns' audit, so their records
    stay comparable; the shape quality is new.

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
        ``edge_max_km``, shape quality (``radius_ratio_min``,
        ``radius_ratio_mean``, ``radius_ratio_p1`` the 1st percentile,
        ``dihedral_min_deg``, ``dihedral_max_deg``, ``inradius_min_km``),
        ``bounds_min`` and ``bounds_max``,
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
                  **_quality(p, "dihedral"),
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
               seafloor: FacetedSurface | None = None,
               family: str = "KMV") -> dict:
    """Metadata of a mesh file: hash, dimension, DOF budget and audit.

    Parameters
    ----------
    path : str or pathlib.Path
        A triangular (spyro ``(z, x)``) or tetrahedral gmsh mesh.
    degree : int
        Degree of the element of the budget.
    conforming_depths : sequence of float, optional
        Horizontal interfaces of a 2D mesh.
    seafloor : FacetedSurface, optional
        Interface of a 3D mesh.
    family : str, optional
        Element family of the budget, a FIAT name or a Firedrake short name
        (see :func:`~femwavemesh.counting.element_family`). Default ``"KMV"``.

    Returns
    -------
    dict
        ``path``, ``sha256``, ``dimension``, ``element`` (FIAT name),
        ``degree``, ``dofs`` (one component) and ``geometry``, the output of
        :func:`audit_triangles` or :func:`audit_tetrahedra`. For KMV, also
        ``kmv_degree`` and ``kmv_dofs``, the keys of earlier records.
    """
    points, cells = read_mesh(path)
    dimension = points.shape[1]
    geometry = (audit_triangles(points, cells, conforming_depths) if dimension == 2
                else audit_tetrahedra(points, cells, seafloor))
    element = element_family(family)
    dofs = count_dofs(cells, element, degree)
    record = {"path": str(path),
              "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              "dimension": dimension, "element": element, "degree": degree,
              "dofs": dofs}
    if element == KMV:
        record.update(kmv_degree=degree, kmv_dofs=dofs)
    record["geometry"] = geometry
    return record
