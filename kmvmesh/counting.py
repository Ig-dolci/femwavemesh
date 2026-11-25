"""Degrees of freedom and local size of a mesh.

The KMV (Kong-Mulder-Veldhuizen) mass-lumped elements put a fixed number of
degrees of freedom on each vertex, edge, face and cell interior, so the
dimension of the scalar space follows from the entity counts of the mesh.
The numbers per entity are those of FIAT's ``KongMulderVeldhuizen`` element.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def kmv_entity_dofs(dimension: int, degree: int) -> tuple[int, ...]:
    """KMV degrees of freedom per vertex, edge, (face,) and cell interior.

    Parameters
    ----------
    dimension : int
        Dimension of the simplex: 2 for triangles, 3 for tetrahedra.
    degree : int
        Polynomial degree of the KMV element.

    Returns
    -------
    tuple of int
        Degrees of freedom on each entity of dimension 0 to ``dimension``,
        from ``FIAT.KongMulderVeldhuizen``.

    Raises
    ------
    ImportError
        If FIAT is not installed.
    ValueError
        If FIAT has no KMV element of that degree on those simplices.
    """
    # Deferred: importing FIAT takes about two seconds.
    import FIAT

    try:
        element = FIAT.KongMulderVeldhuizen(FIAT.ufc_simplex(dimension), degree)
    except NotImplementedError:
        raise ValueError(f"No KMV element of degree {degree} on "
                         f"{dimension}D simplices.") from None
    entity_dofs = element.entity_dofs()
    return tuple(len(entity_dofs[d][0]) for d in range(dimension + 1))


def entity_counts(cells: np.ndarray) -> tuple[int, ...]:
    """Count the vertices, edges, (faces,) and cells of a simplicial mesh.

    Parameters
    ----------
    cells : numpy.ndarray
        Vertex indices of triangles ``(n, 3)`` or tetrahedra ``(n, 4)``.

    Returns
    -------
    tuple of int
        ``(vertices, edges, triangles)`` or
        ``(vertices, edges, faces, tetrahedra)``.
    """
    cells = np.asarray(cells)
    counts = [len(np.unique(cells))]
    for size in range(2, cells.shape[1]):
        entities = np.concatenate([np.sort(cells[:, list(corners)], axis=1)
                                   for corners in combinations(range(cells.shape[1]), size)])
        counts.append(len(np.unique(entities, axis=0)))
    counts.append(len(cells))
    return tuple(counts)


def count_kmv_dofs(cells: np.ndarray, degree: int) -> int:
    """Count the KMV degrees of freedom of a triangulation or tetrahedralization.

    Parameters
    ----------
    cells : numpy.ndarray
        Vertex indices, shape ``(n_triangles, 3)`` or ``(n_tetrahedra, 4)``.
    degree : int
        Polynomial degree of the KMV element: 1 to 6 on triangles, 1 to 3 on
        tetrahedra.

    Returns
    -------
    int
        Dimension of the scalar KMV space on the mesh.

    Raises
    ------
    ImportError
        If FIAT is not installed.
    ValueError
        If there is no KMV element of that degree on those cells.
    """
    per_entity = kmv_entity_dofs(np.shape(cells)[1] - 1, degree)
    return int(sum(dofs * count for dofs, count
                   in zip(per_entity, entity_counts(cells))))


def mean_edge_lengths(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Mean edge length of each triangle.

    This is the convention of the target size handed to gmsh, so a mesh
    built from ``size`` has ``mean_edge_lengths`` close to ``size``.

    Parameters
    ----------
    points : numpy.ndarray
        Vertex coordinates, shape ``(n_vertices, 2)``.
    triangles : numpy.ndarray
        Vertex indices, shape ``(n_triangles, 3)``.

    Returns
    -------
    numpy.ndarray
        One length per triangle.
    """
    q = points[triangles]
    return np.linalg.norm(q[:, [1, 2, 0]] - q, axis=2).mean(axis=1)


def local_mesh_size(points: np.ndarray, triangles: np.ndarray,
                    z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sample the local element size of a mesh on a regular grid.

    Each grid point takes the mean edge length of the triangle that contains
    it; points that fall on no triangle take the value of the nearest point
    that does. A point on an edge or a vertex takes the triangle that
    matplotlib's ``TrapezoidMapTriFinder`` would give (see
    :func:`find_triangles`).

    Parameters
    ----------
    points : numpy.ndarray
        Vertex coordinates ``(z, x)``.
    triangles : numpy.ndarray
        Vertex indices.
    z : numpy.ndarray
        Increasing grid depths, shape ``(nz,)``.
    x : numpy.ndarray
        Increasing grid positions, shape ``(nx,)``.

    Returns
    -------
    numpy.ndarray
        Element size on the grid, shape ``(nz, nx)``.
    """
    from scipy.ndimage import distance_transform_edt

    sizes = mean_edge_lengths(points, triangles)
    cell = find_triangles(points, triangles, z, x)
    h = np.where(cell >= 0, sizes[np.maximum(cell, 0)], np.nan)
    missing = np.isnan(h)
    if missing.any():
        nearest = distance_transform_edt(missing, return_distances=False,
                                         return_indices=True)
        h = h[tuple(nearest)]
    return h


def find_triangles(points: np.ndarray, triangles: np.ndarray, z: np.ndarray,
                   x: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """Triangle containing each point of a regular grid.

    The answer is the one of matplotlib's ``TrapezoidMapTriFinder`` on
    ``Triangulation(points[:, 0], points[:, 1], triangles)``. In that frame
    (first coordinate ``z`` as abscissa) each edge is directed from its
    leftmost end (lowest abscissa, then lowest ordinate) to the other, and
    its orientation test is
    :math:`(P - L)_1 (R - L)_2 - (P - L)_2 (R - L)_1`, exactly as there.
    A point strictly inside a triangle takes it; a point on an edge takes the
    triangle on the negative side of the test ("above" the edge) if there is
    one, else the other; a point on a vertex takes the lowest-index triangle
    around the vertex; a point on no triangle gets ``-1``.

    Parameters
    ----------
    points : numpy.ndarray
        Vertex coordinates ``(z, x)``, shape ``(n_vertices, 2)``.
    triangles : numpy.ndarray
        Vertex indices, shape ``(n_triangles, 3)``.
    z : numpy.ndarray
        Increasing grid depths, shape ``(nz,)``.
    x : numpy.ndarray
        Increasing grid positions, shape ``(nx,)``.
    chunk : int, optional
        Largest number of (point, triangle) candidates handled at once.

    Returns
    -------
    numpy.ndarray
        Triangle index of each grid point, shape ``(nz, nx)``.
    """
    points = np.asarray(points, dtype=float)
    z = np.asarray(z, dtype=float)
    x = np.asarray(x, dtype=float)
    corners = np.array(triangles, dtype=np.int64)
    n_triangles = len(corners)
    # Anticlockwise corners, as matplotlib's Triangulation makes them.
    p = points[corners]
    clockwise = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                 - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])) < 0.0
    corners[clockwise, 1:] = corners[clockwise, 2:0:-1]

    # Each edge of each triangle, directed left to right.
    start = corners
    end = corners[:, [1, 2, 0]]
    ps, pe = points[start], points[end]
    end_right = (pe[..., 0] > ps[..., 0]) | ((pe[..., 0] == ps[..., 0])
                                             & (pe[..., 1] > ps[..., 1]))
    left = np.where(end_right, start, end)
    right = np.where(end_right, end, start)
    # The triangle is above the edges it runs left to right (anticlockwise).
    above = end_right

    # Triangle a point on each edge gets: the one above, else the only one.
    key = left * len(points) + right
    flat_key = key.ravel()
    owner = np.repeat(np.arange(n_triangles), 3)
    order = np.lexsort((~above.ravel(), flat_key))
    unique_key, first = np.unique(flat_key[order], return_index=True)
    edge_answer = owner[order][first][np.searchsorted(unique_key, key)]

    # Triangle a point on each vertex gets: the lowest-index one around it.
    vertex_answer = np.full(len(points), n_triangles, dtype=np.int64)
    np.minimum.at(vertex_answer, corners.ravel(), owner)

    # Candidate grid points of each triangle: its bounding box.
    z_lo = np.searchsorted(z, p[:, :, 0].min(axis=1), side="left")
    z_hi = np.searchsorted(z, p[:, :, 0].max(axis=1), side="right")
    x_lo = np.searchsorted(x, p[:, :, 1].min(axis=1), side="left")
    x_hi = np.searchsorted(x, p[:, :, 1].max(axis=1), side="right")
    n_z = np.maximum(z_hi - z_lo, 0)
    n_x = np.maximum(x_hi - x_lo, 0)
    counts = n_z * n_x
    ends = np.cumsum(counts)
    cell = np.full(len(z) * len(x), -1, dtype=np.int64)
    begin = 0
    while begin < n_triangles:
        # Triangles [begin, stop) have at most `chunk` candidates (or one).
        stop = max(int(np.searchsorted(ends, ends[begin] - counts[begin] + chunk,
                                       side="right")), begin + 1)
        block = np.arange(begin, stop)
        begin = stop
        total = int(counts[block].sum())
        if total == 0:
            continue
        t = np.repeat(block, counts[block])
        offset = np.arange(total) - np.repeat(np.cumsum(counts[block]) - counts[block],
                                              counts[block])
        iz = z_lo[t] + offset // n_x[t]
        ix = x_lo[t] + offset % n_x[t]
        qz, qx = z[iz], x[ix]
        lz, lx = points[left[t], 0], points[left[t], 1]
        rz, rx = points[right[t], 0], points[right[t], 1]
        cross = ((qz[:, None] - lz) * (rx - lx) - (qx[:, None] - lx) * (rz - lz))
        on = cross == 0.0
        inside = np.where(above[t], cross < 0.0, cross > 0.0) | on
        valid = inside.all(axis=1)
        answer = np.where(on.any(axis=1), edge_answer[t, np.argmax(on, axis=1)], t)
        tz, tx = points[corners[t], 0], points[corners[t], 1]
        at_vertex = (tz == qz[:, None]) & (tx == qx[:, None])
        answer = np.where(at_vertex.any(axis=1),
                          vertex_answer[corners[t, np.argmax(at_vertex, axis=1)]], answer)
        cell[(iz * len(x) + ix)[valid]] = answer[valid]
    return cell.reshape(len(z), len(x))
