"""Degrees of freedom and local size of a mesh.

The KMV (Kong-Mulder-Veldhuizen) mass-lumped elements put a fixed number of
degrees of freedom on each vertex, edge, face and cell interior, so the
dimension of the scalar space follows from the entity counts of the mesh.
The table is FIAT's (``FIAT.KongMulderVeldhuizen``, firedrake-fiat
2026.10.0), copied so that counting needs no FIAT.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

#: KMV degrees of freedom per vertex, edge, (face,) and cell interior, by
#: cell dimension and degree.
KMV_ENTITY_DOFS = {
    2: {1: (1, 0, 0), 2: (1, 1, 1), 3: (1, 2, 3), 4: (1, 3, 6),
        5: (1, 4, 15), 6: (1, 5, 21)},
    3: {1: (1, 0, 0, 0), 2: (1, 1, 1, 1), 3: (1, 2, 3, 4)},
}


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
    ValueError
        If there is no KMV element of that degree on those cells.
    """
    dimension = np.shape(cells)[1] - 1
    try:
        per_entity = KMV_ENTITY_DOFS[dimension][degree]
    except KeyError:
        raise ValueError(f"No KMV element of degree {degree} on "
                         f"{dimension}D simplices.") from None
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
                    z: np.ndarray, x: np.ndarray, *, fma: bool = False) -> np.ndarray:
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
    fma : bool, optional
        Locate points as a matplotlib built with fused multiply-add does
        (see :func:`find_triangles`).

    Returns
    -------
    numpy.ndarray
        Element size on the grid, shape ``(nz, nx)``.
    """
    from scipy.ndimage import distance_transform_edt

    sizes = mean_edge_lengths(points, triangles)
    cell = find_triangles(points, triangles, z, x, fma=fma)
    h = np.where(cell >= 0, sizes[np.maximum(cell, 0)], np.nan)
    missing = np.isnan(h)
    if missing.any():
        nearest = distance_transform_edt(missing, return_distances=False,
                                         return_indices=True)
        h = h[tuple(nearest)]
    return h


def find_triangles(points: np.ndarray, triangles: np.ndarray, z: np.ndarray,
                   x: np.ndarray, chunk: int = 2_000_000, *,
                   fma: bool = False) -> np.ndarray:
    """Triangle containing each point of a regular grid.

    The answer is the one of matplotlib's ``TrapezoidMapTriFinder`` on
    ``Triangulation(points[:, 0], points[:, 1], triangles)``. In that frame
    (first coordinate ``z`` as abscissa) each edge is directed from its
    leftmost end (lowest abscissa, then lowest ordinate) to the other, and
    its orientation test is
    :math:`a d - b c` with :math:`(a, b) = P - L` and :math:`(c, d) = R - L`,
    exactly as there. A point is tested only against the edges whose span
    in that left-to-right order holds it, and a point equal to a vertex is
    caught before any test. A point strictly inside a triangle takes it; a
    point on an edge takes the triangle on the negative side of the test
    ("above" the edge) if there is one, else the other; a point on a vertex
    takes the lowest-index triangle around the vertex; a point on no
    triangle gets ``-1``.

    The trifinder is compiled code, and whether its orientation test is
    rounded twice or contracted into one fused multiply-add depends on the
    build: x86-64 wheels round twice, the macOS arm64 wheels compute
    ``fma(a, d, -(b * c))``. The two differ only for points on an edge (or
    within rounding of one) that is neither horizontal nor vertical: rounded
    twice, :math:`a d - b c` is often exactly zero and the point takes the
    triangle above; fused, it has the sign of the rounding error of
    :math:`b c` and the point goes to either side.

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
    fma : bool, optional
        Evaluate the orientation test as a fused multiply-add, like a
        matplotlib built for arm64, instead of rounding both products, like
        one built for x86-64 (the default).

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
        qz, qx = z[iz][:, None], x[ix][:, None]
        lz, lx = points[left[t], 0], points[left[t], 1]
        rz, rx = points[right[t], 0], points[right[t], 1]
        # Only the edges whose left-to-right span holds the point are tested.
        spans = (((qz > lz) | ((qz == lz) & (qx > lx)))
                 & ((qz < rz) | ((qz == rz) & (qx < rx))))
        a, b = qz - lz, qx - lx
        c, d = rz - lz, rx - lx
        cross = fused_difference(a, d, b * c) if fma else a * d - b * c
        on = (cross == 0.0) & spans
        inside = np.where(above[t], cross < 0.0, cross > 0.0) | on | ~spans
        valid = inside.all(axis=1) & spans.any(axis=1)
        answer = np.where(on.any(axis=1), edge_answer[t, np.argmax(on, axis=1)], t)
        # A vertex is found by equality, before any orientation test.
        tz, tx = points[corners[t], 0], points[corners[t], 1]
        at_vertex = (tz == qz) & (tx == qx)
        on_vertex = at_vertex.any(axis=1)
        answer = np.where(on_vertex,
                          vertex_answer[corners[t, np.argmax(at_vertex, axis=1)]], answer)
        valid |= on_vertex
        cell[(iz * len(x) + ix)[valid]] = answer[valid]
    return cell.reshape(len(z), len(x))


def fused_difference(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Compute ``a * b - c`` with the sign of ``fma(a, b, -c)``, elementwise.

    numpy has no fused multiply-add, so ``a * b`` is split exactly into its
    rounded value ``p`` and its error ``e`` (Dekker's product), and the
    result is ``(p - c) + e``. Its sign is that of the exact ``a * b - c``:
    when ``p`` and ``c`` are within a factor of two of each other ``p - c``
    is exact (Sterbenz), otherwise ``|p - c| >= |p| / 2`` outweighs
    ``|e| <= 2**-53 |p|``. The value itself may differ from the fused one
    in the last bit, which the orientation test never reads.

    Parameters
    ----------
    a, b : numpy.ndarray
        Factors, small enough for their halves not to overflow
        (``|a|, |b| < 2**996``).
    c : numpy.ndarray
        Subtrahend.

    Returns
    -------
    numpy.ndarray
        A number with the sign of ``a * b - c``, zero only when it is zero.
    """
    p = a * b
    split = 134217729.0  # 2**27 + 1
    a_big, b_big = split * a, split * b
    a_hi, b_hi = a_big - (a_big - a), b_big - (b_big - b)
    a_lo, b_lo = a - a_hi, b - b_hi
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return (p - c) + e
