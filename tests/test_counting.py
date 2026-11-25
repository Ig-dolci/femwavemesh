"""KMV counts, edge lengths, point location and local size."""

from fractions import Fraction

import numpy as np
import pytest

from kmvmesh import count_kmv_dofs, local_mesh_size, mean_edge_lengths
from kmvmesh.counting import entity_counts, find_triangles, fused_difference

#: Dimension of KMV on one cell, by dimension and degree (FIAT).
SPACE = {2: {1: 3, 2: 7, 3: 12, 4: 18, 5: 30, 6: 39}, 3: {1: 4, 2: 15, 3: 32}}


@pytest.mark.parametrize("dimension", [2, 3])
def test_one_cell_has_the_element_dimension(dimension: int) -> None:
    """On one cell the count is the dimension of the element.

    Parameters
    ----------
    dimension : int
        Cell dimension.
    """
    cell = np.arange(dimension + 1)[None, :]
    for degree, size in SPACE[dimension].items():
        assert count_kmv_dofs(cell, degree) == size


def test_shared_entities_count_once() -> None:
    """Two triangles sharing an edge; a cube split into six tetrahedra."""
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    assert entity_counts(triangles) == (4, 5, 2)
    assert count_kmv_dofs(triangles, 2) == 4 + 5 + 2
    assert count_kmv_dofs(triangles, 4) == 4 + 3 * 5 + 6 * 2
    cube = np.array([[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7],
                     [0, 4, 5, 7], [0, 4, 6, 7]])
    assert entity_counts(cube) == (8, 19, 18, 6)
    assert count_kmv_dofs(cube, 3) == 8 + 2 * 19 + 3 * 18 + 4 * 6


def test_unknown_degree() -> None:
    """There is no KMV of degree 7 on triangles nor 4 on tetrahedra."""
    with pytest.raises(ValueError):
        count_kmv_dofs(np.array([[0, 1, 2]]), 7)
    with pytest.raises(ValueError):
        count_kmv_dofs(np.array([[0, 1, 2, 3]]), 4)


def square() -> tuple[np.ndarray, np.ndarray]:
    """Split the unit square by its diagonal, in ``(z, x)``.

    Returns
    -------
    points : numpy.ndarray
        The four corners.
    triangles : numpy.ndarray
        Two triangles, the second clockwise.
    """
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    return points, np.array([[0, 1, 2], [1, 2, 3]])


def test_mean_edge_lengths() -> None:
    """Mean of the three edges."""
    points, triangles = square()
    np.testing.assert_allclose(mean_edge_lengths(points, triangles),
                               (2.0 + np.sqrt(2.0)) / 3.0)


def test_point_location_ties() -> None:
    """Interior, edge and vertex points, as matplotlib's trifinder answers."""
    points, triangles = square()
    z = np.array([0.25, 0.5, 0.75])
    x = np.array([0.25, 0.5, 0.75])
    cell = find_triangles(points, triangles, z, x)
    # Triangle 0 is z + x < 1, triangle 1 is z + x > 1. The diagonal runs
    # from (0, 1) to (1, 0) in the trifinder's frame, and triangle 1 is on
    # its negative ("above") side: points on it take triangle 1.
    np.testing.assert_array_equal(cell, [[0, 0, 1], [0, 1, 1], [1, 1, 1]])
    corners = find_triangles(points, triangles, np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    # The vertices on the diagonal take the lowest-index triangle.
    np.testing.assert_array_equal(corners, [[0, 0], [0, 1]])
    outside = find_triangles(points, triangles, np.array([-0.5, 0.5]), np.array([0.5]))
    np.testing.assert_array_equal(outside, [[-1], [1]])


def test_fused_difference_sign() -> None:
    """``a * b - c`` has the exact sign, as a fused multiply-add gives it."""
    rng = np.random.default_rng(1)
    a, b = rng.random(3000) - 0.5, rng.random(3000) - 0.5
    a[:1000], b[:1000] = np.round(a[:1000] * 64) / 64, np.round(b[:1000] * 64) / 64
    near = a * b * (1.0 + rng.integers(-4, 5, 3000) * 2.0**-52)
    c = np.concatenate([a * b, near, rng.random(3000) - 0.5])
    a, b = np.tile(a, 3), np.tile(b, 3)
    exact = [np.sign(Fraction(p) * Fraction(q) - Fraction(r)) for p, q, r in zip(a, b, c)]
    assert 0 in exact and -1 in exact and 1 in exact
    np.testing.assert_array_equal(np.sign(fused_difference(a, b, c)), exact)


@pytest.mark.parametrize("fma", [False, True])
def test_point_on_vertex(fma: bool) -> None:
    """Every vertex takes the lowest-index triangle around it.

    Parameters
    ----------
    fma : bool
        Fused orientation test, which is not zero on the far end of an edge.
    """
    from scipy.spatial import Delaunay

    points = np.random.default_rng(2).random((40, 2))
    triangles = Delaunay(points).simplices
    z, x = np.sort(points[:, 0]), np.sort(points[:, 1])
    cell = find_triangles(points, triangles, z, x, fma=fma)
    lowest = [np.flatnonzero((triangles == v).any(axis=1)).min() for v in range(len(points))]
    np.testing.assert_array_equal(
        cell[np.searchsorted(z, points[:, 0]), np.searchsorted(x, points[:, 1])], lowest)


@pytest.mark.parametrize("fma", [False, True])
def test_point_tested_against_spanning_edges(fma: bool) -> None:
    """Only the edges whose left-to-right span holds a point decide it.

    One ulp above the vertex ``(1, 0.1)``, on the vertical edge of triangle 0,
    the point is also on both edges of triangle 1 that end at the vertex once
    the differences are rounded, but neither spans it: the trifinder gives
    triangle 0.

    Parameters
    ----------
    fma : bool
        Fused orientation test (the products here are exact either way).
    """
    points = np.array([[1.0, 0.1], [1.0, 2.0], [0.0, 1.0], [0.0, -1.0]])
    triangles = np.array([[0, 1, 2], [3, 0, 2]])
    x = np.array([0.1, np.nextafter(0.1, 1.0)])
    np.testing.assert_array_equal(
        find_triangles(points, triangles, np.array([1.0]), x, fma=fma), [[0, 0]])


def trifinder_fma(mtri) -> bool:
    """Whether matplotlib's trifinder evaluates its orientation test fused.

    Points on the diagonal of a square of side 1/11 are on it when the
    products are rounded, and on either side when fused.

    Parameters
    ----------
    mtri : module
        ``matplotlib.tri``.

    Returns
    -------
    bool
        True for a fused multiply-add (arm64 builds), False for rounded
        products (x86-64 builds).
    """
    side = 1.0 / 11.0
    points = np.array([[0.0, 0.0], [side, 0.0], [0.0, side], [side, side]])
    triangles = np.array([[0, 1, 3], [0, 3, 2]])
    z = x = np.linspace(0.0, side, 25)
    Z, X = np.meshgrid(z, x, indexing="ij")
    found = mtri.Triangulation(points[:, 0], points[:, 1], triangles).get_trifinder()(Z, X)
    rounded = find_triangles(points, triangles, z, x)
    fused = find_triangles(points, triangles, z, x, fma=True)
    assert not np.array_equal(rounded, fused), "the probe does not tell them apart"
    assert np.array_equal(found, rounded) or np.array_equal(found, fused), \
        "matplotlib's trifinder is neither rounded nor fused as fma(a, d, -(b * c))"
    return np.array_equal(found, fused)


def test_point_location_against_matplotlib() -> None:
    """Random triangulations with grids that hit edges and vertices."""
    mtri = pytest.importorskip("matplotlib.tri")
    from scipy.spatial import Delaunay

    fma = trifinder_fma(mtri)
    rng = np.random.default_rng(0)
    for n in (5, 12):
        axis = np.linspace(0.0, 1.0, n)
        lattice = np.array([(a, b) for a in axis for b in axis])
        jittered = np.unique(np.round(rng.random((200, 2)) * 20) / 20, axis=0)
        for points in (lattice, jittered):
            triangles = Delaunay(points).simplices
            triangles = triangles[rng.permutation(len(triangles))]
            grid = np.linspace(-0.05, 1.05, 4 * n + 1)
            exact = np.linspace(0.0, 1.0, 4 * n - 3)
            for z, x in ((grid, grid), (exact, exact)):
                Z, X = np.meshgrid(z, x, indexing="ij")
                finder = mtri.Triangulation(points[:, 0], points[:, 1], triangles).get_trifinder()
                np.testing.assert_array_equal(find_triangles(points, triangles, z, x, fma=fma),
                                              finder(Z, X))


def test_local_mesh_size_fills_outside() -> None:
    """Points on no triangle take the nearest located value."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [3.0, 2.0], [2.0, 3.0]])
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    z = x = np.linspace(0.0, 3.0, 7)
    h = local_mesh_size(points, triangles, z, x)
    sizes = mean_edge_lengths(points, triangles)
    assert h[0, 0] == sizes[0] and h[-1, -2] == sizes[1]
    assert not np.isnan(h).any()
    assert set(np.unique(h)) <= set(sizes)
