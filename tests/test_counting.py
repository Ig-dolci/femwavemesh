"""KMV counts, edge lengths, point location and local size."""

import numpy as np
import pytest

from kmvmesh import count_kmv_dofs, local_mesh_size, mean_edge_lengths
from kmvmesh.counting import entity_counts, find_triangles

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


def test_point_location_against_matplotlib() -> None:
    """Random triangulations with grids that hit edges and vertices."""
    mtri = pytest.importorskip("matplotlib.tri")
    from scipy.spatial import Delaunay

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
                np.testing.assert_array_equal(find_triangles(points, triangles, z, x),
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
