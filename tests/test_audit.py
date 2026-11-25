"""Audits of triangulations and tetrahedralizations."""

import numpy as np
import pytest

from kmvmesh import FacetedSurface, audit_tetrahedra, audit_triangles
from kmvmesh.io import zero_based

CUBE_POINTS = np.array([[x, y, -z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
CUBE = np.array([[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7],
                 [0, 4, 5, 7], [0, 4, 6, 7]])


def strip() -> tuple[np.ndarray, np.ndarray]:
    """Two layers of two triangles over ``[-1, 0] x [0, 1]``, joined at ``z = -0.5``.

    Returns
    -------
    points : numpy.ndarray
        Six vertices ``(z, x)``.
    triangles : numpy.ndarray
        Four triangles.
    """
    points = np.array([[-1.0, 0.0], [-1.0, 1.0], [-0.5, 0.0], [-0.5, 1.0], [0.0, 0.0],
                       [0.0, 1.0]])
    return points, np.array([[0, 1, 3], [0, 3, 2], [2, 3, 5], [2, 5, 4]])


def test_triangles() -> None:
    """Counts, area, edges and conformity to the line ``z = -0.5``."""
    points, triangles = strip()
    audit = audit_triangles(points, triangles, [-0.5, -0.25])
    assert (audit["vertices"], audit["edges"], audit["triangles"]) == (6, 9, 4)
    assert audit["area_km2"] == pytest.approx(1.0)
    assert audit["edge_min_km"] == pytest.approx(0.5)
    assert audit["edge_max_km"] == pytest.approx(np.hypot(0.5, 1.0))
    assert audit["crossing_triangles"] == {"-0.5": 0, "-0.25": 2}
    assert not audit["interface_conforming"]
    assert audit["signed_area_counts"]["positive"] + audit["signed_area_counts"]["negative"] == 4
    assert audit["bounds_min"] == [-1.0, 0.0] and audit["bounds_max"] == [0.0, 1.0]


def test_invalid_meshes() -> None:
    """Bad shapes, unknown vertices, degenerate and duplicate cells."""
    points, triangles = strip()
    with pytest.raises(ValueError):
        audit_triangles(points, triangles[:, :2])
    with pytest.raises(ValueError):
        audit_triangles(points, triangles + 3)
    with pytest.raises(ValueError):
        audit_triangles(points, np.array([[0, 2, 4]]))
    with pytest.raises(ValueError):
        audit_triangles(points, np.vstack([triangles, triangles[:1, ::-1]]))
    with pytest.raises(ValueError):
        audit_tetrahedra(CUBE_POINTS, np.vstack([CUBE, CUBE[:1]]))


def test_tetrahedra_and_seafloor() -> None:
    """The unit cube: counts, KMV3, volume and a floor across it."""
    audit = audit_tetrahedra(CUBE_POINTS, CUBE)
    assert (audit["vertices"], audit["edges"], audit["faces"], audit["tetrahedra"]) == \
        (8, 19, 18, 6)
    assert audit["KMV3_dofs_predicted"] == 8 + 2 * 19 + 3 * 18 + 4 * 6
    assert audit["volume_km3"] == pytest.approx(1.0)
    assert audit["edge_min_km"] == pytest.approx(1.0)
    assert audit["edge_max_km"] == pytest.approx(np.sqrt(3.0))
    assert "seafloor_crossing_tetrahedra" not in audit
    floor = FacetedSurface(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.full((2, 2), 0.5))
    crossed = audit_tetrahedra(CUBE_POINTS, CUBE, floor)
    assert crossed["seafloor_crossing_tetrahedra"] == 6
    assert not crossed["interface_conforming"]


def test_zero_based() -> None:
    """Turn non-contiguous gmsh tags into zero-based indices."""
    tags = np.array([10, 4, 7])
    coordinates = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    points, cells = zero_based(tags, coordinates, np.array([4, 7, 10]), 3, 2)
    np.testing.assert_array_equal(points, [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]])
    np.testing.assert_array_equal(cells, [[1, 2, 0]])
