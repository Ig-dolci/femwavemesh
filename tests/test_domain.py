"""Domains and the faceted surface."""

import numpy as np
import pytest

from kmvmesh import BoxDomain, FacetedSurface, RectangularDomain


def test_rectangle_interior_and_grid() -> None:
    """The pad is excluded on three sides; the grid covers the domain."""
    domain = RectangularDomain(z_min=-1.0, x_min=0.0, x_max=1.2, pad=0.2)
    inside = domain.in_interior(np.array([-0.5, -0.9, -0.5, 0.0]),
                                np.array([0.6, 0.6, 1.1, 0.2]))
    np.testing.assert_array_equal(inside, [True, False, False, True])
    z, x = domain.grid(0.1)
    assert (z[0], x[0]) == (-1.0, 0.0)
    assert z[-1] == pytest.approx(0.0) and x[-1] == pytest.approx(1.2)
    assert len(z) == 11 and len(x) == 13


@pytest.mark.parametrize("extents", [dict(z_min=0.0, x_min=0.0, x_max=1.0),
                                     dict(z_min=-1.0, x_min=0.0, x_max=1.0, pad=0.5),
                                     dict(z_min=-1.0, x_min=0.0, x_max=1.0, pad=-0.1)])
def test_rectangle_rejects_bad_extents(extents: dict) -> None:
    """Empty domains and pads that leave no interior are rejected.

    Parameters
    ----------
    extents : dict
        Invalid arguments.
    """
    with pytest.raises(ValueError):
        RectangularDomain(**extents)


def test_box_interior_and_validation() -> None:
    """The box pad is excluded on five sides, not at the top."""
    box = BoxDomain(x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0, z_min=-1.0, pad=0.2)
    inside = box.in_interior(np.array([1.0, 0.1, 1.0, 1.0, 1.0]),
                             np.array([0.5, 0.5, 0.9, 0.5, 0.5]),
                             np.array([-0.5, -0.5, -0.5, -0.9, 0.0]))
    np.testing.assert_array_equal(inside, [True, False, False, False, True])
    with pytest.raises(ValueError):
        BoxDomain(x_min=0.0, x_max=2.0, y_min=0.0, y_max=0.3, z_min=-1.0, pad=0.2)
    with pytest.raises(ValueError):
        BoxDomain(x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0, z_min=0.5)


def test_faceted_surface_is_planar_on_each_triangle() -> None:
    """Exact at the nodes, linear on the two triangles of each cell."""
    x, y = np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])
    depth = np.array([[0.1, 0.2, 0.4], [0.3, 0.5, 0.2]])
    surface = FacetedSurface(x, y, depth)
    X, Y = np.meshgrid(x, y)
    np.testing.assert_allclose(surface.depth_at(X, Y), depth, rtol=0, atol=1e-11)
    # Lower triangle (0,0), (1,0), (1,1) of the first cell: u >= v.
    assert surface.depth_at(0.75, 0.25) == pytest.approx(0.1 + 0.75 * 0.1 + 0.25 * 0.3)
    # Upper triangle (0,0), (1,1), (0,1): u < v.
    assert surface.depth_at(0.25, 0.75) == pytest.approx(0.1 + 0.75 * 0.2 + 0.25 * 0.2)
    side = surface.side(np.array([[0.0, 0.0, -0.2], [0.0, 0.0, -0.05], [0.0, 0.0, -0.1]]))
    np.testing.assert_array_equal(side, [1, -1, 0])


def test_faceted_surface_from_water_depth() -> None:
    """A flat water depth stays flat; the facets span the box."""
    box = BoxDomain(x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.6, z_min=-1.0)
    xs, ys = np.linspace(-0.1, 1.1, 61), np.linspace(-0.1, 0.7, 41)
    surface = FacetedSurface.from_water_depth(np.full((41, 61), 0.08), xs, ys, box,
                                              sigma=0.05, spacing=0.2)
    np.testing.assert_array_equal(surface.x, np.linspace(0.0, 1.0, 6))
    np.testing.assert_array_equal(surface.y, np.linspace(0.0, 0.6, 4))
    np.testing.assert_allclose(surface.depth, 0.08)
    with pytest.raises(ValueError):
        FacetedSurface(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.zeros((3, 2)))
