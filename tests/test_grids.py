"""2D and 3D velocity grids."""

import numpy as np
import pytest

from kmvmesh import FacetedSurface, VelocityGrid, VelocityGrid3D


def linear_grid() -> VelocityGrid:
    """Sample ``2 + z + 3 x``, which bilinear interpolation reproduces.

    Returns
    -------
    VelocityGrid
        Grid over ``[-1, 0] x [0, 1.2]`` with spacing 0.1.
    """
    z = np.linspace(-1.0, 0.0, 11)
    x = np.linspace(0.0, 1.2, 13)
    Z, X = np.meshgrid(z, x, indexing="ij")
    return VelocityGrid(2.0 + Z + 3.0 * X, -1.0, 0.0, 0.1, 0.1)


def test_bilinear_interpolation_and_clamping() -> None:
    """Exact on a linear field inside, nearest edge outside."""
    grid = linear_grid()
    z, x = np.array([-0.55, -0.12, -0.9]), np.array([0.33, 1.05, 0.71])
    np.testing.assert_allclose(grid(z, x), 2.0 + z + 3.0 * x)
    np.testing.assert_allclose(grid(np.array([-3.0]), np.array([0.5])),
                               grid(np.array([-1.0]), np.array([0.5])))


def test_scaled_and_fixed_layer() -> None:
    """Scaling keeps the geometry; the layer applies strictly above its depth."""
    grid = linear_grid()
    scaled = grid.scaled(0.25)
    assert (scaled.z0, scaled.x0, scaled.dz, scaled.dx) == (grid.z0, grid.x0,
                                                            grid.dz, grid.dx)
    z = np.array([-1.3, -0.4, 0.0])
    x = np.array([0.1, 0.7, 1.9])
    np.testing.assert_allclose(scaled(z, x), 0.25 * grid(z, x))
    layered = VelocityGrid(grid.values, -1.0, 0.0, 0.1, 0.1, fixed_layer=(-0.3, 1.5))
    values = layered(np.array([-0.2, -0.3, -0.5]), np.array([0.5, 0.5, 0.5]))
    assert values[0] == 1.5
    np.testing.assert_allclose(values[1:], grid(np.array([-0.3, -0.5]), np.array([0.5, 0.5])))
    assert layered.scaled(2.0).fixed_layer == (-0.3, 3.0)
    with pytest.raises(ValueError):
        layered.scaled(np.ones(grid.values.shape))
    with pytest.raises(ValueError):
        VelocityGrid(np.ones(5), 0.0, 0.0, 1.0, 1.0)


def linear_grid_3d(layer: tuple | None = None) -> VelocityGrid3D:
    """Sample ``2 + x - y - z`` in 3D, reproduced by trilinear interpolation.

    Parameters
    ----------
    layer : tuple, optional
        Fixed layer of the grid.

    Returns
    -------
    VelocityGrid3D
        Grid with spacing 0.1 over ``[0, 1] x [0, 0.5] x [0, 0.8]`` (depth).
    """
    x, y, depth = np.linspace(0, 1, 11), np.linspace(0, 0.5, 6), np.linspace(0, 0.8, 9)
    D, Y, X = np.meshgrid(depth, y, x, indexing="ij")
    return VelocityGrid3D(2.0 + X - Y + D, x, y, depth, fixed_layer=layer)


def test_trilinear_interpolation_grid_and_extremes() -> None:
    """Exact on a linear field, clamped outside, grid points in grid order."""
    grid = linear_grid_3d()
    x, y, z = np.array([0.33, 0.91]), np.array([0.12, 0.47]), np.array([-0.25, -0.71])
    np.testing.assert_allclose(grid(x, y, z), 2.0 + x - y - z)
    np.testing.assert_allclose(grid(np.array([2.0]), np.array([0.2]), np.array([-0.4])),
                               grid(np.array([1.0]), np.array([0.2]), np.array([-0.4])))
    points, values = grid.on_grid()
    assert points.shape == (9 * 6 * 11, 3) and values.shape == (9, 6, 11)
    np.testing.assert_allclose(points[1] - points[0], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(values, grid.values)
    assert grid.extremes() == (float(grid.values.min()), float(grid.values.max()))
    assert grid.spacing == pytest.approx(0.1)


def test_3d_fixed_layer() -> None:
    """The layer value holds on and above the surface, and counts in the extremes."""
    surface = FacetedSurface(np.array([0.0, 1.0]), np.array([0.0, 0.5]), np.full((2, 2), 0.3))
    grid = linear_grid_3d((surface, 1.5))
    values = grid(np.array([0.5, 0.5, 0.5]), np.array([0.2, 0.2, 0.2]),
                  np.array([-0.1, -0.3, -0.5]))
    np.testing.assert_allclose(values, [1.5, 1.5, 2.0 + 0.5 - 0.2 + 0.5])
    assert grid.extremes()[0] == 1.5
    assert np.all(grid.on_grid()[1][:4] == 1.5)


def test_3d_grid_validation() -> None:
    """Shapes must match the axes, which must share one spacing."""
    x, y, depth = np.linspace(0, 1, 11), np.linspace(0, 0.5, 6), np.linspace(0, 0.8, 9)
    with pytest.raises(ValueError):
        VelocityGrid3D(np.ones((9, 6, 10)), x, y, depth)
    with pytest.raises(ValueError):
        VelocityGrid3D(np.ones((9, 6, 11)), x, y, np.linspace(0, 1.6, 9))
