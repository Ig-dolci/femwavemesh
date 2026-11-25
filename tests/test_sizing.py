"""Size laws, the budget fit and the 3D point sizes."""

import math

import numpy as np
import pytest

from kmvmesh import (AdaptiveSizeField, CPWSize, ScatteredSize, SizeLaw, UniformSize,
                     VelocityGrid3D, cpw_size, dof_calibration, fit_to_budget,
                     size_perturbation, source_refinement)


def test_cpw_size() -> None:
    """``h = v / (C f)`` for numbers and arrays."""
    assert cpw_size(1.5, 3.0, 5.0) == pytest.approx(0.1)
    np.testing.assert_allclose(cpw_size(np.array([1.5, 3.0]), 3.0, 5.0), [0.1, 0.2])


def test_size_perturbation_statistics() -> None:
    """Seed 0 is unperturbed; other seeds are smooth, reproducible and distinct."""
    shape, spacing = (120, 160), 0.005
    np.testing.assert_array_equal(size_perturbation(shape, spacing, 0), 1.0)
    first = size_perturbation(shape, spacing, 1)
    np.testing.assert_array_equal(first, size_perturbation(shape, spacing, 1))
    assert first.mean() == pytest.approx(1.0)
    assert first.std() == pytest.approx(0.03)
    assert np.abs(np.diff(first, axis=1)).max() < 0.01
    assert not np.allclose(first, size_perturbation(shape, spacing, 2))
    assert size_perturbation((20, 20, 20), 0.02, 3).shape == (20, 20, 20)


def test_source_refinement() -> None:
    """``1/factor`` inside the radius, 1 beyond the taper, log-linear between."""
    z, x = np.linspace(-1.0, 0.0, 101), np.linspace(0.0, 1.0, 101)
    scale = source_refinement((z, x), [(0.0, 0.5)], 0.2, 2.0, 0.1)
    assert scale[-1, 50] == 0.5
    assert scale[-1, 0] == 1.0
    assert scale[-1, 75] == pytest.approx(2.0 ** -0.5)  # 0.25 km: halfway
    step = source_refinement((z, x), [(0.0, 0.5)], 0.2, 2.0, 0.0)
    assert set(np.unique(step)) == {0.5, 1.0}
    cube = source_refinement((x, x, z), [(0.5, 0.5, 0.0), (0.1, 0.1, -0.9)], 0.1, 3.0, 0.05)
    assert cube.shape == (101, 101, 101)
    assert cube[50, 50, -1] == pytest.approx(1 / 3) and cube[10, 10, 10] == pytest.approx(1 / 3)


def test_dof_calibration() -> None:
    """DOFs per unit of the integral of ``h^-d``."""
    assert dof_calibration(1000, np.full((10, 10), 0.1), 0.1) == pytest.approx(1000 / 100.0)
    assert dof_calibration(1000, np.full((4, 5, 5), 0.1), 0.1) == pytest.approx(1000 / 100.0)


def two_layer_law(dimension: int, **kwargs: object) -> SizeLaw:
    """Build a law with a density peak in the middle of a uniform current size.

    Parameters
    ----------
    dimension : int
        2 or 3.
    **kwargs : object
        Extra arguments of :class:`SizeLaw`.

    Returns
    -------
    SizeLaw
        Law on a grid of 21 points per axis, spacing 0.05.
    """
    axis = np.linspace(0.0, 1.0, 21)
    grid = np.meshgrid(*[axis] * dimension, indexing="ij")
    density = 1.0 + 50.0 * np.exp(-sum((g - 0.5) ** 2 for g in grid) / 0.01)
    return SizeLaw(density, np.full(density.shape, 0.1), 0.05, **kwargs)


@pytest.mark.parametrize("dimension", [2, 3])
def test_size_law_budget_bounds_and_shape(dimension: int) -> None:
    """The law meets a budget, stays within bounds and refines the peak.

    Parameters
    ----------
    dimension : int
        Dimension of the grid.
    """
    law = two_layer_law(dimension, order=3.0)
    assert law.dimension == dimension
    current = law.h_max.copy()
    law.set_floor(np.full(current.shape, 2.0), 5.0, equivalent_cpw=8.0)
    np.testing.assert_allclose(law.h_min, 2.0 / (1.25 * 8.0 * 5.0))
    base = float((current ** -dimension).sum() * 0.05 ** dimension)
    scale = law.scale_for(1.5 * base)
    law.set_scale(scale)
    assert law.relative_cost(scale) == pytest.approx(1.5 * base, rel=1e-3)
    assert np.all(law.h <= current) and np.all(law.h >= law.h_min)
    middle = (10,) * dimension
    assert law.h[middle] < law.h[(0,) * dimension]
    # The shape follows the exponents q/(q+d) and -1/(q+d).
    np.testing.assert_allclose(law.shape[middle] / law.shape[(0,) * dimension],
                               (51.0 / (1.0 + 50.0 * np.exp(-0.25 * dimension / 0.01)))
                               ** (-1.0 / (3.0 + dimension)))


def test_size_law_ceiling_and_validation() -> None:
    """A ceiling replaces the current size as upper bound, not in the shape."""
    ceiling = np.full((21, 21), 0.05)
    law = two_layer_law(2, ceiling=ceiling)
    plain = two_layer_law(2)
    np.testing.assert_array_equal(law.shape, plain.shape)
    assert np.all(law.size(100.0) == 0.05)
    with pytest.raises(ValueError):
        SizeLaw(np.ones((3, 3)), np.ones((3, 4)), 0.1)
    with pytest.raises(ValueError):
        SizeLaw(np.ones((3, 3)), np.ones((3, 3)), 0.1, ceiling=np.ones(9))


def test_adaptive_size_field_grid() -> None:
    """The 2D field hands gmsh a grid with the law's size."""
    z, x = np.linspace(-1.0, 0.0, 21), np.linspace(0.0, 1.0, 21)
    field = AdaptiveSizeField(np.ones((21, 21)), np.full((21, 21), 0.1), z, x)
    assert field.grid is None and field.h is None
    field.set_scale(0.05)
    assert (field.grid.z0, field.grid.x0) == (-1.0, 0.0)
    assert field.grid.dz == field.grid.dx == pytest.approx(0.05)
    np.testing.assert_array_equal(field.grid.values, field.h)
    with pytest.raises(ValueError):
        AdaptiveSizeField(np.ones((21, 20)), np.ones((21, 20)), z, x)


@pytest.mark.parametrize("dimension", [2, 3])
def test_fit_to_budget(dimension: int) -> None:
    """The fit corrects the scale by ``(N / N_target)^(1/d)`` until in tolerance.

    Parameters
    ----------
    dimension : int
        Dimension of the law.
    """
    # A ceiling out of reach keeps the cost a power of the scale, so one
    # correction lands on the target.
    shape = (21,) * dimension
    law = two_layer_law(dimension, ceiling=np.full(shape, 10.0))
    calibration = 100.0
    scales = []

    def build(fitted: SizeLaw) -> int:
        """Pretend to mesh, making 30% more DOFs than predicted.

        Parameters
        ----------
        fitted : SizeLaw
            The law at its current scale.

        Returns
        -------
        int
            DOFs of the "mesh".
        """
        scales.append(float(fitted.h.flat[0] / fitted.shape.flat[0]))
        return round(1.3 * calibration * float((fitted.h ** -dimension).sum()
                                               * 0.05 ** dimension))

    target = 5000
    dofs = fit_to_budget(law, target, calibration, build, tolerance=0.02)
    assert abs(dofs / target - 1.0) <= 0.02
    assert len(scales) == 2
    assert scales[1] / scales[0] == pytest.approx(1.3 ** (1.0 / dimension), rel=1e-3)
    # Never more attempts than allowed, and the last mesh is kept.
    assert fit_to_budget(law, target, calibration, lambda fitted: 10 * target,
                         attempts=3) == 10 * target


def test_fit_to_budget_uses_the_square_root_in_2d() -> None:
    """The 2D correction is ``math.sqrt``, the arithmetic of spyro's loop."""
    law = two_layer_law(2)
    seen = []
    answers = iter([2000, 1000])

    def build(fitted: SizeLaw) -> int:
        """Record the scale through the size of a free cell, and answer.

        Parameters
        ----------
        fitted : SizeLaw
            The law at its current scale.

        Returns
        -------
        int
            The next scripted DOF count.
        """
        seen.append(float(fitted.h[0, 0] / fitted.shape[0, 0]))
        return next(answers)

    law.h_max = np.full(law.shape.shape, 1e9)
    fit_to_budget(law, 1000, 1000 / law.relative_cost(1.0), build)
    assert seen[1] == pytest.approx(seen[0] * math.sqrt(2.0), rel=1e-15)


def test_point_sizes() -> None:
    """Uniform, CPW and scattered sizes and their bounds."""
    uniform = UniformSize(0.2)
    assert uniform(1.0, 2.0, -3.0) == 0.2 and uniform.bounds == (0.2, 0.2)
    x, y, depth = np.linspace(0, 1, 11), np.linspace(0, 0.5, 6), np.linspace(0, 0.8, 9)
    D, Y, X = np.meshgrid(depth, y, x, indexing="ij")
    velocity = VelocityGrid3D(2.0 + X + D, x, y, depth)
    cpw = CPWSize(velocity, 4.0, 5.0)
    assert cpw(0.5, 0.2, -0.4) == pytest.approx((2.0 + 0.5 + 0.4) / 20.0)
    assert cpw.bounds == pytest.approx((2.0 / 20.0, 3.8 / 20.0))
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0],
                       [1.0, 1.0, -1.0]])
    scattered = ScatteredSize(points, np.array([0.1, 0.2, 0.2, 0.2, 0.4]))
    assert scattered(0.0, 0.0, 0.0) == 0.1
    assert scattered(1.0, 1.0, -1.0) == 0.4
    assert 0.1 < scattered(0.3, 0.3, -0.3) < 0.2
    assert scattered.bounds == (0.1, 0.4)
    few = ScatteredSize(points[:2], np.array([0.1, 0.4]))
    assert few(0.5, 0.0, 0.0) == pytest.approx(0.2)  # geometric mean, equal weights
    with pytest.raises(ValueError):
        ScatteredSize(points, np.array([0.1, 0.2, 0.2, 0.2, -0.4]))
    with pytest.raises(ValueError):
        ScatteredSize(points[:, :2], np.ones(5))
