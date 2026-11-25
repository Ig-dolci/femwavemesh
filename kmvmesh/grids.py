"""Velocities (and sizes) sampled on regular grids.

:class:`VelocityGrid` is the 2D sampler of spyro, in its ``(z, x)``
coordinates; :func:`~kmvmesh.generate_mesh` takes a mesh size in the same
form. :class:`VelocityGrid3D` is the 3D sampler of the Salt campaigns, on a
``(depth, y, x)`` grid.
"""

from __future__ import annotations

import numpy as np

from .domain import FacetedSurface


class VelocityGrid:
    """P-wave velocity sampled on a regular ``(z, x)`` grid.

    Values outside the grid are taken from its closest edge, so a model can be
    extended into an absorbing pad.

    Parameters
    ----------
    values : numpy.ndarray
        Velocities, shape ``(nz, nx)``, row ``i`` at depth ``z0 + i * dz``.
    z0 : float
        Depth of the first row (the deepest, since depths increase upwards).
    x0 : float
        Position of the first column.
    dz : float
        Row spacing.
    dx : float
        Column spacing.
    fixed_layer : tuple of float, optional
        ``(depth, value)``: the velocity is ``value`` strictly above the
        horizontal interface at ``depth``, whatever ``values`` hold there.
        Only the callable applies it; :attr:`values` are the grid alone.
    """

    def __init__(self, values: np.ndarray, z0: float, x0: float,
                 dz: float, dx: float,
                 fixed_layer: tuple[float, float] | None = None) -> None:
        self.values = np.asarray(values, dtype=float)
        if self.values.ndim != 2 or min(self.values.shape) < 2:
            raise ValueError("The velocity grid must be 2D with at least 2x2 points.")
        self.z0 = float(z0)
        self.x0 = float(x0)
        self.dz = float(dz)
        self.dx = float(dx)
        self.fixed_layer = fixed_layer

    def __call__(self, z: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of the velocity.

        Parameters
        ----------
        z : numpy.ndarray
            Depths.
        x : numpy.ndarray
            Horizontal positions, broadcastable with ``z``.

        Returns
        -------
        numpy.ndarray
            Velocities at the points.
        """
        nz, nx = self.values.shape
        fz = np.clip((np.asarray(z, dtype=float) - self.z0) / self.dz, 0, nz - 1.000001)
        fx = np.clip((np.asarray(x, dtype=float) - self.x0) / self.dx, 0, nx - 1.000001)
        i = fz.astype(int)
        j = fx.astype(int)
        tz = fz - i
        tx = fx - j
        v = self.values
        bilinear = ((1 - tz) * (1 - tx) * v[i, j] + tz * (1 - tx) * v[i + 1, j]
                    + (1 - tz) * tx * v[i, j + 1] + tz * tx * v[i + 1, j + 1])
        if self.fixed_layer is None:
            return bilinear
        depth, value = self.fixed_layer
        return np.where(np.asarray(z, dtype=float) > depth, value, bilinear)

    def scaled(self, factor: float | np.ndarray) -> VelocityGrid:
        """Return the same grid with every value multiplied by ``factor``.

        Parameters
        ----------
        factor : float or numpy.ndarray
            Multiplier, e.g. :math:`1/(C f)` for the uniform-CPW edge length,
            or an array of the grid's shape (see
            :func:`~kmvmesh.size_perturbation`).

        Returns
        -------
        VelocityGrid
            New grid with the same geometry.

        Raises
        ------
        ValueError
            If an array factor is applied to a grid with a fixed layer, whose
            value has no position to take a factor from.
        """
        layer = self.fixed_layer
        if layer is not None:
            if np.ndim(factor) != 0:
                raise ValueError("A grid with a fixed layer scales by a number only.")
            layer = (layer[0], layer[1] * float(factor))
        return VelocityGrid(self.values * factor, self.z0, self.x0, self.dz, self.dx,
                            fixed_layer=layer)


class VelocityGrid3D:
    """Velocity sampled on a regular ``(depth, y, x)`` grid.

    The three axes share one spacing, ``x[1] - x[0]``. Points outside the
    grid take the value of its closest face.

    Parameters
    ----------
    values : numpy.ndarray
        Velocities (km/s), shape ``(len(depth), len(y), len(x))``.
    x : numpy.ndarray
        Increasing ``x`` of the grid (km).
    y : numpy.ndarray
        Increasing ``y`` of the grid (km).
    depth : numpy.ndarray
        Increasing depths of the grid (positive downwards, km); the point
        ``(x, y, z)`` is at depth ``-z``.
    fixed_layer : tuple, optional
        ``(surface, value)`` with ``surface`` a
        :class:`~kmvmesh.FacetedSurface`: the velocity is ``value`` on and
        above the surface (within 1e-9 km), whatever ``values`` hold there,
        such as water over a sea floor. The callable and :meth:`on_grid`
        apply it.

    Attributes
    ----------
    spacing : float
        Grid spacing, ``x[1] - x[0]``.
    """

    def __init__(self, values: np.ndarray, x: np.ndarray, y: np.ndarray,
                 depth: np.ndarray,
                 fixed_layer: tuple[FacetedSurface, float] | None = None) -> None:
        self.values = np.asarray(values, dtype=float)
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.depth = np.asarray(depth, dtype=float)
        if self.values.shape != (len(self.depth), len(self.y), len(self.x)) \
                or min(self.values.shape) < 2:
            raise ValueError("The values must have shape (len(depth), len(y), len(x)), "
                             "at least 2 x 2 x 2.")
        self.spacing = float(self.x[1] - self.x[0])
        for axis in (self.x, self.y, self.depth):
            if not np.allclose(np.diff(axis), self.spacing, rtol=1e-6, atol=0.0):
                raise ValueError("The three axes must share one spacing.")
        self.fixed_layer = fixed_layer

    def _trilinear(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of :attr:`values`.

        Parameters
        ----------
        x : numpy.ndarray
            ``x`` of the points.
        y : numpy.ndarray
            ``y`` of the points.
        z : numpy.ndarray
            ``z`` of the points (negative downwards).

        Returns
        -------
        numpy.ndarray
            Interpolated values, without the fixed layer.
        """
        q = [np.clip((-np.asarray(z, float) - self.depth[0]) / self.spacing, 0,
                     len(self.depth) - 1),
             np.clip((np.asarray(y, float) - self.y[0]) / self.spacing, 0, len(self.y) - 1),
             np.clip((np.asarray(x, float) - self.x[0]) / self.spacing, 0, len(self.x) - 1)]
        i = [np.minimum(t.astype(int), n - 2) for t, n in zip(q, self.values.shape)]
        w = [t - j for t, j in zip(q, i)]
        value = 0.0
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    weight = ((w[0] if dz else 1 - w[0]) * (w[1] if dy else 1 - w[1])
                              * (w[2] if dx else 1 - w[2]))
                    value = value + self.values[i[0] + dz, i[1] + dy, i[2] + dx] * weight
        return np.asarray(value, float)

    def __call__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Velocity at points, with the fixed layer.

        Parameters
        ----------
        x : numpy.ndarray
            ``x`` of the points.
        y : numpy.ndarray
            ``y`` of the points, broadcastable with ``x``.
        z : numpy.ndarray
            ``z`` of the points (negative downwards).

        Returns
        -------
        numpy.ndarray
            Velocities (km/s).
        """
        v = self._trilinear(x, y, z)
        if self.fixed_layer is None:
            return v
        surface, value = self.fixed_layer
        layer = -np.asarray(z, float) <= surface.depth_at(x, y) + 1e-9
        return np.where(layer, value, v)

    def on_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the grid points and the velocity there.

        Returns
        -------
        points : numpy.ndarray
            Points ``(x, y, z)``, shape ``(n, 3)``, in the order of the grid
            (depth slowest, ``x`` fastest).
        velocity : numpy.ndarray
            Velocity with the fixed layer, shape of :attr:`values`.
        """
        Z, Y, X = np.meshgrid(self.depth, self.y, self.x, indexing="ij")
        points = np.column_stack([X.ravel(), Y.ravel(), -Z.ravel()])
        return points, self(points[:, 0], points[:, 1], points[:, 2]).reshape(Z.shape)

    def extremes(self) -> tuple[float, float]:
        """Smallest and largest velocity, the fixed layer included.

        Returns
        -------
        tuple of float
            ``(v_min, v_max)``.
        """
        v_min, v_max = float(self.values.min()), float(self.values.max())
        if self.fixed_layer is None:
            return v_min, v_max
        value = self.fixed_layer[1]
        return min(v_min, value), max(v_max, value)
