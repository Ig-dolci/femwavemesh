"""Domains and conforming interfaces.

Two coordinate conventions are kept, those of the codes the meshes feed:

* 2D (spyro): the first coordinate is the depth ``z`` (negative downwards,
  ``z = 0`` at the surface) and the second is ``x``;
* 3D: ``(x, y, z)`` with ``z`` negative downwards.

Boundary markers, as physical groups of the mesh:

``1`` (:data:`TOP`)
    top, ``z = z_max``;
``2`` (:data:`BOTTOM`)
    bottom, ``z = z_min``;
``3`` (:data:`X_MIN`), ``4`` (:data:`X_MAX`)
    ``x = x_min`` and ``x = x_max``;
``5`` (:data:`Y_MIN`), ``6`` (:data:`Y_MAX`)
    ``y = y_min`` and ``y = y_max`` (3D);
``7`` (:data:`SEAFLOOR`)
    faces of a conforming sea floor (3D).

The cells of every mesh form the physical group ``1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Boundary markers.
TOP, BOTTOM, X_MIN, X_MAX, Y_MIN, Y_MAX, SEAFLOOR = 1, 2, 3, 4, 5, 6, 7


@dataclass(frozen=True)
class RectangularDomain:
    """Padded rectangular domain in spyro's ``(z, x)`` coordinates.

    The physical region of interest is
    ``[z_min + pad, z_max] x [x_min + pad, x_max - pad]``: the pad is the
    absorbing layer at the bottom and on both sides. The top is not padded.

    Parameters
    ----------
    z_min : float
        Depth of the bottom of the padded domain (negative, km).
    x_min : float
        Left end of the padded domain (km).
    x_max : float
        Right end of the padded domain (km).
    z_max : float, optional
        Top of the domain (km). Default is ``0``.
    pad : float, optional
        Width of the absorbing pad (km). Default is ``0``.
    """

    z_min: float
    x_min: float
    x_max: float
    z_max: float = 0.0
    pad: float = 0.0

    def __post_init__(self) -> None:
        """Validate the extents.

        Raises
        ------
        ValueError
            If the domain is empty or the pad leaves no region of interest.
        """
        if not (self.z_min < self.z_max and self.x_min < self.x_max):
            raise ValueError("The domain must have a positive extent.")
        if self.pad < 0 or 2 * self.pad >= self.x_max - self.x_min \
                or self.pad >= self.z_max - self.z_min:
            raise ValueError("The pad must be non-negative and leave an interior.")

    def in_interior(self, z: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Tell which points lie in the region of interest (outside the pad).

        Parameters
        ----------
        z : numpy.ndarray
            Depths of the points.
        x : numpy.ndarray
            Horizontal positions of the points.

        Returns
        -------
        numpy.ndarray
            Boolean array, ``True`` inside the region of interest.
        """
        z = np.asarray(z)
        x = np.asarray(x)
        return ((z >= self.z_min + self.pad) & (x >= self.x_min + self.pad)
                & (x <= self.x_max - self.pad))

    def grid(self, spacing: float) -> tuple[np.ndarray, np.ndarray]:
        """Regular grid covering the padded domain.

        Parameters
        ----------
        spacing : float
            Grid spacing (km).

        Returns
        -------
        tuple of numpy.ndarray
            One-dimensional ``z`` and ``x`` coordinates, both increasing.
        """
        z = np.arange(self.z_min, self.z_max + spacing / 2, spacing)
        x = np.arange(self.x_min, self.x_max + spacing / 2, spacing)
        return z, x


@dataclass(frozen=True)
class BoxDomain:
    """Padded box in ``(x, y, z)`` coordinates, ``z`` negative downwards.

    The physical region of interest is ``[x_min + pad, x_max - pad] x
    [y_min + pad, y_max - pad] x [z_min + pad, z_max]``: the pad is the
    absorbing layer on the four sides and at the bottom. The top is not
    padded.

    Parameters
    ----------
    x_min : float
        Smallest ``x`` of the padded box (km).
    x_max : float
        Largest ``x`` of the padded box (km).
    y_min : float
        Smallest ``y`` of the padded box (km).
    y_max : float
        Largest ``y`` of the padded box (km).
    z_min : float
        Bottom of the padded box (negative, km).
    z_max : float, optional
        Top of the box (km). Default is ``0``.
    pad : float, optional
        Width of the absorbing pad (km). Default is ``0``.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float = 0.0
    pad: float = 0.0

    def __post_init__(self) -> None:
        """Validate the extents.

        Raises
        ------
        ValueError
            If the box is empty or the pad leaves no region of interest.
        """
        if not (self.x_min < self.x_max and self.y_min < self.y_max
                and self.z_min < self.z_max):
            raise ValueError("The box must have a positive extent.")
        if self.pad < 0 or 2 * self.pad >= self.x_max - self.x_min \
                or 2 * self.pad >= self.y_max - self.y_min \
                or self.pad >= self.z_max - self.z_min:
            raise ValueError("The pad must be non-negative and leave an interior.")

    def in_interior(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Tell which points lie in the region of interest (outside the pad).

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
            Boolean array, ``True`` inside the region of interest.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        return ((z >= self.z_min + self.pad)
                & (x >= self.x_min + self.pad) & (x <= self.x_max - self.pad)
                & (y >= self.y_min + self.pad) & (y <= self.y_max - self.pad))


@dataclass(frozen=True)
class FacetedSurface:
    """Surface ``depth(x, y)`` made of planar triangles, such as a sea floor.

    Each cell of the regular ``(x, y)`` grid is split into two triangles by
    the diagonal from node ``(i, j)`` to node ``(i + 1, j + 1)``. A mesh made
    by :func:`~femwavemesh.generate_mesh_3d` uses these triangles as faces, and a
    velocity model that switches at :meth:`depth_at` has its jump exactly on
    mesh faces.

    Parameters
    ----------
    x : numpy.ndarray
        Increasing, equally spaced ``x`` of the nodes (km), shape ``(nx,)``.
    y : numpy.ndarray
        Increasing, equally spaced ``y`` of the nodes (km), shape ``(ny,)``.
    depth : numpy.ndarray
        Depth of the surface at the nodes (positive, km), shape ``(ny, nx)``.
    """

    x: np.ndarray
    y: np.ndarray
    depth: np.ndarray

    def __post_init__(self) -> None:
        """Validate the node arrays.

        Raises
        ------
        ValueError
            If the grid has fewer than 2 x 2 nodes or the depths do not
            match it.
        """
        if len(self.x) < 2 or len(self.y) < 2 \
                or np.shape(self.depth) != (len(self.y), len(self.x)):
            raise ValueError("The surface needs depth[j, i] on at least 2 x 2 nodes.")

    @classmethod
    def from_water_depth(cls, water_depth: np.ndarray, x: np.ndarray,
                         y: np.ndarray, domain: BoxDomain, sigma: float,
                         spacing: float) -> FacetedSurface:
        """Smooth a gridded water depth and resample it on the facet nodes.

        The water depth is smoothed by a Gaussian of standard deviation
        ``sigma`` and sampled (bilinearly) on a regular grid of step
        ``spacing`` spanning the box laterally; nodes outside the data are
        taken at its closest edge.

        Parameters
        ----------
        water_depth : numpy.ndarray
            Depth of the sea floor (positive, km), shape ``(len(y), len(x))``.
        x : numpy.ndarray
            Equally spaced ``x`` of the data (km).
        y : numpy.ndarray
            Equally spaced ``y`` of the data (km).
        domain : BoxDomain
            Box whose lateral extent the facets span.
        sigma : float
            Standard deviation of the smoothing (km).
        spacing : float
            Side of the facet cells (km).

        Returns
        -------
        FacetedSurface
            The surface.
        """
        from scipy.interpolate import RegularGridInterpolator
        from scipy.ndimage import gaussian_filter

        water_depth = np.asarray(water_depth, dtype=float)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        smooth = gaussian_filter(water_depth, sigma=sigma / float(x[1] - x[0]),
                                 mode="nearest")
        interpolator = RegularGridInterpolator((y, x), smooth)
        nx = int(round((domain.x_max - domain.x_min) / spacing)) + 1
        ny = int(round((domain.y_max - domain.y_min) / spacing)) + 1
        nodes_x = np.linspace(domain.x_min, domain.x_max, nx)
        nodes_y = np.linspace(domain.y_min, domain.y_max, ny)
        Y, X = np.meshgrid(nodes_y, nodes_x, indexing="ij")
        points = np.column_stack([np.clip(Y.ravel(), y[0], y[-1]),
                                  np.clip(X.ravel(), x[0], x[-1])])
        return cls(nodes_x, nodes_y, interpolator(points).reshape(ny, nx))

    def depth_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Depth of the surface, planar on each triangle.

        Parameters
        ----------
        x : numpy.ndarray
            ``x`` of the points (km), inside the surface's grid.
        y : numpy.ndarray
            ``y`` of the points (km), broadcastable with ``x``.

        Returns
        -------
        numpy.ndarray
            Depth of the surface (positive, km).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        dx, dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
        fi = np.clip((x - self.x[0]) / dx, 0.0, len(self.x) - 1 - 1e-12)
        fj = np.clip((y - self.y[0]) / dy, 0.0, len(self.y) - 1 - 1e-12)
        i = np.minimum(fi.astype(int), len(self.x) - 2)
        j = np.minimum(fj.astype(int), len(self.y) - 2)
        u, v = fi - i, fj - j
        d00, d10 = self.depth[j, i], self.depth[j, i + 1]
        d01, d11 = self.depth[j + 1, i], self.depth[j + 1, i + 1]
        lower = d00 + u * (d10 - d00) + v * (d11 - d10)  # (0,0), (1,0), (1,1)
        upper = d00 + v * (d01 - d00) + u * (d11 - d01)  # (0,0), (1,1), (0,1)
        return np.where(u >= v, lower, upper)

    def side(self, xyz: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
        """Side of the surface each point lies on.

        Parameters
        ----------
        xyz : numpy.ndarray
            Points ``(x, y, z)``, shape ``(n, 3)``, ``z`` negative downwards.
        tolerance : float, optional
            Distance (km, along ``z``) within which a point is on the surface.
            Default ``1e-9``.

        Returns
        -------
        numpy.ndarray
            ``+1`` below the surface, ``-1`` above it, ``0`` on it.
        """
        below = -xyz[:, 2] - self.depth_at(xyz[:, 0], xyz[:, 1])
        return np.where(below > tolerance, 1,
                        np.where(below < -tolerance, -1, 0))
