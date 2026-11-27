r"""Mesh-size laws.

Uniform cells per wavelength (CPW), :func:`cpw_size`, sets the target edge
length :math:`h = v / (C f)` from the local velocity.

The adaptive law (:class:`SizeLaw`) redistributes a budget of degrees of
freedom according to a refinement density :math:`\rho` given by the caller:

.. math::

    h = \operatorname{clip}\left(\alpha\, h_k^{q/(q+d)}
        \rho^{-1/(q+d)},\ h_{\min},\ h_{\max}\right),

where :math:`d` is the dimension, :math:`h_k` the local size of the current
mesh, :math:`h_{\max}` its ceiling (by default :math:`h_k`, so a new mesh is
never coarser than the current one), :math:`q` the convergence order of the
error and :math:`\alpha` the scale that gives the requested number of degrees
of freedom (:func:`fit_to_budget`). The floor
:math:`h_{\min} = v / (c_{\max} C_{eq} f)` keeps the time step of an explicit
scheme from collapsing, with :math:`C_{eq}` the cells per wavelength of a
uniform-CPW mesh with the same budget.

Everything on a grid lives on a regular grid with one spacing, indexed like
the caller's arrays. In 3D, gmsh asks for the size point by point:
:class:`UniformSize`, :class:`CPWSize` and :class:`ScatteredSize` answer it.
:class:`MeshSize`, a size at the vertices of triangles or tetrahedra, is
handed to gmsh whole instead, and gmsh interpolates it without calling back
into Python.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import reduce

import numpy as np
from scipy.ndimage import gaussian_filter

from .grids import VelocityGrid, VelocityGrid3D


def cpw_size(velocity: float | np.ndarray, cpw: float,
             frequency: float) -> float | np.ndarray:
    """Edge length of a uniform cells-per-wavelength mesh.

    Parameters
    ----------
    velocity : float or numpy.ndarray
        Local velocity (km/s).
    cpw : float
        Cells per wavelength :math:`C`.
    frequency : float
        Frequency :math:`f` (Hz).

    Returns
    -------
    float or numpy.ndarray
        :math:`h = v / (C f)` (km).
    """
    return velocity / (cpw * frequency)


def size_perturbation(shape: tuple[int, ...], spacing: float, seed: int,
                      amplitude: float = 0.03, correlation: float = 0.1
                      ) -> np.ndarray:
    r"""Smooth random factor on a mesh size, for mesh replicas.

    gmsh's Frontal-Delaunay algorithm is deterministic: one size field gives
    one mesh whatever ``Mesh.RandomSeed`` is. A replica ``seed > 0``
    therefore multiplies the size by :math:`1 + A\,\xi_s(x)`, with
    :math:`\xi_s` a Gaussian field smoothed over ``correlation`` and
    normalized to zero mean and unit standard deviation. Seed 0 is not
    perturbed. Every strategy of one replica uses the same field.

    Parameters
    ----------
    shape : tuple of int
        Grid shape, such as ``(n_z, n_x)``.
    spacing : float
        Grid spacing (km).
    seed : int
        Replica; 0 returns ones.
    amplitude : float, optional
        Standard deviation :math:`A` of the factor. Default ``0.03``.
    correlation : float, optional
        Standard deviation of the Gaussian smoothing (km). Default ``0.1``.

    Returns
    -------
    numpy.ndarray
        Factor of shape ``shape``.
    """
    if seed == 0 or amplitude == 0.0:
        return np.ones(shape)
    rng = np.random.default_rng(1000 + seed)
    xi = gaussian_filter(rng.standard_normal(shape), correlation / spacing,
                         mode="wrap")
    return 1.0 + amplitude * (xi - xi.mean()) / xi.std()


def source_refinement(axes: Sequence[np.ndarray], sources: Sequence[Sequence[float]],
                      radius: float, factor: float, taper: float) -> np.ndarray:
    """Multiplier of a mesh size that refines it around point sources.

    The size is divided by ``factor`` within ``radius`` of the nearest
    source and goes back to itself log-linearly over ``taper``.

    Parameters
    ----------
    axes : sequence of numpy.ndarray
        One-dimensional coordinates of the grid, in the order of its
        indices, e.g. ``(z, x)`` for spyro's 2D grids.
    sources : sequence of sequence of float
        Source positions, in the coordinates of ``axes``.
    radius : float
        Radius (km) of full refinement around each source.
    factor : float
        Size divisor inside the radius (``>= 1``).
    taper : float
        Width (km) of the log-linear transition back to 1; ``0`` makes it a
        step.

    Returns
    -------
    numpy.ndarray
        ``1/factor`` within ``radius``, ``1`` beyond ``radius + taper``, of
        shape ``(len(a) for a in axes)``.
    """
    grid = np.meshgrid(*axes, indexing="ij")
    distance = np.min([reduce(np.hypot, [g - s for g, s in zip(grid, source)])
                       for source in sources], axis=0)
    t = np.clip((distance - radius) / taper, 0.0, 1.0) if taper > 0 else \
        (distance > radius).astype(float)
    return factor ** (-(1.0 - t))


def dof_calibration(dofs: int, size: np.ndarray, spacing: float) -> float:
    r"""Degrees of freedom per unit of :math:`\int h^{-d}`, for a mesh.

    Parameters
    ----------
    dofs : int
        Degrees of freedom of the mesh.
    size : numpy.ndarray
        Its local size on a grid of dimension ``d = size.ndim``.
    spacing : float
        Grid spacing.

    Returns
    -------
    float
        ``dofs`` divided by :math:`\sum h^{-d} \Delta^d`, so that the
        budget of a size field ``g`` is predicted as the calibration times
        :meth:`SizeLaw.relative_cost`.
    """
    d = size.ndim
    return dofs / float((size ** -d).sum() * spacing ** d)


class SizeLaw:
    r"""Adaptive mesh-size law of a refinement density, in any dimension.

    Parameters
    ----------
    density : numpy.ndarray
        Refinement density on the grid; its dimension is the mesh's.
    current_size : numpy.ndarray
        Local size of the current mesh on the grid: the :math:`h_k` of the
        law and, unless ``ceiling`` is given, its upper bound.
    spacing : float
        Grid spacing.
    order : float, optional
        Convergence order :math:`q` of the error. Default is ``2.7``.
    ceiling : numpy.ndarray, optional
        Upper bound of the size on the grid, replacing ``current_size`` in
        the clip of the law (not in its shape). ``None`` (the default) keeps
        ``current_size``, so the new mesh is never coarser than the current
        one; a redesign that may coarsen gives its own bound here.

    Attributes
    ----------
    dimension : int
        Dimension ``d`` of the grid.
    shape : numpy.ndarray
        :math:`h_k^{q/(q+d)} \rho^{-1/(q+d)}`, the size for :math:`\alpha = 1`
        before the clip.
    h_min : numpy.ndarray
        Lower bound of the size, set by :meth:`set_floor` (zero before).
    h_max : numpy.ndarray
        Upper bound of the size.
    h : numpy.ndarray or None
        Size on the grid for the last :meth:`set_scale`; ``None`` before.

    Raises
    ------
    ValueError
        If the arrays do not share one shape.
    """

    def __init__(self, density: np.ndarray, current_size: np.ndarray,
                 spacing: float, order: float = 2.7,
                 ceiling: np.ndarray | None = None) -> None:
        if density.shape != current_size.shape:
            raise ValueError("Density and current size must have matching shapes.")
        if ceiling is not None and ceiling.shape != density.shape:
            raise ValueError("The ceiling must have the shape of the grid.")
        rho = np.maximum(density, 1e-6 * max(float(density.max()), 1e-300))
        self.dimension = density.ndim
        self.spacing = float(spacing)
        current_size = np.asarray(current_size, dtype=float)
        self.h_max = (current_size if ceiling is None
                      else np.asarray(ceiling, dtype=float))
        self.h_min = np.zeros_like(self.h_max)
        d = self.dimension
        self.shape = current_size ** (order / (order + d)) * rho ** (-1.0 / (order + d))
        self.h = None

    def set_floor(self, velocity: np.ndarray, frequency: float,
                  equivalent_cpw: float, cmax_ratio: float = 1.25) -> None:
        r"""Set :math:`h_{\min} = v / (c_{\max} C_{eq} f)`, capped by the ceiling.

        Parameters
        ----------
        velocity : numpy.ndarray
            Velocity on the grid.
        frequency : float
            Peak frequency of the source (Hz).
        equivalent_cpw : float
            Cells per wavelength of a uniform-CPW mesh with the target budget.
        cmax_ratio : float, optional
            Largest refinement relative to that mesh. Default ``1.25``.

        Returns
        -------
        None
            The floor is stored in :attr:`h_min`.
        """
        self.h_min = np.minimum(velocity / (cmax_ratio * equivalent_cpw * frequency),
                                self.h_max)

    def size(self, scale: float) -> np.ndarray:
        r"""Size on the grid for a given scale :math:`\alpha`.

        Parameters
        ----------
        scale : float
            The factor :math:`\alpha` of the law.

        Returns
        -------
        numpy.ndarray
            Clipped size on the grid.
        """
        return np.clip(scale * self.shape, self.h_min, self.h_max)

    def relative_cost(self, scale: float) -> float:
        r"""Integral of :math:`h^{-d}`, proportional to the number of DOFs.

        Parameters
        ----------
        scale : float
            The factor :math:`\alpha` of the law.

        Returns
        -------
        float
            :math:`\sum h^{-d} \Delta^d` over the grid.
        """
        d = self.dimension
        return float((self.size(scale) ** -d).sum() * self.spacing**d)

    def scale_for(self, target_cost: float) -> float:
        """Find the scale whose :meth:`relative_cost` matches a target.

        Parameters
        ----------
        target_cost : float
            Desired value of :meth:`relative_cost`.

        Returns
        -------
        float
            The scale, found by bisection in log space. Targets outside the
            reachable range return the closest bound.
        """
        lo, hi = 1e-8, 1e8
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if self.relative_cost(mid) > target_cost:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    def set_scale(self, scale: float) -> None:
        r"""Fix the scale of the size handed to the mesh generator.

        Parameters
        ----------
        scale : float
            The factor :math:`\alpha` of the law.

        Returns
        -------
        None
            The size is stored in :attr:`h`.
        """
        self.h = self.size(scale)


class AdaptiveSizeField(SizeLaw):
    """The adaptive law on spyro's 2D ``(z, x)`` grid.

    Parameters
    ----------
    density : numpy.ndarray
        Refinement density on the grid, shape ``(nz, nx)``.
    current_size : numpy.ndarray
        Local size of the current mesh on the grid; the new mesh is never
        coarser unless ``ceiling`` says otherwise.
    z : numpy.ndarray
        Grid depths, equally spaced.
    x : numpy.ndarray
        Grid positions, with the same spacing.
    order : float, optional
        Convergence order :math:`q` of the error. Default is ``2.7``.
    ceiling : numpy.ndarray, optional
        Upper bound of the size on the grid (see :class:`SizeLaw`).

    Attributes
    ----------
    grid : VelocityGrid or None
        The size of the last :meth:`set_scale` as a grid that
        :func:`~femwavemesh.generate_mesh` accepts; ``None`` before.

    Raises
    ------
    ValueError
        If the arrays do not match the grid.
    """

    def __init__(self, density: np.ndarray, current_size: np.ndarray,
                 z: np.ndarray, x: np.ndarray, order: float = 2.7,
                 ceiling: np.ndarray | None = None) -> None:
        if density.shape != current_size.shape or density.shape != (len(z), len(x)):
            raise ValueError("Density, current size and grid must have matching shapes.")
        self.z = np.asarray(z)
        self.x = np.asarray(x)
        super().__init__(density, current_size, float(self.z[1] - self.z[0]),
                         order, ceiling)
        self.grid = None

    def set_scale(self, scale: float) -> None:
        r"""Fix the scale of the size handed to the mesh generator.

        Parameters
        ----------
        scale : float
            The factor :math:`\alpha` of the law.

        Returns
        -------
        None
            The size is stored in :attr:`h` and :attr:`grid`.
        """
        super().set_scale(scale)
        self.grid = VelocityGrid(self.h, self.z[0], self.x[0],
                                 self.spacing, self.spacing)


def fit_to_budget(law: SizeLaw, target: int, calibration: float,
                  build: Callable[[SizeLaw], int], tolerance: float = 0.02,
                  attempts: int = 6) -> int:
    """Rescale a size law until the mesh it gives meets a DOF budget.

    The first scale is the one whose predicted budget,
    ``calibration * law.relative_cost(scale)``, is ``target``. While the
    mesh misses the target by more than ``tolerance``, the scale is
    multiplied by :math:`(N/N_{target})^{1/d}`. The last mesh is kept when
    no attempt is within the tolerance.

    Parameters
    ----------
    law : SizeLaw
        The law, with its floor already set.
    target : int
        Target number of degrees of freedom.
    calibration : float
        Degrees of freedom per unit of relative cost
        (:func:`dof_calibration` of the current mesh).
    build : callable
        ``build(law)`` generates the mesh of ``law.h`` (or of
        :attr:`AdaptiveSizeField.grid`) and returns its degrees of freedom.
    tolerance : float, optional
        Relative tolerance on the budget. Default ``0.02``.
    attempts : int, optional
        Largest number of meshes. Default ``6``.

    Returns
    -------
    int
        Degrees of freedom of the last mesh built.
    """
    scale = law.scale_for(target / calibration)
    dofs = 0
    for _ in range(attempts):
        law.set_scale(scale)
        dofs = build(law)
        if abs(dofs / target - 1.0) <= tolerance:
            break
        ratio = dofs / target
        # math.sqrt keeps the exact arithmetic of spyro's 2D loop.
        scale *= math.sqrt(ratio) if law.dimension == 2 else ratio ** (1.0 / law.dimension)
    return dofs


class UniformSize:
    """Constant mesh size, for gmsh's size callback.

    Parameters
    ----------
    size : float
        Edge length (km).
    """

    def __init__(self, size: float) -> None:
        self.size = float(size)

    def __call__(self, x: float, y: float, z: float) -> float:
        """Size at a point.

        Parameters
        ----------
        x : float
            ``x`` of the point.
        y : float
            ``y`` of the point.
        z : float
            ``z`` of the point.

        Returns
        -------
        float
            The constant size.
        """
        return self.size

    @property
    def bounds(self) -> tuple[float, float]:
        """Smallest and largest size.

        Returns
        -------
        tuple of float
            ``(size, size)``.
        """
        return self.size, self.size


class CPWSize:
    """Uniform cells per wavelength in 3D, :math:`h = v/(C f)` at each point.

    Parameters
    ----------
    velocity : VelocityGrid3D
        Velocity model.
    cpw : float
        Cells per wavelength :math:`C`.
    frequency : float
        Frequency :math:`f` (Hz).
    """

    def __init__(self, velocity: VelocityGrid3D, cpw: float, frequency: float) -> None:
        self.velocity = velocity
        self.cpw = float(cpw)
        self.frequency = float(frequency)

    def __call__(self, x: float, y: float, z: float) -> float:
        """Size at a point.

        Parameters
        ----------
        x : float
            ``x`` of the point.
        y : float
            ``y`` of the point.
        z : float
            ``z`` of the point.

        Returns
        -------
        float
            :math:`v(x, y, z) / (C f)`.
        """
        v = float(self.velocity(np.array([x]), np.array([y]), np.array([z]))[0])
        return cpw_size(v, self.cpw, self.frequency)

    @property
    def bounds(self) -> tuple[float, float]:
        """Smallest and largest size, from the extremes of the velocity.

        Returns
        -------
        tuple of float
            ``(v_min / (C f), v_max / (C f))``.
        """
        v_min, v_max = self.velocity.extremes()
        return (cpw_size(v_min, self.cpw, self.frequency),
                cpw_size(v_max, self.cpw, self.frequency))


class ScatteredSize:
    r"""Mesh size given at scattered points, such as cell centroids.

    Between the points the size is the weighted mean of :math:`\log h` of
    the four nearest points, with weights :math:`1/r^2`; at a point it is
    its own value.

    Parameters
    ----------
    points : numpy.ndarray
        Points ``(x, y, z)``, shape ``(n, 3)``.
    sizes : numpy.ndarray
        Size at each point (km), positive, shape ``(n,)``.

    Raises
    ------
    ValueError
        If the arrays have the wrong shape or a size is not finite and
        positive.
    """

    def __init__(self, points: np.ndarray, sizes: np.ndarray) -> None:
        from scipy.spatial import cKDTree

        points = np.asarray(points, float)
        sizes = np.asarray(sizes, float)
        if points.ndim != 2 or points.shape[1] != 3 or sizes.shape != (len(points),) \
                or not (np.isfinite(sizes).all() and (sizes > 0).all()):
            raise ValueError("Expected points (n, 3) and n finite positive sizes.")
        self.sizes = sizes
        self._tree = cKDTree(points)
        self._log_sizes = np.log(sizes)
        self._neighbours = min(4, len(points))

    def __call__(self, x: float, y: float, z: float) -> float:
        """Size at a point.

        Parameters
        ----------
        x : float
            ``x`` of the point.
        y : float
            ``y`` of the point.
        z : float
            ``z`` of the point.

        Returns
        -------
        float
            Interpolated size.
        """
        distance, index = self._tree.query([x, y, z], k=self._neighbours)
        distance, index = np.atleast_1d(distance), np.atleast_1d(index)
        if distance[0] < 1e-12:
            return float(self.sizes[index[0]])
        weight = 1.0 / np.maximum(distance, 1e-12) ** 2
        return float(np.exp(np.dot(weight, self._log_sizes[index]) / weight.sum()))

    @property
    def bounds(self) -> tuple[float, float]:
        """Smallest and largest size.

        Returns
        -------
        tuple of float
            ``(min(sizes), max(sizes))``.
        """
        return float(self.sizes.min()), float(self.sizes.max())


class MeshSize:
    r"""Mesh size at the vertices of a simplicial mesh, linear on each cell.

    :func:`~femwavemesh.generate_mesh` and :func:`~femwavemesh.generate_mesh_3d`
    hand it to gmsh as a post-processing view, the background field
    ``PostView``, which gmsh interpolates in its own code. In 3D this avoids
    the Python call per point of :class:`ScatteredSize` and :class:`CPWSize`
    (the interpolation differs: linear on the cells here, a mean of
    :math:`\log h` over four neighbours there). In 2D it takes a size defined
    on a triangulation, such as the current mesh of an adaptive loop, without
    resampling it on a grid.

    The cells must cover the domain to mesh. The generators check that the
    bounding box of their vertices contains the domain and that their total
    area or volume is at least the domain's, not each point.

    Parameters
    ----------
    points : numpy.ndarray
        Vertices: shape ``(n, 2)`` in spyro's ``(z, x)`` for triangles,
        ``(n, 3)`` in ``(x, y, z)`` for tetrahedra.
    cells : numpy.ndarray
        Zero-based vertex indices of the triangles, shape ``(m, 3)``, or of
        the tetrahedra, shape ``(m, 4)``.
    sizes : numpy.ndarray
        Size at each vertex (km), finite and positive, shape ``(n,)``.

    Attributes
    ----------
    dimension : int
        2 for triangles, 3 for tetrahedra.

    Raises
    ------
    ValueError
        If the arrays have the wrong shapes, a cell refers to a missing
        vertex or a size is not finite and positive.
    """

    def __init__(self, points: np.ndarray, cells: np.ndarray, sizes: np.ndarray) -> None:
        points = np.asarray(points, float)
        cells = np.asarray(cells)
        sizes = np.asarray(sizes, float)
        if points.ndim != 2 or points.shape[1] not in (2, 3) or cells.ndim != 2 \
                or cells.shape[1] != points.shape[1] + 1 or len(cells) == 0:
            raise ValueError("Expected points (n, 2) with triangles (m, 3), "
                             "or points (n, 3) with tetrahedra (m, 4).")
        if not np.issubdtype(cells.dtype, np.integer) or cells.min() < 0 \
                or cells.max() >= len(points):
            raise ValueError("The cells must hold indices of the points.")
        if sizes.shape != (len(points),) or not (np.isfinite(sizes).all() and (sizes > 0).all()):
            raise ValueError("Expected one finite positive size per point.")
        self.points = points
        self.cells = cells
        self.sizes = sizes
        self.dimension = points.shape[1]

    @property
    def bounds(self) -> tuple[float, float]:
        """Smallest and largest size.

        Returns
        -------
        tuple of float
            ``(min(sizes), max(sizes))``.
        """
        return float(self.sizes.min()), float(self.sizes.max())

    def extent(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounding box of the vertices of the cells.

        Returns
        -------
        lower : numpy.ndarray
            Smallest coordinates, one per axis.
        upper : numpy.ndarray
            Largest coordinates, one per axis.
        """
        used = self.points[np.unique(self.cells)]
        return used.min(axis=0), used.max(axis=0)

    def measure(self) -> float:
        """Total area of the triangles, or volume of the tetrahedra.

        Returns
        -------
        float
            Sum of the cell measures.
        """
        corners = self.points[self.cells]
        edges = corners[:, 1:] - corners[:, :1]
        return float(np.abs(np.linalg.det(edges)).sum() / math.factorial(self.dimension))
