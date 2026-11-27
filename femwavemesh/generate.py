"""Mesh generation with gmsh.

:func:`generate_mesh` builds triangles of a :class:`~femwavemesh.RectangularDomain`
(spyro's 2D meshes); :func:`generate_mesh_3d` builds tetrahedra of a
:class:`~femwavemesh.BoxDomain` (the 3D Salt meshes). Both write gmsh format 2.2,
with the boundary markers of :mod:`femwavemesh.domain`, and run gmsh in the
calling process only: under MPI, call them on a single rank and share the
file name. Both also take a :class:`~femwavemesh.MeshSize`, a size on
triangles or tetrahedra that gmsh reads as a background view.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Protocol

import numpy as np

from . import _gmsh
from .audit import crossing_cells
from .domain import (BOTTOM, SEAFLOOR, TOP, X_MAX, X_MIN, Y_MAX, Y_MIN,
                     BoxDomain, FacetedSurface, RectangularDomain)
from .grids import VelocityGrid
from .io import TETRAHEDRON, zero_based
from .sizing import MeshSize


class PointSize(Protocol):
    """Mesh size that gmsh queries point by point (3D).

    Attributes
    ----------
    bounds : tuple of float
        Smallest and largest size, handed to gmsh as ``Mesh.MeshSizeMin``
        and ``Mesh.MeshSizeMax``.
    """

    bounds: tuple[float, float]

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
            Target edge length (km).
        """


def generate_mesh(
    path: str | Path,
    domain: RectangularDomain,
    size: VelocityGrid | MeshSize,
    conforming_depths: Sequence[float] = (),
    seed: int = 0,
    algorithm: int = 6,
) -> Path:
    """Generate a triangular mesh with gmsh and write it to a file.

    The first coordinate is spyro's depth ``z``, the second ``x``.

    Parameters
    ----------
    path : str or pathlib.Path
        Output ``.msh`` file (gmsh format 2.2).
    domain : RectangularDomain
        Domain to mesh.
    size : VelocityGrid or MeshSize
        Target edge length (km), either on a regular ``(z, x)`` grid
        covering the domain (gmsh's ``Structured`` background field) or at
        the vertices of triangles covering it, in ``(z, x)`` (gmsh's
        ``PostView`` background field; its bounds become
        ``Mesh.MeshSizeMin`` and ``Mesh.MeshSizeMax``). gmsh interpolates
        either linearly, without calling back into Python.
    conforming_depths : sequence of float, optional
        Depths of horizontal interfaces imposed as mesh edges.
    seed : int, optional
        Seed of gmsh's random perturbations, for replicas of the same design.
    algorithm : int, optional
        gmsh 2D algorithm. Default ``6`` (Frontal-Delaunay).

    Returns
    -------
    pathlib.Path
        The written mesh file.

    Raises
    ------
    ImportError
        If gmsh is not installed.
    ValueError
        If an interface depth is outside the domain, or a
        :class:`~femwavemesh.MeshSize` is not made of triangles covering it.
    """
    gmsh = _gmsh.load("generate adaptive meshes")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for depth in conforming_depths:
        if not domain.z_min < depth < domain.z_max:
            raise ValueError(f"Interface depth {depth} is outside the domain.")
    if isinstance(size, MeshSize):
        _check_covers(size, (domain.z_min, domain.x_min), (domain.z_max, domain.x_max))

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(path.stem)
        occ = gmsh.model.occ
        # gmsh's first coordinate is spyro's z, its second is x.
        surface = occ.addRectangle(domain.z_min, domain.x_min, 0.0,
                                   domain.z_max - domain.z_min,
                                   domain.x_max - domain.x_min)
        cuts = []
        for depth in conforming_depths:
            p1 = occ.addPoint(depth, domain.x_min, 0.0)
            p2 = occ.addPoint(depth, domain.x_max, 0.0)
            cuts.append((1, occ.addLine(p1, p2)))
        if cuts:
            occ.fragment([(2, surface)], cuts)
        occ.synchronize()
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]

        markers = {TOP: [], BOTTOM: [], X_MIN: [], X_MAX: []}
        boundary = gmsh.model.getBoundary([(2, s) for s in surfaces],
                                          oriented=False, combined=True)
        for dim, tag in boundary:
            z, x, _ = occ.getCenterOfMass(dim, abs(tag))
            if abs(z - domain.z_max) < 1e-9:
                markers[TOP].append(abs(tag))
            elif abs(z - domain.z_min) < 1e-9:
                markers[BOTTOM].append(abs(tag))
            elif abs(x - domain.x_min) < 1e-9:
                markers[X_MIN].append(abs(tag))
            else:
                markers[X_MAX].append(abs(tag))
        for marker, curves in markers.items():
            gmsh.model.addPhysicalGroup(1, curves, marker)
        gmsh.model.addPhysicalGroup(2, surfaces, 1)

        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)
        gmsh.option.setNumber("Mesh.Smoothing", 10)
        gmsh.option.setNumber("Mesh.RandomSeed", seed + 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        if isinstance(size, MeshSize):
            size_min, size_max = size.bounds
            gmsh.option.setNumber("Mesh.MeshSizeMin", size_min)
            gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
            _set_view_size(gmsh, size)
            gmsh.model.mesh.generate(2)
        else:
            with tempfile.TemporaryDirectory() as folder:
                field = gmsh.model.mesh.field.add("Structured")
                gmsh.model.mesh.field.setString(
                    field, "FileName", str(_write_structured_size(size, Path(folder))))
                gmsh.model.mesh.field.setNumber(field, "TextFormat", 1)
                gmsh.model.mesh.field.setAsBackgroundMesh(field)
                gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def _check_covers(size: MeshSize, lower: Sequence[float], upper: Sequence[float]) -> None:
    """Check that a mesh size has the domain's dimension and can cover it.

    Parameters
    ----------
    size : MeshSize
        The size.
    lower : sequence of float
        Smallest coordinates of the domain, in the order of ``size.points``.
    upper : sequence of float
        Largest coordinates of the domain, in the same order.

    Raises
    ------
    ValueError
        If the dimensions differ, the bounding box of the cells does not
        contain the domain or their measure is smaller than the domain's.
    """
    if size.dimension != len(lower):
        raise ValueError(f"A size on {'triangles' if size.dimension == 2 else 'tetrahedra'} "
                         f"cannot size a {len(lower)}D domain.")
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    low, high = size.extent()
    if np.any(low > lower + 1e-9) or np.any(high < upper - 1e-9) \
            or size.measure() < np.prod(upper - lower) * (1 - 1e-9):
        raise ValueError("The cells of the mesh size must cover the domain.")


def _set_view_size(gmsh: ModuleType, size: MeshSize) -> None:
    """Make a mesh size the background size of the current gmsh model.

    The size becomes a list-based view of scalar triangles (``ST``) or
    tetrahedra (``SS``), and the ``PostView`` field of that view the
    background mesh. In 2D, spyro's ``(z, x)`` are gmsh's first two
    coordinates, as in :func:`generate_mesh`, and the third is 0.

    Parameters
    ----------
    gmsh : types.ModuleType
        The initialized gmsh module.
    size : MeshSize
        The size.
    """
    corners = size.points[size.cells]
    if size.dimension == 2:
        corners = np.concatenate([corners, np.zeros(corners.shape[:2] + (1,))], axis=2)
    # Per cell: the x of its vertices, then their y, their z and the sizes.
    data = np.concatenate([corners[:, :, 0], corners[:, :, 1], corners[:, :, 2],
                           size.sizes[size.cells]], axis=1).ravel()
    view = gmsh.view.add("size")
    gmsh.view.addListData(view, "ST" if size.dimension == 2 else "SS", len(size.cells), data)
    field = gmsh.model.mesh.field.add("PostView")
    gmsh.model.mesh.field.setNumber(field, "ViewTag", view)
    gmsh.model.mesh.field.setAsBackgroundMesh(field)


def _write_structured_size(size: VelocityGrid, folder: Path) -> Path:
    """Write a size grid in the text format of gmsh's ``Structured`` field.

    The format is the origin, the spacings and the node counts in gmsh's
    ``(x, y, z)`` order, followed by the values with the last index varying
    fastest. gmsh's ``x`` and ``y`` are spyro's ``z`` and ``x``; the grid is
    given two identical layers in gmsh's ``z``, one on each side of the
    ``z = 0`` plane of the mesh.

    Parameters
    ----------
    size : VelocityGrid
        Target edge length on a regular ``(z, x)`` grid.
    folder : pathlib.Path
        Directory of the file.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    n_z, n_x = size.values.shape
    path = folder / "size.txt"
    with path.open("w") as stream:
        stream.write(f"{size.z0!r} {size.x0!r} -1.0\n")
        stream.write(f"{size.dz!r} {size.dx!r} 2.0\n")
        stream.write(f"{n_z} {n_x} 2\n")
        np.savetxt(stream, np.repeat(size.values.ravel(), 2))
    return path


def generate_mesh_3d(
    path: str | Path,
    domain: BoxDomain,
    size: PointSize | MeshSize,
    seafloor: FacetedSurface | None = None,
    threads: int = 1,
) -> Path:
    """Generate a tetrahedral mesh with gmsh and write it to a file.

    gmsh's HXT algorithm meshes the box (or the water and the rock of a box
    split by ``seafloor``) with the target edge length of ``size``, with
    random seed 0. The file is gmsh format 2.2, binary, written to a
    temporary name and renamed when complete.

    Parameters
    ----------
    path : str or pathlib.Path
        Output ``.msh`` file.
    domain : BoxDomain
        Box to mesh.
    size : PointSize or MeshSize
        Target edge length (km). A point size
        (:class:`~femwavemesh.UniformSize`, :class:`~femwavemesh.CPWSize`,
        :class:`~femwavemesh.ScatteredSize`) answers gmsh's size callback
        point by point; a :class:`~femwavemesh.MeshSize`, on tetrahedra
        covering the box, is gmsh's ``PostView`` background field. Their
        bounds become ``Mesh.MeshSizeMin`` and ``Mesh.MeshSizeMax``.
    seafloor : FacetedSurface, optional
        Surface imposed as mesh faces (marker :data:`~femwavemesh.SEAFLOOR`),
        spanning the box laterally. ``None`` (the default) meshes the box
        alone.
    threads : int, optional
        gmsh threads. Default ``1``.

    Returns
    -------
    pathlib.Path
        The written mesh file.

    Raises
    ------
    ImportError
        If gmsh is not installed.
    ValueError
        If the sea floor does not span the box or leaves it, or a
        :class:`~femwavemesh.MeshSize` is not made of tetrahedra covering
        the box.
    RuntimeError
        If gmsh made cells other than tetrahedra, or tetrahedra that cross
        the sea floor.
    """
    gmsh = _gmsh.load("generate meshes")
    path = Path(path)
    if seafloor is not None:
        _check_seafloor(domain, seafloor)
    if isinstance(size, MeshSize):
        _check_covers(size, (domain.x_min, domain.y_min, domain.z_min),
                      (domain.x_max, domain.y_max, domain.z_max))
    size_min, size_max = size.bounds
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", threads)
        gmsh.model.add(path.stem)
        if seafloor is None:
            _box(gmsh, domain)
        else:
            _box_with_seafloor(gmsh, domain, seafloor)
        for key, value in {"Mesh.MeshSizeFromPoints": 0, "Mesh.MeshSizeFromCurvature": 0,
                           "Mesh.MeshSizeExtendFromBoundary": 0, "Mesh.Algorithm3D": 10,
                           "Mesh.RandomSeed": 0, "Mesh.MshFileVersion": 2.2,
                           "Mesh.Binary": 1, "Mesh.MeshSizeMin": size_min,
                           "Mesh.MeshSizeMax": size_max}.items():
            gmsh.option.setNumber(key, value)
        if isinstance(size, MeshSize):
            _set_view_size(gmsh, size)
        else:
            gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: size(x, y, z))
        gmsh.model.mesh.generate(3)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.msh")
        gmsh.write(str(temporary))
        types, _, node_tags = gmsh.model.mesh.getElements(3)
        if set(types) != {TETRAHEDRON}:
            raise RuntimeError(f"Unexpected 3D elements: {types}")
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
    finally:
        gmsh.finalize()
    os.replace(temporary, path)
    if seafloor is not None:
        points, tetrahedra = zero_based(tags, coordinates, node_tags[0], 4, 3)
        crossing = crossing_cells(seafloor.side(points)[tetrahedra])
        if crossing:
            raise RuntimeError(f"{crossing} tetrahedra cross the sea floor: {path}")
    return path


def _check_seafloor(domain: BoxDomain, seafloor: FacetedSurface) -> None:
    """Check that a sea floor spans the box laterally and stays inside it.

    Parameters
    ----------
    domain : BoxDomain
        The box.
    seafloor : FacetedSurface
        The surface.

    Raises
    ------
    ValueError
        If the surface's grid does not end on the sides of the box or the
        surface touches its top or bottom.
    """
    ends = (seafloor.x[0], seafloor.x[-1], seafloor.y[0], seafloor.y[-1])
    sides = (domain.x_min, domain.x_max, domain.y_min, domain.y_max)
    if not np.allclose(ends, sides, rtol=0.0, atol=1e-9):
        raise ValueError("The sea floor must span the box laterally.")
    z = -np.asarray(seafloor.depth)
    if not np.all((z > domain.z_min) & (z < domain.z_max)):
        raise ValueError("The sea floor must lie strictly inside the box.")


def _box(gmsh: ModuleType, domain: BoxDomain) -> None:
    """Add the box and its boundary markers to the current gmsh model.

    Parameters
    ----------
    gmsh : types.ModuleType
        The initialized gmsh module.
    domain : BoxDomain
        The box.
    """
    occ = gmsh.model.occ
    box = occ.addBox(domain.x_min, domain.y_min, domain.z_min,
                     domain.x_max - domain.x_min, domain.y_max - domain.y_min,
                     domain.z_max - domain.z_min)
    occ.synchronize()
    markers = {}
    for dim, tag in gmsh.model.getBoundary([(3, box)], oriented=False):
        x, y, z = occ.getCenterOfMass(dim, tag)
        marker = (TOP if abs(z - domain.z_max) < 1e-8
                  else BOTTOM if abs(z - domain.z_min) < 1e-8
                  else X_MIN if abs(x - domain.x_min) < 1e-8
                  else X_MAX if abs(x - domain.x_max) < 1e-8
                  else Y_MIN if abs(y - domain.y_min) < 1e-8 else Y_MAX)
        markers.setdefault(marker, []).append(tag)
    for marker, faces in markers.items():
        gmsh.model.addPhysicalGroup(2, faces, marker)
    gmsh.model.addPhysicalGroup(3, [box], 1)


def _box_with_seafloor(gmsh: ModuleType, domain: BoxDomain,
                       seafloor: FacetedSurface) -> None:
    """Add the box split by a faceted sea floor (gmsh's geo kernel).

    The water and the rock are two volumes of the physical group ``1``; the
    facets form the group :data:`~femwavemesh.SEAFLOOR`, and each side of the box
    is made of a water face and a rock face.

    Parameters
    ----------
    gmsh : types.ModuleType
        The initialized gmsh module.
    domain : BoxDomain
        The box.
    seafloor : FacetedSurface
        The sea floor, spanning the box laterally.
    """
    geo = gmsh.model.geo
    nx, ny = len(seafloor.x), len(seafloor.y)
    S = [[geo.addPoint(seafloor.x[i], seafloor.y[j], -seafloor.depth[j, i])
          for i in range(nx)] for j in range(ny)]
    corners = {(0, 0): (domain.x_min, domain.y_min),
               (nx - 1, 0): (domain.x_max, domain.y_min),
               (nx - 1, ny - 1): (domain.x_max, domain.y_max),
               (0, ny - 1): (domain.x_min, domain.y_max)}
    T = {k: geo.addPoint(x, y, domain.z_max) for k, (x, y) in corners.items()}
    B = {k: geo.addPoint(x, y, domain.z_min) for k, (x, y) in corners.items()}
    lines = {}

    def line(a: int, b: int) -> int:
        """Line from point ``a`` to point ``b``, reused (signed) if it exists.

        Parameters
        ----------
        a : int
            Tag of the first point.
        b : int
            Tag of the second point.

        Returns
        -------
        int
            Signed tag of the line.
        """
        if (a, b) in lines:
            return lines[(a, b)]
        if (b, a) in lines:
            return -lines[(b, a)]
        lines[(a, b)] = geo.addLine(a, b)
        return lines[(a, b)]

    def face(points: list[int]) -> int:
        """Plane face bounded by the closed polygon ``points``.

        Parameters
        ----------
        points : list of int
            Point tags, in order.

        Returns
        -------
        int
            Tag of the face.
        """
        loop = geo.addCurveLoop([line(a, b) for a, b in zip(points, points[1:] + points[:1])])
        return geo.addPlaneSurface([loop])

    floor = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            floor.append(face([S[j][i], S[j][i + 1], S[j + 1][i + 1]]))
            floor.append(face([S[j][i], S[j + 1][i + 1], S[j + 1][i]]))
    # Sea-floor polylines on each side, from its first corner to its last
    # (anticlockwise seen from above).
    side_ymin = [S[0][i] for i in range(nx)]
    side_xmax = [S[j][nx - 1] for j in range(ny)]
    side_ymax = [S[ny - 1][i] for i in reversed(range(nx))]
    side_xmin = [S[j][0] for j in reversed(range(ny))]
    sides = [((0, 0), (nx - 1, 0), side_ymin, Y_MIN),
             ((nx - 1, 0), (nx - 1, ny - 1), side_xmax, X_MAX),
             ((nx - 1, ny - 1), (0, ny - 1), side_ymax, Y_MAX),
             ((0, ny - 1), (0, 0), side_xmin, X_MIN)]
    top = face([T[(0, 0)], T[(nx - 1, 0)], T[(nx - 1, ny - 1)], T[(0, ny - 1)]])
    bottom = face([B[(0, 0)], B[(nx - 1, 0)], B[(nx - 1, ny - 1)], B[(0, ny - 1)]])
    water_sides, rock_sides, markers = [], [], {}
    for c0, c1, polyline, marker in sides:
        water = face([T[c0], T[c1], *reversed(polyline)])  # top c0->c1, down, back on the floor
        rock = face([*polyline, B[c1], B[c0]])              # floor c0->c1, down, back on the bottom
        water_sides.append(water)
        rock_sides.append(rock)
        markers.setdefault(marker, []).extend([water, rock])
    water_volume = geo.addVolume([geo.addSurfaceLoop([top, *floor, *water_sides])])
    rock_volume = geo.addVolume([geo.addSurfaceLoop([bottom, *floor, *rock_sides])])
    geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [top], TOP)
    gmsh.model.addPhysicalGroup(2, [bottom], BOTTOM)
    for marker, faces in markers.items():
        gmsh.model.addPhysicalGroup(2, faces, marker)
    gmsh.model.addPhysicalGroup(2, floor, SEAFLOOR)
    gmsh.model.addPhysicalGroup(3, [water_volume, rock_volume], 1)
