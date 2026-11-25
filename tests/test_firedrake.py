"""The meshes as Firedrake reads them: KMV spaces, markers and interfaces.

kmvmesh does not depend on Firedrake; these tests check that what it
promises (DOF counts, boundary markers, conforming interfaces) is what the
solver sees once it loads the ``.msh`` file. They are skipped where Firedrake
is not installed.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import kmvmesh
from kmvmesh import (BOTTOM, SEAFLOOR, TOP, X_MAX, X_MIN, Y_MAX, Y_MIN,
                     FacetedSurface, UniformSize)

from .test_generate import BOX, DOMAIN, INTERFACE, bumpy_floor, cpw_size

pytest.importorskip("gmsh")
fd = pytest.importorskip("firedrake")


@pytest.fixture(scope="module")
def flat(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """2D mesh of :data:`DOMAIN` conforming to :data:`INTERFACE`.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Temporary directories.

    Returns
    -------
    pathlib.Path
        The mesh file.
    """
    return kmvmesh.generate_mesh(tmp_path_factory.mktemp("flat") / "flat.msh", DOMAIN,
                                 cpw_size(3.0), [INTERFACE])


@pytest.fixture(scope="module")
def solid(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """3D mesh of :data:`BOX` split by :func:`bumpy_floor`.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Temporary directories.

    Returns
    -------
    pathlib.Path
        The mesh file.
    """
    return kmvmesh.generate_mesh_3d(tmp_path_factory.mktemp("solid") / "solid.msh", BOX,
                                    UniformSize(0.15), bumpy_floor())


def floor_area(floor: FacetedSurface) -> float:
    """Area of the facets of a surface, split as :class:`FacetedSurface` does.

    Parameters
    ----------
    floor : FacetedSurface
        The surface.

    Returns
    -------
    float
        Total area of its triangles (km^2).
    """
    X, Y = np.meshgrid(floor.x, floor.y)
    p = np.stack([X, Y, -floor.depth], axis=-1)
    a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, 1:], p[1:, :-1]
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a), axis=-1).sum()
                       + np.linalg.norm(np.cross(c - a, d - a), axis=-1).sum())


def measures(mesh: "fd.MeshGeometry", measure: "fd.Measure", markers: tuple) -> dict:
    """Measure of each marked subdomain, and of the whole integration domain.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        The mesh.
    measure : firedrake.Measure
        ``ds``, ``dS`` or ``dx``.
    markers : tuple of int
        Subdomain markers.

    Returns
    -------
    dict
        Measure of each marker, and of all of the domain under ``None``.
    """
    one = fd.Constant(1.0)
    result = {m: fd.assemble(one * measure(m, domain=mesh)) for m in markers}
    result[None] = fd.assemble(one * measure(domain=mesh))
    return result


def separating_facets(mesh: "fd.MeshGeometry",
                      below: Callable[[np.ndarray], np.ndarray]) -> "fd.Function":
    """Mark the cells on one side of an interface, by their centroids.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        The mesh.
    below : callable
        ``below(centroids)`` is ``True`` for the cells under the interface,
        with ``centroids`` of shape ``(n_cells, d)``.

    Returns
    -------
    firedrake.Function
        Piecewise constant, 1 below the interface and 0 above. Its jump is
        non-zero on the facets between the two sides only.
    """
    centroids = fd.Function(fd.VectorFunctionSpace(mesh, "DG", 0)).interpolate(
        fd.SpatialCoordinate(mesh))
    indicator = fd.Function(fd.FunctionSpace(mesh, "DG", 0))
    indicator.dat.data[:] = below(centroids.dat.data_ro).astype(float)
    return indicator


@pytest.mark.parametrize("degree", range(1, 7))
def test_kmv_dimension_2d(flat: Path, degree: int) -> None:
    """``count_kmv_dofs`` is the dimension of Firedrake's KMV space, in 2D.

    Parameters
    ----------
    flat : pathlib.Path
        2D mesh.
    degree : int
        KMV degree.
    """
    mesh = fd.Mesh(str(flat))
    triangles = kmvmesh.read_triangulation(flat)[1]
    assert fd.FunctionSpace(mesh, "KMV", degree).dim() == \
        kmvmesh.count_kmv_dofs(triangles, degree)


@pytest.mark.parametrize("degree", range(1, 4))
def test_kmv_dimension_3d(solid: Path, degree: int) -> None:
    """``count_kmv_dofs`` is the dimension of Firedrake's KMV space, in 3D.

    Parameters
    ----------
    solid : pathlib.Path
        3D mesh.
    degree : int
        KMV degree.
    """
    mesh = fd.Mesh(str(solid))
    tetrahedra = kmvmesh.read_tetrahedra(solid)[1]
    assert fd.FunctionSpace(mesh, "KMV", degree).dim() == \
        kmvmesh.count_kmv_dofs(tetrahedra, degree)


def test_markers_and_interface_2d(flat: Path) -> None:
    """Every boundary edge is marked, and the interface is made of edges.

    Parameters
    ----------
    flat : pathlib.Path
        2D mesh, ``(z, x)`` coordinates.
    """
    mesh = fd.Mesh(str(flat))
    width, height = DOMAIN.x_max - DOMAIN.x_min, DOMAIN.z_max - DOMAIN.z_min
    lengths = measures(mesh, fd.ds, (TOP, BOTTOM, X_MIN, X_MAX))
    assert lengths == pytest.approx({TOP: width, BOTTOM: width, X_MIN: height,
                                     X_MAX: height, None: 2 * (width + height)})
    assert measures(mesh, fd.dx, (1,)) == pytest.approx({1: width * height,
                                                         None: width * height})
    jump = abs(fd.jump(separating_facets(mesh, lambda c: c[:, 0] < INTERFACE)))
    z = fd.SpatialCoordinate(mesh)[0]
    # The facets between the layers have the length of the interface, and
    # all of them lie on it.
    assert fd.assemble(jump * fd.dS) == pytest.approx(width)
    assert fd.assemble(jump * (z("+") - INTERFACE) ** 2 * fd.dS) == pytest.approx(0.0, abs=1e-20)


def test_markers_and_seafloor_3d(solid: Path) -> None:
    """Every boundary face is marked, and the sea floor is the marked facets.

    Parameters
    ----------
    solid : pathlib.Path
        3D mesh.
    """
    mesh = fd.Mesh(str(solid))
    floor = bumpy_floor()
    dx, dy, dz = BOX.x_max - BOX.x_min, BOX.y_max - BOX.y_min, BOX.z_max - BOX.z_min
    areas = measures(mesh, fd.ds, (TOP, BOTTOM, X_MIN, X_MAX, Y_MIN, Y_MAX))
    assert areas == pytest.approx({TOP: dx * dy, BOTTOM: dx * dy, X_MIN: dy * dz,
                                   X_MAX: dy * dz, Y_MIN: dx * dz, Y_MAX: dx * dz,
                                   None: 2 * (dx * dy + dy * dz + dx * dz)})
    assert measures(mesh, fd.dx, (1,)) == pytest.approx({1: dx * dy * dz,
                                                         None: dx * dy * dz})
    area = floor_area(floor)
    assert fd.assemble(fd.Constant(1.0) * fd.dS(SEAFLOOR, domain=mesh)) == pytest.approx(area)
    jump = abs(fd.jump(separating_facets(mesh, lambda c: floor.side(c) > 0)))
    # The facets between water and rock have the area of the sea floor, and
    # all of them are marked as sea floor.
    assert fd.assemble(jump * fd.dS) == pytest.approx(area)
    assert fd.assemble(jump * fd.dS(SEAFLOOR)) == pytest.approx(area)
