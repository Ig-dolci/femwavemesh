"""kmvmesh against spyro's 2D mesh code, frozen in ``legacy``.

Each case builds the same mesh with the old code and with kmvmesh and asks
for the same file (SHA-256), the same vertices and triangles and the same
intermediate arrays, bit for bit. The local mesh size of the old code needs
matplotlib, its DOF count FIAT; the cases that use them are skipped without.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

import kmvmesh

from .legacy import cpw_fontes_source_scale as legacy_sources
from .legacy import spyro_mesh_generation as legacy
from .legacy import spyro_size_fields as legacy_sizes

pytest.importorskip("gmsh")

INTERFACE = -0.3
FREQUENCY = 5.0
SPACING = 0.01
BOX = dict(z_min=-1.0, x_min=0.0, x_max=1.2, pad=0.2)
SOURCES = [(-0.05, 0.4), (-0.05, 0.8)]


def digest(path: Path) -> str:
    """SHA-256 of a file.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def two_layer() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Water over a faster layer with a lateral gradient, on the design grid.

    Returns
    -------
    z, x : numpy.ndarray
        Grid axes.
    velocity : numpy.ndarray
        Velocity on the grid (km/s).
    """
    z, x = kmvmesh.RectangularDomain(**BOX).grid(SPACING)
    Z, X = np.meshgrid(z, x, indexing="ij")
    return z, x, np.where(Z - 1e-9 > INTERFACE, 1.5, 2.5 + 0.5 * X - 0.4 * Z)


def assert_same_mesh(old: Path, new: Path) -> tuple[np.ndarray, np.ndarray]:
    """Assert that two mesh files are identical.

    Parameters
    ----------
    old : pathlib.Path
        Mesh of the old code.
    new : pathlib.Path
        Mesh of kmvmesh.

    Returns
    -------
    tuple of numpy.ndarray
        Vertices and triangles of the mesh.
    """
    old_points, old_triangles = legacy.read_triangulation(old)
    points, triangles = kmvmesh.read_triangulation(new)
    np.testing.assert_array_equal(points, old_points)
    np.testing.assert_array_equal(triangles, old_triangles)
    assert digest(new) == digest(old)
    return points, triangles


@pytest.mark.parametrize("depths, seed", [((), 0), ((INTERFACE,), 0),
                                          ((INTERFACE, -0.55), 1)])
def test_cpw_mesh(tmp_path: Path, depths: tuple, seed: int) -> None:
    """A uniform-CPW mesh, with and without interfaces and with a gmsh seed.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    depths : tuple
        Conforming depths.
    seed : int
        gmsh seed.
    """
    z, x, v = two_layer()
    old = legacy.generate_mesh(
        tmp_path / "old.msh", legacy.RectangularDomain(**BOX),
        legacy_sizes.VelocityGrid(v / (3.0 * FREQUENCY), z[0], x[0], SPACING, SPACING),
        depths, seed=seed)
    new = kmvmesh.generate_mesh(
        tmp_path / "new.msh", kmvmesh.RectangularDomain(**BOX),
        kmvmesh.VelocityGrid(kmvmesh.cpw_size(v, 3.0, FREQUENCY), z[0], x[0],
                             SPACING, SPACING), depths, seed=seed)
    points, triangles = assert_same_mesh(old, new)
    audit = kmvmesh.audit_triangles(points, triangles, depths)
    assert audit["interface_conforming"]
    assert audit["area_km2"] == pytest.approx(1.2)


def test_uniform_mesh(tmp_path: Path) -> None:
    """A constant size, as the uniform 2D references of the campaigns.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    z, x, v = two_layer()
    h = float(v.min()) / (4.0 * FREQUENCY)
    old = legacy.generate_mesh(
        tmp_path / "old.msh", legacy.RectangularDomain(**BOX),
        legacy_sizes.VelocityGrid(np.full(v.shape, h), z[0], x[0], SPACING, SPACING),
        (INTERFACE,), seed=0)
    new = kmvmesh.generate_mesh(
        tmp_path / "new.msh", kmvmesh.RectangularDomain(**BOX),
        kmvmesh.VelocityGrid(np.full(v.shape, h), z[0], x[0], SPACING, SPACING),
        (INTERFACE,), seed=0)
    assert_same_mesh(old, new)


def test_perturbed_replica(tmp_path: Path) -> None:
    """A replica: the size times the smooth perturbation, gmsh seed 1.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    z, x, v = two_layer()
    old_factor = legacy_sizes.size_perturbation(v.shape, SPACING, 1)
    factor = kmvmesh.size_perturbation(v.shape, SPACING, 1)
    np.testing.assert_array_equal(factor, old_factor)
    old = legacy.generate_mesh(
        tmp_path / "old.msh", legacy.RectangularDomain(**BOX),
        legacy_sizes.VelocityGrid(v, z[0], x[0], SPACING, SPACING)
        .scaled(1 / (3.0 * FREQUENCY)).scaled(old_factor), (INTERFACE,), seed=1)
    new = kmvmesh.generate_mesh(
        tmp_path / "new.msh", kmvmesh.RectangularDomain(**BOX),
        kmvmesh.VelocityGrid(v, z[0], x[0], SPACING, SPACING)
        .scaled(1 / (3.0 * FREQUENCY)).scaled(factor), (INTERFACE,), seed=1)
    assert_same_mesh(old, new)


def test_source_refined_cpw(tmp_path: Path) -> None:
    """The CPW mesh refined around the sources of ``cpw_fontes.py``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    z, x, v = two_layer()
    factor = kmvmesh.size_perturbation(v.shape, SPACING, 2)
    old_scale = legacy_sources.source_scale(z, x, SOURCES, 0.25, 2.0, 0.1)
    scale = kmvmesh.source_refinement((z, x), SOURCES, 0.25, 2.0, 0.1)
    np.testing.assert_array_equal(scale, old_scale)
    np.testing.assert_array_equal(
        kmvmesh.source_refinement((z, x), SOURCES, 0.25, 3.0, 0.0),
        legacy_sources.source_scale(z, x, SOURCES, 0.25, 3.0, 0.0))
    # cpw_fontes.SourceRefinedCPW.cpw_mesh with seed 2.
    old = legacy.generate_mesh(
        tmp_path / "old.msh", legacy.RectangularDomain(**BOX),
        legacy_sizes.VelocityGrid(v * factor * old_scale / (2.0 * FREQUENCY),
                                  z[0], x[0], SPACING, SPACING), (INTERFACE,), seed=2)
    new = kmvmesh.generate_mesh(
        tmp_path / "new.msh", kmvmesh.RectangularDomain(**BOX),
        kmvmesh.VelocityGrid(kmvmesh.cpw_size(v * factor * scale, 2.0, FREQUENCY),
                             z[0], x[0], SPACING, SPACING), (INTERFACE,), seed=2)
    assert_same_mesh(old, new)


def legacy_adaptive_mesh(path: Path, current: Path, density: np.ndarray,
                         velocity: np.ndarray, z: np.ndarray, x: np.ndarray,
                         target: int, equivalent_cpw: float, size_factor: np.ndarray,
                         seed: int, ceiling: np.ndarray | None,
                         dof_tolerance: float) -> tuple:
    """Build the mesh as spyro's ``AutomaticMeshAdaptation.adaptive_mesh`` did before kmvmesh.

    The body is that method's, with the old modules; a ``ceiling`` adds the
    bound of ``MeshAdaptiveFWI._designed_mesh``.

    Parameters
    ----------
    path : pathlib.Path
        New mesh.
    current : pathlib.Path
        Current mesh.
    density : numpy.ndarray
        Refinement density.
    velocity : numpy.ndarray
        Velocity on the grid.
    z, x : numpy.ndarray
        Grid axes.
    target : int
        Target DOFs.
    equivalent_cpw : float
        Cells per wavelength of the uniform mesh with that budget.
    size_factor : numpy.ndarray
        Replica factor.
    seed : int
        Replica.
    ceiling : numpy.ndarray or None
        Upper bound of the size.
    dof_tolerance : float
        Relative tolerance of the budget.

    Returns
    -------
    tuple
        Current size, the sizes of every attempt and the DOFs obtained.
    """
    order, cmax_ratio, degree = 2.7, 1.25, 4
    points, triangles = legacy.read_triangulation(current)
    current_size = legacy.local_mesh_size(points, triangles, z, x)
    field = legacy_sizes.AdaptiveSizeField(density, current_size, z, x, order,
                                           ceiling=ceiling)
    field.set_floor(velocity, FREQUENCY, equivalent_cpw, cmax_ratio)
    calibration = (1 * legacy.count_kmv_dofs(triangles, degree)
                   / float((current_size ** -2).sum() * SPACING**2))
    scale = field.scale_for(target / calibration)
    dofs, sizes = 0, []
    for _ in range(6):
        field.set_scale(scale)
        sizes.append(field.h.copy())
        legacy.generate_mesh(path, legacy.RectangularDomain(**BOX),
                             field.grid.scaled(size_factor), (INTERFACE,), seed=seed)
        dofs = legacy.count_kmv_dofs(legacy.read_triangulation(path)[1], degree)
        if abs(dofs / target - 1.0) <= dof_tolerance:
            break
        scale *= math.sqrt(dofs / target)
    return current_size, sizes, dofs


def kmvmesh_adaptive_mesh(path: Path, current: Path, density: np.ndarray,
                          velocity: np.ndarray, z: np.ndarray, x: np.ndarray,
                          target: int, equivalent_cpw: float, size_factor: np.ndarray,
                          seed: int, ceiling: np.ndarray | None,
                          dof_tolerance: float) -> tuple:
    """Build the same mesh with kmvmesh, as spyro now does.

    Parameters
    ----------
    path : pathlib.Path
        New mesh.
    current : pathlib.Path
        Current mesh.
    density : numpy.ndarray
        Refinement density.
    velocity : numpy.ndarray
        Velocity on the grid.
    z, x : numpy.ndarray
        Grid axes.
    target : int
        Target DOFs.
    equivalent_cpw : float
        Cells per wavelength of the uniform mesh with that budget.
    size_factor : numpy.ndarray
        Replica factor.
    seed : int
        Replica.
    ceiling : numpy.ndarray or None
        Upper bound of the size.
    dof_tolerance : float
        Relative tolerance of the budget.

    Returns
    -------
    tuple
        Current size, the sizes of every attempt and the DOFs obtained.
    """
    points, triangles = kmvmesh.read_triangulation(current)
    current_size = kmvmesh.local_mesh_size(points, triangles, z, x)
    field = kmvmesh.AdaptiveSizeField(density, current_size, z, x, 2.7, ceiling=ceiling)
    field.set_floor(velocity, FREQUENCY, equivalent_cpw, 1.25)
    calibration = kmvmesh.dof_calibration(1 * kmvmesh.count_kmv_dofs(triangles, 4),
                                          current_size, SPACING)
    sizes = []

    def build(law: kmvmesh.AdaptiveSizeField) -> int:
        """Generate the mesh of the law and count its DOFs.

        Parameters
        ----------
        law : kmvmesh.AdaptiveSizeField
            The law at the current scale.

        Returns
        -------
        int
            KMV4 DOFs.
        """
        sizes.append(law.h.copy())
        kmvmesh.generate_mesh(path, kmvmesh.RectangularDomain(**BOX),
                              law.grid.scaled(size_factor), (INTERFACE,), seed=seed)
        return kmvmesh.count_kmv_dofs(kmvmesh.read_triangulation(path)[1], 4)

    dofs = kmvmesh.fit_to_budget(field, target, calibration, build, dof_tolerance)
    return current_size, sizes, dofs


@pytest.mark.parametrize("seed, capped, tolerance", [
    (0, False, 0.02), (1, False, 0.02), (0, True, 0.02), (0, False, 0.2)])
def test_adaptive_mesh(tmp_path: Path, seed: int, capped: bool, tolerance: float) -> None:
    """The adaptive law with a budget, from a CPW mesh and a density.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    seed : int
        Replica (size factor and gmsh seed).
    capped : bool
        With a ceiling below the current size on the right half.
    tolerance : float
        Relative tolerance of the budget (0.2 stops before the last attempt).
    """
    pytest.importorskip("matplotlib")
    pytest.importorskip("FIAT")
    z, x, v = two_layer()
    Z, X = np.meshgrid(z, x, indexing="ij")
    current = kmvmesh.generate_mesh(
        tmp_path / "current.msh", kmvmesh.RectangularDomain(**BOX),
        kmvmesh.VelocityGrid(kmvmesh.cpw_size(v, 3.0, FREQUENCY), z[0], x[0],
                             SPACING, SPACING), (INTERFACE,))
    interior = kmvmesh.RectangularDomain(**BOX).in_interior(Z, X)
    density = legacy_sizes.gradient_density(v, SPACING, 0.05, interior)
    density += 0.2 * np.exp(-((Z + 0.6) ** 2 + (X - 0.6) ** 2) / 0.02)
    factor = kmvmesh.size_perturbation(v.shape, SPACING, seed)
    ceiling = v / (2.5 * FREQUENCY) * np.where(X > 0.6, 0.8, 1.2) if capped else None
    target, cpw = 1800, 3.0 * math.sqrt(1800 / 1500)
    old = legacy_adaptive_mesh(tmp_path / "old.msh", current, density, v, z, x,
                               target, cpw, factor, seed, ceiling, tolerance)
    new = kmvmesh_adaptive_mesh(tmp_path / "new.msh", current, density, v, z, x,
                                target, cpw, factor, seed, ceiling, tolerance)
    np.testing.assert_array_equal(new[0], old[0])
    assert len(new[1]) == len(old[1])
    for h_new, h_old in zip(new[1], old[1]):
        np.testing.assert_array_equal(h_new, h_old)
    assert new[2] == old[2]
    assert_same_mesh(tmp_path / "old.msh", tmp_path / "new.msh")


def cpw_mesh(tmp_path: Path) -> Path:
    """Generate a CPW mesh with the interface.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.

    Returns
    -------
    pathlib.Path
        The mesh file.
    """
    z, x, v = two_layer()
    return kmvmesh.generate_mesh(
        tmp_path / "m.msh", kmvmesh.RectangularDomain(**BOX),
        kmvmesh.VelocityGrid(kmvmesh.cpw_size(v, 4.0, FREQUENCY), z[0], x[0],
                             SPACING, SPACING), (INTERFACE,))


def test_reading_and_edge_lengths(tmp_path: Path) -> None:
    """Reading and mean edge lengths.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    path = cpw_mesh(tmp_path)
    points, triangles = kmvmesh.read_triangulation(path)
    old_points, old_triangles = legacy.read_triangulation(path)
    np.testing.assert_array_equal(points, old_points)
    np.testing.assert_array_equal(triangles, old_triangles)
    np.testing.assert_array_equal(kmvmesh.mean_edge_lengths(points, triangles),
                                  legacy.mean_edge_lengths(points, triangles))


def test_local_size(tmp_path: Path) -> None:
    """Local size on grids whose rows fall on the interface and the boundary.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    pytest.importorskip("matplotlib")
    points, triangles = kmvmesh.read_triangulation(cpw_mesh(tmp_path))
    for spacing in (0.005, 0.01, 0.0125, 0.05):
        z, x = kmvmesh.RectangularDomain(**BOX).grid(spacing)
        np.testing.assert_array_equal(kmvmesh.local_mesh_size(points, triangles, z, x),
                                      legacy.local_mesh_size(points, triangles, z, x))


def test_kmv_counts(tmp_path: Path) -> None:
    """KMV DOFs of degrees 1 to 6 against FIAT's entity counts.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    pytest.importorskip("FIAT")
    triangles = kmvmesh.read_triangulation(cpw_mesh(tmp_path))[1]
    for degree in range(1, 7):
        assert kmvmesh.count_kmv_dofs(triangles, degree) == \
            legacy.count_kmv_dofs(triangles, degree)


def test_size_law_matches() -> None:
    """The law's arrays, cost and scale, bit for bit."""
    z, x, v = two_layer()
    Z, X = np.meshgrid(z, x, indexing="ij")
    density = 1.0 + np.exp(-((Z + 0.5) ** 2 + (X - 0.6) ** 2) / 0.05)
    current = v / (3.0 * FREQUENCY)
    old = legacy_sizes.AdaptiveSizeField(density, current, z, x, 2.7)
    new = kmvmesh.AdaptiveSizeField(density, current, z, x, 2.7)
    for field in (old, new):
        field.set_floor(v, FREQUENCY, 4.0)
    np.testing.assert_array_equal(new.shape, old.shape)
    np.testing.assert_array_equal(new.h_min, old.h_min)
    target = 1.2 * old.relative_cost(1.0)
    assert new.scale_for(target) == old.scale_for(target)
    assert new.relative_cost(0.7) == old.relative_cost(0.7)
    old.set_scale(0.9)
    new.set_scale(0.9)
    np.testing.assert_array_equal(new.grid.values, old.grid.values)
    assert (new.grid.z0, new.grid.x0, new.grid.dz, new.grid.dx) == \
        (old.grid.z0, old.grid.x0, old.grid.dz, old.grid.dx)
    layered = (INTERFACE, 1.5)
    queries = (np.array([-1.3, -0.4, -0.3, 0.0]), np.array([0.1, 0.7, 1.9, 0.5]))
    np.testing.assert_array_equal(
        kmvmesh.VelocityGrid(v, z[0], x[0], SPACING, SPACING, layered)(*queries),
        legacy_sizes.VelocityGrid(v, z[0], x[0], SPACING, SPACING, layered)(*queries))
