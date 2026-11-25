"""Mesh generation with gmsh, 2D and 3D, and reading back."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

import kmvmesh
from kmvmesh import (BOTTOM, SEAFLOOR, TOP, X_MAX, X_MIN, Y_MAX, Y_MIN, BoxDomain,
                     FacetedSurface, RectangularDomain, UniformSize, VelocityGrid)

gmsh = pytest.importorskip("gmsh")

DOMAIN = RectangularDomain(z_min=-1.0, x_min=0.0, x_max=1.2, pad=0.2)
INTERFACE = -0.3
BOX = BoxDomain(x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.8, z_min=-0.6)


def cpw_size(cpw: float, frequency: float = 5.0) -> VelocityGrid:
    """Uniform-CPW size of a two-layer model with the jump at ``INTERFACE``.

    Parameters
    ----------
    cpw : float
        Cells per wavelength.
    frequency : float, optional
        Frequency (Hz).

    Returns
    -------
    VelocityGrid
        Target edge length on a grid of spacing 0.01.
    """
    z, x = DOMAIN.grid(0.01)
    Z, _ = np.meshgrid(z, x, indexing="ij")
    velocity = np.where(Z - 1e-9 > INTERFACE, 1.5, 2.5)
    return VelocityGrid(kmvmesh.cpw_size(velocity, cpw, frequency), z[0], x[0], 0.01, 0.01)


def physical_measures(path: Path, dimension: int) -> dict:
    """Length, area or volume of each physical group of a dimension.

    Parameters
    ----------
    path : pathlib.Path
        Mesh file.
    dimension : int
        Dimension of the groups.

    Returns
    -------
    dict
        Measure of each physical tag.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        measures = {}
        for _, tag in gmsh.model.getPhysicalGroups(dimension):
            total = 0.0
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dimension, tag):
                types, _, _ = gmsh.model.mesh.getElements(dimension, entity)
                for element_type in types:
                    jacobians = gmsh.model.mesh.getJacobians(element_type, [1 / 3] * 3,
                                                             tag=entity)[1]
                    # Reference measures: line [-1, 1], triangle 1/2, tetrahedron 1/6.
                    factor = {1: 2.0, 2: 0.5, 4: 1 / 6}[element_type]
                    total += factor * float(np.abs(jacobians).sum())
            measures[tag] = total
    finally:
        gmsh.finalize()
    return measures


def test_markers_interface_and_size(tmp_path: Path) -> None:
    """Check the markers, the interface as mesh edges and the requested size.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    path = kmvmesh.generate_mesh(tmp_path / "m.msh", DOMAIN, cpw_size(3.0), [INTERFACE])
    lengths = physical_measures(path, 1)
    width, height = DOMAIN.x_max - DOMAIN.x_min, DOMAIN.z_max - DOMAIN.z_min
    assert set(lengths) == {TOP, BOTTOM, X_MIN, X_MAX}
    assert [lengths[k] for k in (TOP, BOTTOM, X_MIN, X_MAX)] == pytest.approx(
        [width, width, height, height])
    points, triangles = kmvmesh.read_triangulation(path)
    assert np.sum(np.isclose(points[:, 0], INTERFACE)) >= 5
    audit = kmvmesh.audit_triangles(points, triangles, [INTERFACE])
    assert audit["interface_conforming"] and audit["area_km2"] == pytest.approx(width * height)
    lengths = kmvmesh.mean_edge_lengths(points, triangles)
    above = points[triangles][:, :, 0].mean(axis=1) > INTERFACE
    assert np.median(lengths[above]) == pytest.approx(1.5 / 15.0, rel=0.15)
    assert np.median(lengths[~above]) == pytest.approx(2.5 / 15.0, rel=0.15)


def test_determinism_seed_and_perturbation(tmp_path: Path) -> None:
    """Check that gmsh repeats the mesh and the perturbed size field changes it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    size = cpw_size(3.0)
    factor = kmvmesh.size_perturbation(size.values.shape, size.dz, 1)
    files = {}
    for name, field, seed in [("plain", size, 0), ("again", size, 0), ("gmsh_seed", size, 1),
                              ("perturbed", size.scaled(factor), 1)]:
        files[name] = kmvmesh.generate_mesh(tmp_path / f"{name}.msh", DOMAIN, field, seed=seed)
    digest = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in files.items()}
    assert digest["plain"] == digest["again"]
    meshes = {k: kmvmesh.read_triangulation(v) for k, v in files.items()}
    assert np.array_equal(meshes["plain"][0], meshes["gmsh_seed"][0])
    assert (meshes["plain"][1].shape != meshes["perturbed"][1].shape
            or not np.allclose(meshes["plain"][0], meshes["perturbed"][0]))


def test_interface_outside_the_domain(tmp_path: Path) -> None:
    """An interface depth outside the domain is rejected.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    with pytest.raises(ValueError):
        kmvmesh.generate_mesh(tmp_path / "m.msh", DOMAIN, cpw_size(3.0), [0.5])


def bumpy_floor() -> FacetedSurface:
    """Build a sea floor with a bump, spanning :data:`BOX`.

    Returns
    -------
    FacetedSurface
        Depths between 0.15 and 0.25 km on a 0.2 km grid.
    """
    x, y = np.linspace(0.0, 1.0, 6), np.linspace(0.0, 0.8, 5)
    X, Y = np.meshgrid(x, y)
    return FacetedSurface(x, y, 0.15 + 0.1 * np.exp(-((X - 0.5) ** 2 + (Y - 0.4) ** 2) / 0.05))


def test_box_markers_and_volume(tmp_path: Path) -> None:
    """The box gets the markers 1 to 6, a single volume and an atomic write.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    path = kmvmesh.generate_mesh_3d(tmp_path / "box.msh", BOX, UniformSize(0.2))
    assert not (tmp_path / "box.tmp.msh").exists()
    areas = physical_measures(path, 2)
    assert set(areas) == {TOP, BOTTOM, X_MIN, X_MAX, Y_MIN, Y_MAX}
    assert areas[TOP] == pytest.approx(0.8) and areas[X_MIN] == pytest.approx(0.48)
    assert areas[Y_MAX] == pytest.approx(0.6)
    assert physical_measures(path, 3) == {1: pytest.approx(0.48)}
    points, tetrahedra = kmvmesh.read_tetrahedra(path)
    audit = kmvmesh.audit_tetrahedra(points, tetrahedra)
    assert audit["volume_km3"] == pytest.approx(0.48)
    assert audit["KMV3_dofs_predicted"] == kmvmesh.count_kmv_dofs(tetrahedra, 3)


def test_box_with_seafloor(tmp_path: Path) -> None:
    """The sea floor is a set of mesh faces that no tetrahedron crosses.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    floor = bumpy_floor()
    path = kmvmesh.generate_mesh_3d(tmp_path / "floor.msh", BOX, UniformSize(0.15), floor)
    areas = physical_measures(path, 2)
    assert set(areas) == {TOP, BOTTOM, X_MIN, X_MAX, Y_MIN, Y_MAX, SEAFLOOR}
    assert areas[SEAFLOOR] > 0.8
    points, tetrahedra = kmvmesh.read_tetrahedra(path)
    audit = kmvmesh.audit_tetrahedra(points, tetrahedra, floor)
    assert audit["seafloor_crossing_tetrahedra"] == 0 and audit["interface_conforming"]
    assert audit["volume_km3"] == pytest.approx(0.48)
    assert np.sum(floor.side(points) == 0) >= len(floor.x) * len(floor.y)


def test_seafloor_must_fit_the_box(tmp_path: Path) -> None:
    """A floor that does not span the box, or leaves it, is rejected.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    floor = bumpy_floor()
    short = FacetedSurface(floor.x * 0.9, floor.y, floor.depth)
    deep = FacetedSurface(floor.x, floor.y, floor.depth + 1.0)
    for surface in (short, deep):
        with pytest.raises(ValueError):
            kmvmesh.generate_mesh_3d(tmp_path / "m.msh", BOX, UniformSize(0.2), surface)


def test_read_mesh_and_audit_mesh(tmp_path: Path) -> None:
    """``read_mesh`` and ``audit_mesh`` take the dimension from the file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    flat = kmvmesh.generate_mesh(tmp_path / "flat.msh", DOMAIN, cpw_size(2.0), [INTERFACE])
    solid = kmvmesh.generate_mesh_3d(tmp_path / "solid.msh", BOX, UniformSize(0.3))
    points, cells = kmvmesh.read_mesh(flat)
    assert points.shape[1] == 2 and cells.shape[1] == 3
    points, cells = kmvmesh.read_mesh(solid)
    assert points.shape[1] == 3 and cells.shape[1] == 4
    record = kmvmesh.audit_mesh(flat, 4, [INTERFACE])
    assert record["dimension"] == 2 and record["kmv_degree"] == 4
    assert record["kmv_dofs"] == kmvmesh.count_kmv_dofs(kmvmesh.read_triangulation(flat)[1], 4)
    assert record["sha256"] == hashlib.sha256(flat.read_bytes()).hexdigest()
    assert record["geometry"]["interface_conforming"]
    assert record["element"] == "Kong-Mulder-Veldhuizen" and record["dofs"] == record["kmv_dofs"]
    record = kmvmesh.audit_mesh(solid, 3)
    assert record["dimension"] == 3
    assert record["kmv_dofs"] == record["geometry"]["KMV3_dofs_predicted"]
    record = kmvmesh.audit_mesh(flat, 2, [INTERFACE], family="CG")
    assert record["element"] == "Lagrange" and record["degree"] == 2
    assert record["dofs"] == kmvmesh.count_dofs(kmvmesh.read_triangulation(flat)[1], "CG", 2)
    assert "kmv_dofs" not in record and "kmv_degree" not in record
