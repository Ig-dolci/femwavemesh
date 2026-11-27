"""Mesh generation with gmsh, 2D and 3D, and reading back."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

import femwavemesh
from femwavemesh import (BOTTOM, SEAFLOOR, TOP, X_MAX, X_MIN, Y_MAX, Y_MIN, BoxDomain,
                         FacetedSurface, RectangularDomain, UniformSize, VelocityGrid)

gmsh = pytest.importorskip("gmsh")

DOMAIN = RectangularDomain(z_min=-1.0, x_min=0.0, x_max=1.2, pad=0.2)
INTERFACE = -0.3
BOX = BoxDomain(x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.8, z_min=-0.6)

#: Shape-quality bounds of the generated meshes, with room to spare. gmsh 4.15.2
#: gives, on the meshes of these tests, radius ratios of at least 0.73 and angles
#: from 36.7 to 98.6 degrees in 2D; radius ratios of at least 0.26 and dihedral
#: angles from 12.6 to 151.1 degrees in 3D. Slivers and needles fall outside.
QUALITY = {"radius_ratio_min": (0.45, 0.15), "angle_min_deg": (22.0, None),
           "angle_max_deg": (130.0, None), "dihedral_min_deg": (None, 7.0),
           "dihedral_max_deg": (None, 165.0)}


def assert_good_shapes(audit: dict, dimension: int) -> None:
    """Check the shape quality of an audited mesh against :data:`QUALITY`.

    Parameters
    ----------
    audit : dict
        Output of ``audit_triangles`` or ``audit_tetrahedra``.
    dimension : int
        2 or 3.
    """
    for key, bounds in QUALITY.items():
        bound = bounds[dimension - 2]
        if bound is not None:
            assert (audit[key] < bound if key.endswith("max_deg") else audit[key] > bound), \
                f"{key} = {audit[key]:.3f} (bound {bound})"


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
    return VelocityGrid(femwavemesh.cpw_size(velocity, cpw, frequency), z[0], x[0], 0.01, 0.01)


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
    path = femwavemesh.generate_mesh(tmp_path / "m.msh", DOMAIN, cpw_size(3.0), [INTERFACE])
    lengths = physical_measures(path, 1)
    width, height = DOMAIN.x_max - DOMAIN.x_min, DOMAIN.z_max - DOMAIN.z_min
    assert set(lengths) == {TOP, BOTTOM, X_MIN, X_MAX}
    assert [lengths[k] for k in (TOP, BOTTOM, X_MIN, X_MAX)] == pytest.approx(
        [width, width, height, height])
    points, triangles = femwavemesh.read_triangulation(path)
    assert np.sum(np.isclose(points[:, 0], INTERFACE)) >= 5
    audit = femwavemesh.audit_triangles(points, triangles, [INTERFACE])
    assert audit["interface_conforming"] and audit["area_km2"] == pytest.approx(width * height)
    assert_good_shapes(audit, 2)
    lengths = femwavemesh.mean_edge_lengths(points, triangles)
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
    factor = femwavemesh.size_perturbation(size.values.shape, size.dz, 1)
    files = {}
    for name, field, seed in [("plain", size, 0), ("again", size, 0), ("gmsh_seed", size, 1),
                              ("perturbed", size.scaled(factor), 1)]:
        files[name] = femwavemesh.generate_mesh(tmp_path / f"{name}.msh", DOMAIN, field, seed=seed)
    digest = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in files.items()}
    assert digest["plain"] == digest["again"]
    meshes = {k: femwavemesh.read_triangulation(v) for k, v in files.items()}
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
        femwavemesh.generate_mesh(tmp_path / "m.msh", DOMAIN, cpw_size(3.0), [0.5])


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
    path = femwavemesh.generate_mesh_3d(tmp_path / "box.msh", BOX, UniformSize(0.2))
    assert not (tmp_path / "box.tmp.msh").exists()
    areas = physical_measures(path, 2)
    assert set(areas) == {TOP, BOTTOM, X_MIN, X_MAX, Y_MIN, Y_MAX}
    assert areas[TOP] == pytest.approx(0.8) and areas[X_MIN] == pytest.approx(0.48)
    assert areas[Y_MAX] == pytest.approx(0.6)
    assert physical_measures(path, 3) == {1: pytest.approx(0.48)}
    points, tetrahedra = femwavemesh.read_tetrahedra(path)
    audit = femwavemesh.audit_tetrahedra(points, tetrahedra)
    assert audit["volume_km3"] == pytest.approx(0.48)
    assert audit["KMV3_dofs_predicted"] == femwavemesh.count_kmv_dofs(tetrahedra, 3)
    assert_good_shapes(audit, 3)


def test_box_with_seafloor(tmp_path: Path) -> None:
    """The sea floor is a set of mesh faces that no tetrahedron crosses.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    floor = bumpy_floor()
    path = femwavemesh.generate_mesh_3d(tmp_path / "floor.msh", BOX, UniformSize(0.15), floor)
    areas = physical_measures(path, 2)
    assert set(areas) == {TOP, BOTTOM, X_MIN, X_MAX, Y_MIN, Y_MAX, SEAFLOOR}
    assert areas[SEAFLOOR] > 0.8
    points, tetrahedra = femwavemesh.read_tetrahedra(path)
    audit = femwavemesh.audit_tetrahedra(points, tetrahedra, floor)
    assert audit["seafloor_crossing_tetrahedra"] == 0 and audit["interface_conforming"]
    assert audit["volume_km3"] == pytest.approx(0.48)
    assert_good_shapes(audit, 3)
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
            femwavemesh.generate_mesh_3d(tmp_path / "m.msh", BOX, UniformSize(0.2), surface)


def test_read_mesh_and_audit_mesh(tmp_path: Path) -> None:
    """``read_mesh`` and ``audit_mesh`` take the dimension from the file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    flat = femwavemesh.generate_mesh(tmp_path / "flat.msh", DOMAIN, cpw_size(2.0), [INTERFACE])
    solid = femwavemesh.generate_mesh_3d(tmp_path / "solid.msh", BOX, UniformSize(0.3))
    points, cells = femwavemesh.read_mesh(flat)
    assert points.shape[1] == 2 and cells.shape[1] == 3
    points, cells = femwavemesh.read_mesh(solid)
    assert points.shape[1] == 3 and cells.shape[1] == 4
    record = femwavemesh.audit_mesh(flat, 4, [INTERFACE])
    assert record["dimension"] == 2 and record["kmv_degree"] == 4
    assert record["kmv_dofs"] == femwavemesh.count_kmv_dofs(femwavemesh.read_triangulation(flat)[1], 4)
    assert record["sha256"] == hashlib.sha256(flat.read_bytes()).hexdigest()
    assert record["geometry"]["interface_conforming"]
    assert record["element"] == "Kong-Mulder-Veldhuizen" and record["dofs"] == record["kmv_dofs"]
    record = femwavemesh.audit_mesh(solid, 3)
    assert record["dimension"] == 3
    assert record["kmv_dofs"] == record["geometry"]["KMV3_dofs_predicted"]
    record = femwavemesh.audit_mesh(flat, 2, [INTERFACE], family="CG")
    assert record["element"] == "Lagrange" and record["degree"] == 2
    assert record["dofs"] == femwavemesh.count_dofs(femwavemesh.read_triangulation(flat)[1], "CG", 2)
    assert "kmv_dofs" not in record and "kmv_degree" not in record


def no_view_data(path: Path) -> bool:
    """Tell whether a mesh file holds no post-processing data.

    Parameters
    ----------
    path : pathlib.Path
        Mesh file.

    Returns
    -------
    bool
        ``True`` if the file has no ``$NodeData``, ``$ElementData`` or view.
    """
    content = path.read_bytes()
    return not any(key in content for key in (b"$NodeData", b"$ElementData", b"$View"))


def test_mesh_size_2d(tmp_path: Path) -> None:
    """A size on triangles meshes like the same size on a grid.

    The size is linear in depth, so the grid (bilinear) and the triangles
    (linear) give gmsh the same field.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    points, triangles = femwavemesh.read_triangulation(
        femwavemesh.generate_mesh(tmp_path / "coarse.msh", DOMAIN, cpw_size(2.0)))

    def law(z: np.ndarray) -> np.ndarray:
        """Size linear in depth.

        Parameters
        ----------
        z : numpy.ndarray
            Depths.

        Returns
        -------
        numpy.ndarray
            Sizes (km).
        """
        return 0.04 + 0.08 * -z

    z, x = DOMAIN.grid(0.01)
    grid = VelocityGrid(np.repeat(law(z)[:, np.newaxis], len(x), axis=1), z[0], x[0], 0.01, 0.01)
    on_grid = femwavemesh.generate_mesh(tmp_path / "grid.msh", DOMAIN, grid, [INTERFACE])
    on_mesh = femwavemesh.generate_mesh(tmp_path / "mesh.msh", DOMAIN,
                                        femwavemesh.MeshSize(points, triangles, law(points[:, 0])),
                                        [INTERFACE])
    assert no_view_data(on_mesh)
    assert physical_measures(on_mesh, 1) == pytest.approx(physical_measures(on_grid, 1))
    new_points, new_triangles = femwavemesh.read_triangulation(on_mesh)
    grid_triangles = femwavemesh.read_triangulation(on_grid)[1]
    assert len(new_triangles) == pytest.approx(len(grid_triangles), rel=0.05)
    audit = femwavemesh.audit_triangles(new_points, new_triangles, [INTERFACE])
    assert audit["interface_conforming"]
    assert_good_shapes(audit, 2)
    lengths = femwavemesh.mean_edge_lengths(new_points, new_triangles)
    depth = new_points[new_triangles][:, :, 0].mean(axis=1)
    for top, bottom in ((0.0, -0.2), (-0.8, -1.0)):
        band = (depth < top) & (depth > bottom)
        assert np.median(lengths[band]) == pytest.approx(law((top + bottom) / 2), rel=0.15)


def test_mesh_size_3d(tmp_path: Path) -> None:
    """A size on tetrahedra meshes like the same size at scattered points.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    points, tetrahedra = femwavemesh.read_tetrahedra(
        femwavemesh.generate_mesh_3d(tmp_path / "coarse.msh", BOX, UniformSize(0.2)))
    sizes = 0.06 + 0.1 * -points[:, 2]
    on_mesh = femwavemesh.generate_mesh_3d(tmp_path / "mesh.msh", BOX,
                                           femwavemesh.MeshSize(points, tetrahedra, sizes))
    scattered = femwavemesh.generate_mesh_3d(tmp_path / "scattered.msh", BOX,
                                             femwavemesh.ScatteredSize(points, sizes))
    assert not (tmp_path / "mesh.tmp.msh").exists() and no_view_data(on_mesh)
    assert physical_measures(on_mesh, 3) == {1: pytest.approx(0.48)}
    new_points, new_tetrahedra = femwavemesh.read_tetrahedra(on_mesh)
    reference = femwavemesh.read_tetrahedra(scattered)[1]
    assert len(new_tetrahedra) == pytest.approx(len(reference), rel=0.1)
    audit = femwavemesh.audit_tetrahedra(new_points, new_tetrahedra)
    assert audit["volume_km3"] == pytest.approx(0.48)
    assert_good_shapes(audit, 3)
    floor = bumpy_floor()
    split = femwavemesh.generate_mesh_3d(tmp_path / "floor.msh", BOX,
                                         femwavemesh.MeshSize(points, tetrahedra, sizes), floor)
    audit = femwavemesh.audit_tetrahedra(*femwavemesh.read_tetrahedra(split), floor)
    assert audit["interface_conforming"] and audit["volume_km3"] == pytest.approx(0.48)
    assert_good_shapes(audit, 3)


def test_mesh_size_must_fit_the_domain(tmp_path: Path) -> None:
    """A mesh size of the wrong dimension, or short of the domain, is rejected.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    points = np.array([[DOMAIN.z_min, DOMAIN.x_min], [DOMAIN.z_min, DOMAIN.x_max],
                       [DOMAIN.z_max, DOMAIN.x_max], [DOMAIN.z_max, DOMAIN.x_min]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    whole = femwavemesh.MeshSize(points, triangles, np.full(4, 0.2))
    half = femwavemesh.MeshSize(points * [0.5, 1.0], triangles, np.full(4, 0.2))
    swapped = femwavemesh.MeshSize(points[:, ::-1], triangles, np.full(4, 0.2))  # (x, z)
    cube = femwavemesh.MeshSize(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1]], float),
                                np.array([[0, 1, 2, 3]]), np.full(4, 0.2))
    for size in (half, swapped, cube):
        with pytest.raises(ValueError):
            femwavemesh.generate_mesh(tmp_path / "m.msh", DOMAIN, size)
    for size in (whole, cube):  # triangles, and a tetrahedron short of the box
        with pytest.raises(ValueError):
            femwavemesh.generate_mesh_3d(tmp_path / "m.msh", BOX, size)


def test_radius_ratio_is_gmsh_gamma(tmp_path: Path) -> None:
    """``radius_ratios`` agrees with gmsh's own ``gamma`` quality, 2D and 3D.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    paths = {2: femwavemesh.generate_mesh(tmp_path / "flat.msh", DOMAIN, cpw_size(2.0),
                                          [INTERFACE]),
             3: femwavemesh.generate_mesh_3d(tmp_path / "solid.msh", BOX, UniformSize(0.3),
                                             bumpy_floor())}
    for dimension, path in paths.items():
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(str(path))
            tags, _ = gmsh.model.mesh.getElementsByType(2 if dimension == 2 else 4)
            gamma = np.array(gmsh.model.mesh.getElementQualities(tags, "gamma"))
        finally:
            gmsh.finalize()
        points, cells = femwavemesh.read_mesh(path)
        np.testing.assert_allclose(femwavemesh.radius_ratios(points, cells), gamma,
                                   rtol=0, atol=1e-12)
