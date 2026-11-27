"""VTK files of grids and meshes."""

from pathlib import Path

import numpy as np
import pytest

from femwavemesh import (VelocityGrid, VelocityGrid3D, mean_edge_lengths, write_grid_vtk,
                         write_mesh_vtk)


def read_ascii_vtk(path: Path) -> dict:
    """Read the sections of an ASCII legacy VTK file written by femwavemesh.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    dict
        ``DIMENSIONS``, ``ORIGIN`` and ``SPACING`` as lists, ``POINTS``,
        ``CELLS`` and ``CELL_TYPES`` as 2D arrays, and each scalar array by
        its name.
    """
    lines = path.read_text().splitlines()
    assert lines[0] == "# vtk DataFile Version 3.0" and lines[2] == "ASCII"
    sections, i, count = {}, 4, 0
    while i < len(lines):
        words = lines[i].split()
        if words[0] in ("DIMENSIONS", "ORIGIN", "SPACING"):
            sections[words[0]] = [float(w) for w in words[1:]]
            i += 1
        elif words[0] in ("POINTS", "CELLS", "CELL_TYPES"):
            n = int(words[1])
            sections[words[0]] = np.loadtxt(lines[i + 1:i + 1 + n], ndmin=2)
            i += 1 + n
        elif words[0] in ("POINT_DATA", "CELL_DATA"):
            count = int(words[1])
            i += 1
        else:
            assert words[0] == "SCALARS" and lines[i + 1] == "LOOKUP_TABLE default"
            sections[words[1]] = np.loadtxt(lines[i + 2:i + 2 + count], ndmin=1)
            i += 2 + count
    return sections


def test_grid_2d(tmp_path: Path) -> None:
    """A 2D grid stands upright, with its fixed layer, whatever the sign of ``dz``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    z, x = np.linspace(-1.0, 0.0, 5), np.linspace(2.0, 3.0, 3)
    values = 2.0 - z[:, np.newaxis] + x
    grid = VelocityGrid(values, z[0], x[0], 0.25, 0.5, fixed_layer=(-0.3, 1.5))
    sections = read_ascii_vtk(write_grid_vtk(tmp_path / "up.vtk", grid, "vp", binary=False))
    assert sections["DIMENSIONS"] == [3, 5, 1]
    assert sections["ORIGIN"] == [2.0, -1.0, 0.0] and sections["SPACING"] == [0.5, 0.25, 1.0]
    expected = np.where(z[:, np.newaxis] > -0.3, 1.5, values)
    np.testing.assert_array_equal(sections["vp"], expected.ravel())
    reversed_rows = VelocityGrid(values[::-1], z[-1], x[0], -0.25, 0.5, fixed_layer=(-0.3, 1.5))
    down = write_grid_vtk(tmp_path / "down.vtk", reversed_rows, "vp", binary=False)
    assert down.read_bytes() == (tmp_path / "up.vtk").read_bytes()


def test_grid_3d(tmp_path: Path) -> None:
    """A 3D grid is written with ``z = -depth`` increasing and ``x`` fastest.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    x, y, depth = np.linspace(0, 0.4, 5), np.linspace(1, 1.2, 3), np.linspace(0, 0.3, 4)
    D, Y, X = np.meshgrid(depth, y, x, indexing="ij")
    grid = VelocityGrid3D(1.5 + X + 2 * Y + 3 * D, x, y, depth)
    sections = read_ascii_vtk(write_grid_vtk(tmp_path / "g.vtk", grid, "vp", binary=False))
    assert sections["DIMENSIONS"] == [5, 3, 4]
    assert sections["ORIGIN"] == pytest.approx([0.0, 1.0, -0.3])
    assert sections["SPACING"] == pytest.approx([0.1] * 3)
    Z, Y, X = np.meshgrid(-depth[::-1], y, x, indexing="ij")
    np.testing.assert_allclose(sections["vp"], (1.5 + X + 2 * Y - 3 * Z).ravel(), rtol=1e-14)


def test_mesh_2d(tmp_path: Path) -> None:
    """Triangles in ``(z, x)`` are written in ``(x, z, 0)``, with their data.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    points = np.array([[-1.0, 0.0], [-1.0, 2.0], [0.0, 2.0], [0.0, 0.0]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    lengths = mean_edge_lengths(points, triangles)
    path = write_mesh_vtk(tmp_path / "m.vtk", points, triangles,
                          point_data={"size": [0.1, 0.2, 0.3, 0.4]},
                          cell_data={"h": lengths}, binary=False)
    sections = read_ascii_vtk(path)
    np.testing.assert_array_equal(sections["POINTS"],
                                  [[0, -1, 0], [2, -1, 0], [2, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(sections["CELLS"], [[3, 0, 1, 2], [3, 0, 2, 3]])
    np.testing.assert_array_equal(sections["CELL_TYPES"], [[5], [5]])
    np.testing.assert_array_equal(sections["h"], lengths)
    np.testing.assert_array_equal(sections["size"], [0.1, 0.2, 0.3, 0.4])


def test_binary_matches_ascii(tmp_path: Path) -> None:
    """VTK reads the binary and the ASCII files alike, tetrahedra included.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy

    def read(path: Path) -> object:
        """Read a legacy VTK file with VTK's own reader.

        Parameters
        ----------
        path : pathlib.Path
            The file.

        Returns
        -------
        object
            The VTK dataset.
        """
        reader = vtk.vtkDataSetReader()
        reader.SetFileName(str(path))
        reader.Update()
        return reader.GetOutput()

    x, y, depth = np.linspace(0, 0.4, 5), np.linspace(1, 1.2, 3), np.linspace(0, 0.3, 4)
    D, Y, X = np.meshgrid(depth, y, x, indexing="ij")
    grid = VelocityGrid3D(1.5 + X + 2 * Y + 3 * D, x, y, depth)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1], [1, 1, -1]], float)
    tetrahedra = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    for binary in (True, False):
        image = read(write_grid_vtk(tmp_path / f"g{binary}.vtk", grid, "vp", binary=binary))
        assert image.GetDimensions() == (5, 3, 4)
        assert image.GetOrigin() == pytest.approx((0.0, 1.0, -0.3))
        values = vtk_to_numpy(image.GetPointData().GetArray("vp"))
        np.testing.assert_allclose(values, grid.on_grid()[1][::-1].ravel(), rtol=1e-15)
        mesh = read(write_mesh_vtk(tmp_path / f"m{binary}.vtk", points, tetrahedra,
                                   point_data={"size": np.arange(5.0)},
                                   cell_data={"id": [7.0, 8.0]}, binary=binary))
        assert mesh.GetNumberOfCells() == 2 and mesh.GetCellType(1) == 10
        np.testing.assert_array_equal(vtk_to_numpy(mesh.GetPoints().GetData()), points)
        np.testing.assert_array_equal(vtk_to_numpy(mesh.GetPointData().GetArray("size")),
                                      np.arange(5.0))
        np.testing.assert_array_equal(vtk_to_numpy(mesh.GetCellData().GetArray("id")), [7, 8])
        cell = mesh.GetCell(1)
        assert [cell.GetPointId(k) for k in range(4)] == [1, 2, 3, 4]


def test_errors(tmp_path: Path) -> None:
    """Names with spaces, arrays of the wrong length and other inputs are rejected.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    grid = VelocityGrid(np.ones((2, 2)), 0.0, 0.0, 1.0, 1.0)
    points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    triangle = np.array([[0, 1, 2]])
    with pytest.raises(ValueError):
        write_grid_vtk(tmp_path / "g.vtk", grid, "two words")
    with pytest.raises(TypeError):
        write_grid_vtk(tmp_path / "g.vtk", np.ones((2, 2)))
    for kwargs in ({"point_data": {"size": [1.0, 2.0]}}, {"cell_data": {"": [1.0]}}):
        with pytest.raises(ValueError):
            write_mesh_vtk(tmp_path / "m.vtk", points, triangle, **kwargs)
    for bad in ((points, triangle[:, :2]), (points, triangle + 1)):
        with pytest.raises(ValueError):
            write_mesh_vtk(tmp_path / "m.vtk", *bad)
