"""Size fields and meshes as VTK files, to inspect them (ParaView, VisIt).

The files are legacy VTK (format 3.0), binary by default (big-endian, as the
format requires) or ASCII, written with numpy alone. spyro's 2D ``(z, x)``
are written as ``(x, z, 0)``, so that a section stands upright, ``x`` to the
right and ``z`` up; 3D data keep ``(x, y, z)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .grids import VelocityGrid, VelocityGrid3D

#: VTK cell types of the triangle and the tetrahedron.
VTK_TRIANGLE, VTK_TETRA = 5, 10


def write_grid_vtk(path: str | Path, grid: VelocityGrid | VelocityGrid3D,
                   name: str = "value", binary: bool = True) -> Path:
    """Write the values of a grid at its nodes as VTK structured points.

    The values are those of the grid's callable at its nodes, the fixed
    layer included (a :class:`~femwavemesh.VelocityGrid` with a fixed layer
    is handed to gmsh by :func:`~femwavemesh.generate_mesh` without it).

    Parameters
    ----------
    path : str or pathlib.Path
        Output ``.vtk`` file.
    grid : VelocityGrid or VelocityGrid3D
        The grid: a velocity model, or a target size such as
        ``VelocityGrid(cpw_size(...), ...)``.
    name : str, optional
        Name of the point array, without spaces. Default ``"value"``.
    binary : bool, optional
        Binary (the default) or ASCII data.

    Returns
    -------
    pathlib.Path
        The written file.

    Raises
    ------
    TypeError
        If ``grid`` is not a grid.
    ValueError
        If ``name`` is empty or holds spaces.
    """
    _check_name(name)
    if isinstance(grid, VelocityGrid):
        values = grid.values
        if grid.fixed_layer is not None:
            depth, value = grid.fixed_layer
            z = grid.z0 + grid.dz * np.arange(values.shape[0])
            values = np.where(z[:, np.newaxis] > depth, value, values)
        # VTK's x is spyro's x (the columns), its y spyro's z (the rows).
        values = values[np.newaxis]
        origin, spacing = [grid.x0, grid.z0, 0.0], [grid.dx, grid.dz, 1.0]
    elif isinstance(grid, VelocityGrid3D):
        _, values = grid.on_grid()
        origin = [grid.x[0], grid.y[0], -grid.depth[0]]
        spacing = [grid.spacing, grid.spacing, -grid.spacing]
    else:
        raise TypeError(f"Expected a VelocityGrid or a VelocityGrid3D, not {type(grid)}.")
    # values is indexed (VTK z, y, x); VTK wants positive spacings.
    for axis in range(3):
        if spacing[axis] < 0:
            origin[axis] += spacing[axis] * (values.shape[2 - axis] - 1)
            spacing[axis] = -spacing[axis]
            values = np.flip(values, axis=2 - axis)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        _header(stream, f"femwavemesh grid {name}", binary)
        _line(stream, "DATASET STRUCTURED_POINTS")
        _line(stream, "DIMENSIONS " + " ".join(str(n) for n in values.shape[::-1]))
        _line(stream, "ORIGIN " + " ".join(repr(float(v)) for v in origin))
        _line(stream, "SPACING " + " ".join(repr(float(v)) for v in spacing))
        _line(stream, f"POINT_DATA {values.size}")
        _scalars(stream, name, values.ravel(), binary)
    return path


def write_mesh_vtk(path: str | Path, points: np.ndarray, cells: np.ndarray,
                   point_data: Mapping[str, np.ndarray] | None = None,
                   cell_data: Mapping[str, np.ndarray] | None = None,
                   binary: bool = True) -> Path:
    """Write triangles or tetrahedra, with data, as a VTK unstructured grid.

    Parameters
    ----------
    path : str or pathlib.Path
        Output ``.vtk`` file.
    points : numpy.ndarray
        Vertices: ``(n, 2)`` in spyro's ``(z, x)`` for triangles, ``(n, 3)``
        in ``(x, y, z)`` for tetrahedra (as :func:`~femwavemesh.read_mesh`
        returns them).
    cells : numpy.ndarray
        Zero-based vertex indices, ``(m, 3)`` or ``(m, 4)``.
    point_data : mapping of str to numpy.ndarray, optional
        Arrays of one value per vertex, by name (without spaces), such as
        the sizes of a :class:`~femwavemesh.MeshSize`.
    cell_data : mapping of str to numpy.ndarray, optional
        Arrays of one value per cell, by name, such as
        :func:`~femwavemesh.mean_edge_lengths`.
    binary : bool, optional
        Binary (the default) or ASCII data.

    Returns
    -------
    pathlib.Path
        The written file.

    Raises
    ------
    ValueError
        If the arrays have the wrong shapes or a name is empty or holds
        spaces.
    """
    points = np.asarray(points, float)
    cells = np.asarray(cells)
    if points.ndim != 2 or points.shape[1] not in (2, 3) or cells.ndim != 2 \
            or cells.shape[1] != points.shape[1] + 1:
        raise ValueError("Expected points (n, 2) with triangles (m, 3), "
                         "or points (n, 3) with tetrahedra (m, 4).")
    if cells.size and (cells.min() < 0 or cells.max() >= len(points)):
        raise ValueError("The cells must hold indices of the points.")
    point_data = _data(point_data, len(points), "point")
    cell_data = _data(cell_data, len(cells), "cell")
    if points.shape[1] == 2:
        xyz = np.column_stack([points[:, 1], points[:, 0], np.zeros(len(points))])
        cell_type = VTK_TRIANGLE
    else:
        xyz = points
        cell_type = VTK_TETRA
    connectivity = np.column_stack([np.full(len(cells), cells.shape[1]), cells])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        _header(stream, "femwavemesh mesh", binary)
        _line(stream, "DATASET UNSTRUCTURED_GRID")
        _line(stream, f"POINTS {len(xyz)} double")
        _values(stream, xyz, binary)
        _line(stream, f"CELLS {len(cells)} {connectivity.size}")
        _values(stream, connectivity, binary, integer=True)
        _line(stream, f"CELL_TYPES {len(cells)}")
        _values(stream, np.full((len(cells), 1), cell_type), binary, integer=True)
        for section, count, arrays in (("CELL_DATA", len(cells), cell_data),
                                       ("POINT_DATA", len(points), point_data)):
            if arrays:
                _line(stream, f"{section} {count}")
                for name, values in arrays.items():
                    _scalars(stream, name, values, binary)
    return path


def _check_name(name: str) -> None:
    """Check that an array name is a single word.

    Parameters
    ----------
    name : str
        The name.

    Raises
    ------
    ValueError
        If the name is empty or holds whitespace.
    """
    if not name or name.split() != [name]:
        raise ValueError(f"A VTK array name must be one word, not {name!r}.")


def _data(arrays: Mapping[str, np.ndarray] | None, count: int,
          kind: str) -> dict[str, np.ndarray]:
    """Check named arrays of one value per point or per cell.

    Parameters
    ----------
    arrays : mapping of str to numpy.ndarray or None
        The arrays.
    count : int
        Number of points or cells.
    kind : str
        ``"point"`` or ``"cell"``, for the message.

    Returns
    -------
    dict of str to numpy.ndarray
        The arrays as floats.

    Raises
    ------
    ValueError
        If a name is not a single word or an array has the wrong shape.
    """
    checked = {}
    for name, values in (arrays or {}).items():
        _check_name(name)
        values = np.asarray(values, float)
        if values.shape != (count,):
            raise ValueError(f"{name!r} must hold one value per {kind} ({count}).")
        checked[name] = values
    return checked


def _line(stream: BinaryIO, text: str) -> None:
    """Write a line of the header.

    Parameters
    ----------
    stream : typing.BinaryIO
        The open file.
    text : str
        The line, without its newline.
    """
    stream.write(text.encode("ascii") + b"\n")


def _header(stream: BinaryIO, title: str, binary: bool) -> None:
    """Write the three lines that open a legacy VTK file.

    Parameters
    ----------
    stream : typing.BinaryIO
        The open file.
    title : str
        The title line.
    binary : bool
        Binary or ASCII data.
    """
    _line(stream, "# vtk DataFile Version 3.0")
    _line(stream, title)
    _line(stream, "BINARY" if binary else "ASCII")


def _values(stream: BinaryIO, values: np.ndarray, binary: bool,
            integer: bool = False) -> None:
    """Write an array of data, a row per line in ASCII.

    Parameters
    ----------
    stream : typing.BinaryIO
        The open file.
    values : numpy.ndarray
        The values, two dimensions.
    binary : bool
        Big-endian binary or ASCII.
    integer : bool, optional
        32-bit integers instead of doubles. Default ``False``.

    Raises
    ------
    ValueError
        If an integer does not fit in 32 bits.
    """
    if integer and values.size and np.abs(values).max() >= 2**31:
        raise ValueError("Too many vertices for VTK's 32-bit indices.")
    if binary:
        stream.write(np.ascontiguousarray(values, ">i4" if integer else ">f8").tobytes())
        stream.write(b"\n")
    else:
        np.savetxt(stream, values, fmt="%d" if integer else "%.17g")


def _scalars(stream: BinaryIO, name: str, values: np.ndarray, binary: bool) -> None:
    """Write a named scalar array.

    Parameters
    ----------
    stream : typing.BinaryIO
        The open file.
    name : str
        Name of the array.
    values : numpy.ndarray
        One value per point or per cell.
    binary : bool
        Binary or ASCII data.
    """
    _line(stream, f"SCALARS {name} double 1")
    _line(stream, "LOOKUP_TABLE default")
    _values(stream, np.asarray(values, float).reshape(-1, 1), binary)
