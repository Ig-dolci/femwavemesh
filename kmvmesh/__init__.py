"""Unstructured meshes and mesh-size laws for finite-element wave solvers.

kmvmesh writes gmsh meshes of a padded rectangle (triangles, spyro's
``(z, x)`` coordinates) or of a padded box (tetrahedra, ``(x, y, z)``),
conforming to horizontal interfaces in 2D and to a faceted sea floor in 3D,
from a target edge length: uniform cells per wavelength, an adaptive law of a
refinement density with a degree-of-freedom budget, refinement around point
sources and smooth perturbations for replicas. It reads the meshes back,
samples their local size, counts their degrees of freedom for any FIAT
element (the mass-lumped KMV elements among them) and audits their geometry
and conformity. It depends on numpy, scipy, gmsh and FIAT only.
"""

from .audit import audit_mesh, audit_tetrahedra, audit_triangles
from .counting import count_dofs, count_kmv_dofs, local_mesh_size, mean_edge_lengths
from .domain import (BOTTOM, SEAFLOOR, TOP, X_MAX, X_MIN, Y_MAX, Y_MIN,
                     BoxDomain, FacetedSurface, RectangularDomain)
from .generate import generate_mesh, generate_mesh_3d
from .grids import VelocityGrid, VelocityGrid3D
from .io import read_mesh, read_tetrahedra, read_triangulation
from .sizing import (AdaptiveSizeField, CPWSize, ScatteredSize, SizeLaw,
                     UniformSize, cpw_size, dof_calibration, fit_to_budget,
                     size_perturbation, source_refinement)

__version__ = "0.1.0"

__all__ = [
    "TOP",
    "BOTTOM",
    "X_MIN",
    "X_MAX",
    "Y_MIN",
    "Y_MAX",
    "SEAFLOOR",
    "RectangularDomain",
    "BoxDomain",
    "FacetedSurface",
    "VelocityGrid",
    "VelocityGrid3D",
    "cpw_size",
    "size_perturbation",
    "source_refinement",
    "SizeLaw",
    "AdaptiveSizeField",
    "dof_calibration",
    "fit_to_budget",
    "UniformSize",
    "CPWSize",
    "ScatteredSize",
    "generate_mesh",
    "generate_mesh_3d",
    "read_mesh",
    "read_triangulation",
    "read_tetrahedra",
    "count_dofs",
    "count_kmv_dofs",
    "mean_edge_lengths",
    "local_mesh_size",
    "audit_triangles",
    "audit_tetrahedra",
    "audit_mesh",
]
