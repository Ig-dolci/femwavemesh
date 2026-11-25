# kmvmesh

Unstructured meshes and mesh-size laws for wave solvers with KMV
(Kong-Mulder-Veldhuizen) mass-lumped elements.

kmvmesh writes gmsh meshes of a padded rectangle (triangles) or a padded box
(tetrahedra), conforming to horizontal interfaces in 2D and to a faceted sea
floor in 3D, from a target edge length:

* uniform cells per wavelength, `h = v / (C f)`;
* an adaptive law of a refinement density, `h = clip(alpha h_k^(q/(q+d))
  rho^(-1/(q+d)), h_min, h_max)`, with the scale `alpha` fitted to a budget of
  degrees of freedom;
* refinement around point sources and smooth perturbations for mesh
  replicas.

It reads the meshes back, samples their local size on a grid, counts their
KMV degrees of freedom and audits their geometry and conformity. It depends
on numpy, scipy, gmsh and FIAT only: the solver (spyro, with Firedrake) loads the
`.msh` file, and the refinement density comes from the solver too.

The code was extracted from spyro's `spyro.adaptivity` and from the 3D Salt
campaigns. For the same inputs it writes the same meshes, byte for byte
(see [Tests](#tests)).

## Installation

```bash
python3 -m pip install -e /path/to/kmvmesh        # numpy, scipy, gmsh, firedrake-fiat
python3 -m pip install -e "/path/to/kmvmesh[dev]"  # plus pytest, flake8, ruff
```

`import kmvmesh` imports neither gmsh nor FIAT: the functions that mesh or
read meshes load gmsh, and the KMV counts (and the audits that report them)
load FIAT.

## Conventions

* 2D meshes are in spyro's coordinates: the first coordinate is the depth `z`
  (negative downwards, `z = 0` at the surface), the second is `x`.
* 3D meshes are in `(x, y, z)`, `z` negative downwards.
* Boundary markers (physical groups): `TOP = 1`, `BOTTOM = 2`, `X_MIN = 3`,
  `X_MAX = 4`, `Y_MIN = 5`, `Y_MAX = 6`; the faces of a sea floor are
  `SEAFLOOR = 7`; the cells are the group `1`.
* Units are km, km/s and Hz.
* gmsh format 2.2 (binary in 3D). gmsh runs in the calling process: under MPI,
  generate on one rank and share the file name.

## Examples

A 2D mesh with 4 cells per wavelength and a sea floor at 0.45 km:

```python
import numpy as np
import kmvmesh

domain = kmvmesh.RectangularDomain(z_min=-2.0, x_min=0.0, x_max=4.0, pad=0.3)
z, x = domain.grid(0.005)
Z, X = np.meshgrid(z, x, indexing="ij")
velocity = np.where(Z > -0.45, 1.5, 2.0 - 0.3 * Z)
size = kmvmesh.VelocityGrid(kmvmesh.cpw_size(velocity, 4.0, 5.0), z[0], x[0], 0.005, 0.005)
path = kmvmesh.generate_mesh("cpw4.msh", domain, size, conforming_depths=[-0.45])
print(kmvmesh.audit_mesh(path, degree=4, conforming_depths=[-0.45]))
```

An adaptive mesh with a budget, from the current mesh and a density:

```python
# density: refinement density on the same grid, from the solver.
points, triangles = kmvmesh.read_triangulation(path)
current = kmvmesh.local_mesh_size(points, triangles, z, x)
field = kmvmesh.AdaptiveSizeField(density, current, z, x, order=2.7)
field.set_floor(velocity, 5.0, equivalent_cpw=5.0)
calibration = kmvmesh.dof_calibration(kmvmesh.count_kmv_dofs(triangles, 4), current, 0.005)

def build(law):
    kmvmesh.generate_mesh("adaptive.msh", domain, law.grid, [-0.45])
    return kmvmesh.count_kmv_dofs(kmvmesh.read_triangulation("adaptive.msh")[1], 4)

dofs = kmvmesh.fit_to_budget(field, target=200_000, calibration=calibration, build=build)
```

A 3D mesh of a box split by a sea floor, 2 cells per wavelength:

```python
box = kmvmesh.BoxDomain(x_min=6.6, x_max=8.6, y_min=4.1, y_max=6.1, z_min=-2.7, pad=0.3)
floor = kmvmesh.FacetedSurface.from_water_depth(water_depth, xs, ys, box, sigma=0.1, spacing=0.1)
velocity = kmvmesh.VelocityGrid3D(vp, xs, ys, depths, fixed_layer=(floor, 1.5))
path = kmvmesh.generate_mesh_3d("salt.msh", box, kmvmesh.CPWSize(velocity, 2.0, 3.5), floor)
print(kmvmesh.audit_tetrahedra(*kmvmesh.read_tetrahedra(path), floor))
```

## API

| name | what it is |
|---|---|
| `RectangularDomain` | 2D padded rectangle in `(z, x)`; `in_interior`, `grid` |
| `BoxDomain` | 3D padded box in `(x, y, z)`; `in_interior` |
| `FacetedSurface` | surface of planar triangles (sea floor); `from_water_depth`, `depth_at`, `side` |
| `TOP`, `BOTTOM`, `X_MIN`, `X_MAX`, `Y_MIN`, `Y_MAX`, `SEAFLOOR` | boundary markers |
| `VelocityGrid` | 2D bilinear sampler on a `(z, x)` grid, optional fixed layer; the size format of `generate_mesh` |
| `VelocityGrid3D` | 3D trilinear sampler on a `(depth, y, x)` grid, optional layer over a surface; `on_grid`, `extremes` |
| `cpw_size` | `h = v / (C f)` |
| `size_perturbation` | smooth random factor of a size, for replicas |
| `source_refinement` | size multiplier that refines around point sources |
| `SizeLaw` | adaptive law of a density in any dimension: `set_floor`, `size`, `relative_cost`, `scale_for`, `set_scale` |
| `AdaptiveSizeField` | `SizeLaw` on a 2D `(z, x)` grid, with the gmsh-ready `grid` |
| `dof_calibration` | DOFs per unit of `sum(h^-d) dV` of a mesh |
| `fit_to_budget` | rescale a law until its mesh meets a DOF target |
| `UniformSize`, `CPWSize`, `ScatteredSize` | 3D point sizes for gmsh's size callback |
| `generate_mesh` | 2D triangles from a size grid, with horizontal interfaces |
| `generate_mesh_3d` | 3D tetrahedra (HXT) of a box, optionally split by a sea floor |
| `read_mesh`, `read_triangulation`, `read_tetrahedra` | vertices and cells of a mesh file |
| `count_kmv_dofs` | KMV DOFs from FIAT's element, degrees 1 to 6 on triangles, 1 to 3 on tetrahedra |
| `mean_edge_lengths`, `local_mesh_size` | element size, and its sampling on a grid |
| `audit_triangles`, `audit_tetrahedra`, `audit_mesh` | counts, measures, edges, orientation, interface conformity; `audit_mesh` adds the file's SHA-256 and the KMV budget |

## Tests

```bash
make lint
make test
```

`tests/test_*.py` test each module (the generation tests need gmsh).
`tests/test_firedrake.py` loads the meshes in Firedrake and checks what the
solver sees: the dimension of the KMV spaces against `count_kmv_dofs`, the
boundary markers, and the interfaces as mesh facets (`dS(SEAFLOOR)` in 3D).
It is skipped where Firedrake is not installed; to run it, use the Python of
a Firedrake installation:

```bash
/path/to/venv-firedrake/bin/python -m pytest tests/test_firedrake.py -v
```

`tests/regression/` compares kmvmesh with frozen copies of the code it
replaces (`tests/regression/legacy/`): spyro's 2D generator and size laws,
the CPW refinement near sources, and the 3D Salt generators. Each case asks
for the same `.msh` (SHA-256) and the same arrays. The old local size needs
matplotlib, as it did; the 3D cases need the Salt models:

```bash
KMVMESH_SALT3D_DATA=/path/to/data make test-regression
```

with `salt_s250_x6p9_y4p4_d2p4.npz` and `salt_true_x6p6_y4p1_d2p7.npz` in
that directory. Cases whose requirements are missing are skipped.

## License

kmvmesh is distributed under the GNU Lesser General Public License v3 or
later (LGPLv3+), the license of [spyro](https://github.com/NDF-Poli-USP/spyro),
from which most of its code comes. See `LICENSE`.
# kmvmesh
