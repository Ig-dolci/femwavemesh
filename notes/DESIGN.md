# kmvmesh: design

`kmvmesh` generates the unstructured meshes of the KMV wave solvers (spyro and
the 3D Salt campaigns) and holds the mesh-size laws that drive them. It
depends on numpy, scipy, gmsh and FIAT only. It returns a mesh as a gmsh `.msh` file
plus, on request, a metadata dictionary; loading the file (Firedrake) is the
caller's job.

The overriding requirement is preserved behaviour: for the same inputs the
library writes the same mesh as the code it replaces (same SHA-256 of the
`.msh`). Every function below was moved, not rewritten, unless noted.

## Modules and public API

| module | name | origin |
|---|---|---|
| `domain` | `TOP, BOTTOM, X_MIN, X_MAX, Y_MIN, Y_MAX, SEAFLOOR` (1..7) | spyro `mesh_generation`; campaign `malhas_uniformes` |
| | `RectangularDomain` 2D padded rectangle, spyro `(z, x)` | spyro `mesh_generation` |
| | `BoxDomain` 3D padded box, `(x, y, z)` with `z` up | campaign `fundo_salt3d` / `config_acoustic3d` constants |
| | `FacetedSurface` sea floor by planar facets (`from_water_depth`) | campaign `fundo_salt3d.Superficie`, `superficie` |
| `grids` | `VelocityGrid` 2D bilinear sampler, `fixed_layer` | spyro `size_fields` |
| | `VelocityGrid3D` 3D trilinear sampler on `(depth, y, x)`, water layer | campaign `malha3d_matriz.Modelo` |
| `sizing` | `cpw_size(v, C, f)` = `v / (C f)` | spyro loops, campaign generators |
| | `size_perturbation` smooth random factor for replicas | spyro `size_fields` |
| | `source_refinement` size divisor near point sources | `cpw_fontes.source_scale` |
| | `SizeLaw` adaptive law in dimension `d` (exponent `q/(q+d)`, floor `v/(c_max C_eq f)`, ceiling) | spyro `AdaptiveSizeField`, generalized |
| | `AdaptiveSizeField(SizeLaw)` the same law on a 2D `(z, x)` grid, with the gmsh grid | spyro `size_fields` |
| | `dof_calibration` DOFs per unit of `sum(h^-d) dV` | spyro loops (4 copies) |
| | `fit_to_budget` scale of a law until a mesh meets a DOF target | spyro `automatic_loop.adaptive_mesh`, `fwi_loop._designed_mesh` |
| | `UniformSize`, `CPWSize`, `ScatteredSize` 3D point sizes for gmsh's callback | `malhas_uniformes.malha_3d`, `malha3d_matriz.gerar`, `make_mesh3d` |
| `generate` | `generate_mesh` 2D triangles, structured background size, horizontal interfaces | spyro `mesh_generation` |
| | `generate_mesh_3d` 3D tetrahedra (HXT), box or box split by a faceted sea floor | `malha3d_matriz.gerar`, `malhas_uniformes._caixa`, `_caixa_com_fundo` |
| `io` | `read_triangulation`, `read_tetrahedra`, `read_mesh` (either, by the file's dimension) | spyro `mesh_generation`; new (3D) |
| `counting` | `count_kmv_dofs` KMV 1..6 on triangles, 1..3 on tetrahedra (DOFs per entity from `FIAT.KongMulderVeldhuizen`) | spyro `mesh_generation`; v4 `mesh_geometry` (KMV3) |
| | `mean_edge_lengths`, `local_mesh_size` (with `counting.find_triangles`, see below) | spyro `mesh_generation` |
| `audit` | `audit_triangles`, `audit_tetrahedra`, `audit_mesh` geometry, entity counts, DOFs, interface conformity | `malhas_uniformes.malha_2d`; v4 `mesh_geometry.audit_tetrahedra` |

## What changes in spyro

* `spyro/adaptivity/mesh_generation.py` becomes a re-export of the kmvmesh
  names it used to define (`RectangularDomain`, `generate_mesh`,
  `read_triangulation`, `mean_edge_lengths`, `local_mesh_size`,
  `count_kmv_dofs`, `TOP`, `BOTTOM`, `X_MIN`, `X_MAX`).
* `spyro/adaptivity/size_fields.py` keeps the densities (`gradient_density`,
  `adjoint_density`) and re-exports `VelocityGrid`, `AdaptiveSizeField`,
  `size_perturbation`.
* `spyro.adaptivity` exports the same names as before.
* `automatic_loop` and `fwi_loop` import the mesh names from kmvmesh (the
  module-level names stay, so subclasses and monkeypatches keep working) and
  use `cpw_size`, `dof_calibration` and `fit_to_budget` instead of their own
  copies of the CPW law, the calibration and the rescaling loop.
* `fwi_functional` imports `RectangularDomain` and `VelocityGrid` from kmvmesh.
* kmvmesh is a dependency in `pyproject.toml`.

## What stays in spyro

The indicator and the pilot, the densities, the loops and their policies
(growth, stopping, the uniform-mesh rescale of `fwi_loop`), MPI (meshes are
built on rank 0), mesh names and reports, and everything Firedrake.

## What stays in the campaigns

Model file formats (npz keys), the density from indicator files, the
resumable trial records, and the 3D campaigns' own searches for the scale
(`matriz3d.malha_adjunta`: calibrated bisection with a correction factor and
12 trials; v4 `phase3_meshes.tune_mesh`: bracketed search). kmvmesh offers
`fit_to_budget` (spyro's 2D search, generic in `d`) instead.

## Point location without matplotlib

spyro's `local_mesh_size` located grid points with matplotlib's
`TrapezoidMapTriFinder`. kmvmesh does it with numpy (candidate triangles by
bounding box, orientation tests in the trifinder's canonical edge direction)
and reproduces the trifinder's ties: a point on an edge takes the triangle
"above" the edge in the trifinder's frame, a point on a vertex the
lowest-index triangle around it. The regression tests compare both on the
test meshes, including grid points on the interface and on the boundary.

## Known numerical differences (documented, tested to rounding)

* The 3D law is evaluated as `alpha * (h^(q/(q+3)) rho^(-1/(q+3)))`, the form
  of spyro's 2D law. `matriz3d.malha_adjunta` evaluates the same law in log
  form and `campaign_math.size_law` multiplies in another order; the sizes
  agree to a few ulp, not bit for bit.
* `ScatteredSize` queries `min(4, n)` neighbours (`malha3d_matriz` queried 4,
  which fails for fewer than 4 points); identical for `n >= 4`.

## Tests

* `tests/test_*.py`: unit tests of every module (gmsh needed for the
  generation tests).
* `tests/test_firedrake.py`: the meshes loaded in Firedrake (KMV dimensions,
  markers, interfaces as facets); Firedrake is a test requirement only, and
  the file is skipped without it.
* `tests/regression/`: the library against frozen copies of the code it
  replaces (`tests/regression/legacy/`), comparing SHA-256 of the `.msh` and
  the node and element arrays. The 3D cases need the Salt models
  (`KMVMESH_SALT3D_DATA`) and the legacy local size needs matplotlib; each
  case is skipped when its requirement is missing.
