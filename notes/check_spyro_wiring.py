r"""Compare spyro's mesh-building methods before and after kmvmesh, without Firedrake.

The old and the new ``spyro/adaptivity`` modules are loaded side by side as
packages ``old_spyro`` and ``new_spyro``, with stubs for Firedrake, mpi4py
and the solver modules the mesh methods do not use. The methods run for real
(gmsh included) and the script compares the files they write (SHA-256) and
the DOF counts they return:

* ``AutomaticMeshAdaptation.cpw_mesh`` and ``adaptive_mesh``, seeds 0 and 1;
* ``SourceRefinedCPW.cpw_mesh`` of ``cpw_fontes.py`` (``--cpw-fontes``);
* ``MeshAdaptiveFWI._cpw_mesh`` (with and without a cap) and
  ``_designed_mesh`` (``cpw`` and ``gradient``, with and without the unified
  rule);
* the calibration expression that ``dof_calibration`` replaces.

Usage::

    python check_spyro_wiring.py OLD_SPYRO_ROOT NEW_SPYRO_ROOT OUTPUT_DIR \
        [--cpw-fontes path/to/cpw_fontes.py]

The old modules need matplotlib (local size) and FIAT (KMV count) importable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np


class _Comm:
    """Serial communicator: rank 0, broadcast is the identity."""

    rank = 0
    size = 1

    def bcast(self, value: object, root: int = 0) -> object:
        """Return the value.

        Parameters
        ----------
        value : object
            Value to broadcast.
        root : int, optional
            Ignored.

        Returns
        -------
        object
            ``value``.
        """
        return value


def _stub(name: str, **attributes: object) -> types.ModuleType:
    """Register a stub module.

    Parameters
    ----------
    name : str
        Full module name.
    **attributes : object
        Its attributes.

    Returns
    -------
    types.ModuleType
        The module.
    """
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def _dummy(name: str) -> type:
    """Make a placeholder class.

    Parameters
    ----------
    name : str
        Class name.

    Returns
    -------
    type
        An empty class.
    """
    return type(name, (), {})


def load_spyro(alias: str, root: Path) -> dict:
    """Load spyro's loops under a package alias, with stubs around them.

    Parameters
    ----------
    alias : str
        Package name, ``old_spyro`` or ``new_spyro``.
    root : pathlib.Path
        spyro checkout (the folder with ``spyro/adaptivity``).

    Returns
    -------
    dict
        The modules ``automatic_loop``, ``fwi_loop``, ``mesh_generation`` and
        ``size_fields``.
    """
    package = _stub(alias)
    package.__path__ = [str(root / "spyro")]
    adaptivity = _stub(f"{alias}.adaptivity")
    adaptivity.__path__ = [str(root / "spyro" / "adaptivity")]
    tools = _stub(f"{alias}.tools")
    tools.__path__ = []
    _stub(f"{alias}.tools.optimization", minimize_with_tao=lambda *a, **k: None)
    _stub(f"{alias}.adaptivity._firedrake_workarounds", speed_up_unique_names=lambda: None)
    _stub(f"{alias}.adaptivity.paraview", write_pvd=None, write_vti=None)
    _stub(f"{alias}.adaptivity.pilot", PilotResult=_dummy("PilotResult"),
          TwoLevelPilot=_dummy("TwoLevelPilot"), Window=_dummy("Window"))
    _stub(f"{alias}.adaptivity.fwi_functional",
          **{name: _dummy(name) for name in ("EnsembleFWIFunctional", "FWIFunctional",
                                             "ObservedData", "ParameterGrid",
                                             "TraceOperator")})
    return {name: importlib.import_module(f"{alias}.adaptivity.{name}")
            for name in ("mesh_generation", "size_fields", "automatic_loop", "fwi_loop")}


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


BOX = dict(z_min=-1.0, x_min=0.0, x_max=1.2, pad=0.2)
INTERFACE = -0.3


def model(z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Two layers with a lateral gradient below the interface.

    Parameters
    ----------
    z : numpy.ndarray
        Depths.
    x : numpy.ndarray
        Positions.

    Returns
    -------
    numpy.ndarray
        Velocity (km/s).
    """
    return np.where(z - 1e-9 > INTERFACE, 1.5, 2.5 + 0.5 * x - 0.4 * z)


def automatic(modules: dict, output: Path, seed: int, strategy: str) -> object:
    """Build an ``AutomaticMeshAdaptation`` of the two-layer problem.

    Parameters
    ----------
    modules : dict
        The loaded modules.
    output : pathlib.Path
        Output directory of the loop.
    seed : int
        Replica.
    strategy : str
        Strategy name.

    Returns
    -------
    AutomaticMeshAdaptation
        The loop, ready to build meshes.
    """
    mg, sf, al = modules["mesh_generation"], modules["size_fields"], modules["automatic_loop"]
    domain = mg.RectangularDomain(**BOX)
    z, x = domain.grid(0.01)
    Z, X = np.meshgrid(z, x, indexing="ij")
    velocity = sf.VelocityGrid(model(Z, X), z[0], x[0], 0.01, 0.01)
    parameters = al.LoopParameters(tolerance=0.05, grid_spacing=0.01, seed=seed)
    return al.AutomaticMeshAdaptation(strategy, None, velocity, domain, 5.0, 4, output,
                                      parameters, [INTERFACE], comm=_Comm(), components=2)


def fwi(modules: dict, output: Path, strategy: str, unified: bool) -> object:
    """Build a ``MeshAdaptiveFWI`` with only what its mesh methods use.

    Parameters
    ----------
    modules : dict
        The loaded modules.
    output : pathlib.Path
        Output directory.
    strategy : str
        ``cpw`` or ``gradient``.
    unified : bool
        The unified rule.

    Returns
    -------
    MeshAdaptiveFWI
        The driver, built without its constructor.
    """
    mg, sf, fl = modules["mesh_generation"], modules["size_fields"], modules["fwi_loop"]
    driver = object.__new__(fl.MeshAdaptiveFWI)
    driver.policy = fl.MeshPolicy(noise_level=0.1, strategy=strategy, grid_spacing=0.01,
                                  unified_rule=unified)
    driver.domain = mg.RectangularDomain(**BOX)
    driver.z, driver.x = driver.domain.grid(0.01)
    driver._Z, driver._X = np.meshgrid(driver.z, driver.x, indexing="ij")
    driver.interior = driver.domain.in_interior(driver._Z, driver._X)
    grid = sf.VelocityGrid(model(driver._Z, driver._X), driver.z[0], driver.x[0], 0.01, 0.01)
    driver.grid = types.SimpleNamespace(sampler=lambda values: grid)
    driver.model = None
    driver.comm, driver.ensemble, driver._root = _Comm(), None, True
    driver.output_dir, driver._meshes = Path(output), 0
    driver.conforming_depths, driver.degree = (INTERFACE,), 4
    driver._cpw_calibration, driver._design, driver.mesh, driver.dofs = None, None, None, 0
    return driver


def main() -> None:
    """Run every comparison and print a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpw-fontes", type=Path)
    args = parser.parse_args()
    _stub("firedrake", COMM_WORLD=_Comm())
    _stub("mpi4py").MPI = _stub("mpi4py.MPI", Comm=_Comm)
    loaded = {"old": load_spyro("old_spyro", args.old), "new": load_spyro("new_spyro", args.new)}
    rows = []

    def compare(case: str, results: dict) -> None:
        """Record whether the old and the new results agree.

        Parameters
        ----------
        case : str
            Name of the case.
        results : dict
            ``{"old": (path, dofs, ...), "new": (...)}``.
        """
        old, new = results["old"], results["new"]
        row = {"case": case, "old_sha256": digest(old[0]), "new_sha256": digest(new[0]),
               "old_dofs": old[1], "new_dofs": new[1]}
        row["same"] = row["old_sha256"] == row["new_sha256"] and old[1:] == new[1:]
        rows.append(row)
        print(json.dumps(row), flush=True)

    # AutomaticMeshAdaptation: CPW, then an adaptive mesh from it.
    for seed in (0, 1):
        results = {}
        for side, modules in loaded.items():
            loop = automatic(modules, args.output / side / f"auto_s{seed}", seed, "gradient")
            path, dofs = loop.cpw_mesh(3.0)
            density = modules["size_fields"].gradient_density(
                loop.grid_velocity, 0.01, loop.smoothing, loop.interior)
            results[side] = (path, dofs, loop.adaptive_mesh(
                f"gradient_it1_s{seed}", density, path, int(dofs * 1.4),
                3.0 * math.sqrt(1.4)))
        compare(f"automatic_loop.cpw_mesh seed {seed}", {k: v[:2] for k, v in results.items()})
        compare(f"automatic_loop.adaptive_mesh seed {seed}",
                {k: v[2] for k, v in results.items()})

    # SourceRefinedCPW of the campaign, against each spyro.
    if args.cpw_fontes is not None:
        _stub("spyro.mpi")
        _stub("spyro.mpi.spyro_mpi", SpyroEnsemble=None)
        _stub("marmousi_automatic_loop")
        results = {}
        for side, modules in loaded.items():
            sys.modules["spyro"] = types.ModuleType("spyro")
            sys.modules["spyro.adaptivity"] = types.ModuleType("spyro.adaptivity")
            for name in ("automatic_loop", "mesh_generation", "size_fields"):
                sys.modules[f"spyro.adaptivity.{name}"] = modules[name]
            spec = importlib.util.spec_from_file_location(f"cpw_fontes_{side}", args.cpw_fontes)
            fontes = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fontes)
            base = automatic(modules, args.output / side / "fontes", 1, "cpw")
            loop = fontes.SourceRefinedCPW(
                "cpw", None, base.velocity, base.domain, 5.0, 4, args.output / side / "fontes",
                base.parameters, [INTERFACE], comm=_Comm(), components=2,
                sources=[(-0.05, 0.4), (-0.05, 0.8)], radius=0.25, factor=2.0, taper=0.1)
            results[side] = loop.cpw_mesh(2.0)
        compare("cpw_fontes.SourceRefinedCPW.cpw_mesh seed 1", results)

    # MeshAdaptiveFWI: uniform meshes, then designed meshes.
    for strategy in ("cpw", "gradient"):
        for unified in (False, True):
            results = {}
            for side, modules in loaded.items():
                driver = fwi(modules, args.output / side / f"fwi_{strategy}_{unified}",
                             strategy, unified)
                first = driver._cpw_mesh(3.0, 5.0)
                cap = 0.9 * driver._mesh_size(first[0])[0]
                capped = driver._cpw_mesh(2.5, 5.0, cap=cap)
                driver.mesh, driver.dofs = first
                size, dofs = driver._mesh_size(first[0])
                velocity = driver._velocity()
                density = modules["size_fields"].gradient_density(
                    velocity, 0.01, driver._smoothing(5.0), driver.interior)
                driver._design = {"density": density, "size": size,
                                  "calibration": dofs / float((size ** -2).sum() * 0.01 ** 2)}
                designed = driver._designed_mesh("refine", int(dofs * 1.5), 5.0,
                                                 0.95 * size)
                results[side] = (first, capped, designed, driver._cpw_calibration)
            name = f"fwi_loop {strategy} unified={unified}"
            if strategy == "cpw" and not unified:
                compare("fwi_loop._cpw_mesh", {k: v[0] for k, v in results.items()})
                compare("fwi_loop._cpw_mesh with cap", {k: v[1] for k, v in results.items()})
            compare(f"{name}: _designed_mesh", {k: v[2] for k, v in results.items()})
            assert results["old"][3] == results["new"][3]

    # The calibration expression that dof_calibration replaces.
    new = loaded["new"]["fwi_loop"]
    mesh = args.output / "new" / "auto_s0" / "meshes" / "cpw_C3_s0.msh"
    size = new.local_mesh_size(*new.read_triangulation(mesh),
                               *new.RectangularDomain(**BOX).grid(0.01))
    same = (new.dof_calibration(12345, size, 0.01)
            == 12345 / float((size ** -2).sum() * 0.01 ** 2))
    rows.append({"case": "dof_calibration == dofs / float((h ** -2).sum() * s ** 2)",
                 "same": bool(same)})
    print(json.dumps(rows[-1]))
    (args.output / "summary.json").write_text(json.dumps(rows, indent=1))
    print("ALL SAME" if all(row["same"] for row in rows) else "DIFFERENCES")


if __name__ == "__main__":
    main()
