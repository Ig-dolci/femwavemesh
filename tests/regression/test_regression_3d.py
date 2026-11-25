"""kmvmesh against the 3D Salt generators, frozen in ``legacy/salt3d``.

The old scripts run as the campaign runs them, as programs in a temporary
campaign layout, and kmvmesh builds the same mesh in process; the files must
have the same SHA-256 and the audits the same values. The Salt models are
not part of the repository: set ``KMVMESH_SALT3D_DATA`` to a directory with
``salt_s250_x6p9_y4p4_d2p4.npz`` and ``salt_true_x6p6_y4p1_d2p7.npz``.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import kmvmesh

pytest.importorskip("gmsh")

LEGACY = Path(__file__).parent / "legacy" / "salt3d"
S250 = "salt_s250_x6p9_y4p4_d2p4.npz"
TRUE = "salt_true_x6p6_y4p1_d2p7.npz"
DATA = Path(os.environ.get("KMVMESH_SALT3D_DATA", "/nonexistent"))
pytestmark = pytest.mark.skipif(not (DATA / S250).exists() or not (DATA / TRUE).exists(),
                                reason="set KMVMESH_SALT3D_DATA to the Salt models")
#: The box of ``fundo_salt3d.py``: window plus the 0.3 km pad.
BOX = kmvmesh.BoxDomain(x_min=6.6, x_max=8.6, y_min=4.1, y_max=6.1, z_min=-2.7, pad=0.3)
FREQUENCY = 3.5
WATER = 1.5


@pytest.fixture(scope="module")
def campaign(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Lay out a campaign with the frozen scripts and links to the models.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Temporary directory factory.

    Returns
    -------
    pathlib.Path
        The campaign root (``code/``, ``code3d/``, ``entrada/``).
    """
    root = tmp_path_factory.mktemp("campaign")
    for folder in ("code", "code3d"):
        (root / folder).mkdir()
        for script in (LEGACY / folder).glob("*.py"):
            # Copies, not links: the scripts find their data from their own path.
            (root / folder / script.name).write_bytes(script.read_bytes())
    (root / "code3d" / "data").mkdir()
    (root / "entrada").mkdir()
    for name, folder in ((S250, "code3d/data"), (TRUE, "code3d/data"), (S250, "entrada")):
        (root / folder / name).symlink_to(DATA / name)
    return root


def run_legacy(campaign: Path, *args: str) -> None:
    """Run a frozen campaign script.

    Parameters
    ----------
    campaign : pathlib.Path
        Campaign root.
    *args : str
        Script (relative to the root) and its arguments.
    """
    env = {**os.environ, "CAMP": str(campaign),
           "PYTHONPATH": os.pathsep.join([str(campaign / "code3d"), str(campaign / "code")])}
    env.pop("SALT3D_TRUE_NPZ", None)
    done = subprocess.run([sys.executable, str(campaign / args[0]), *args[1:]], env=env,
                          cwd=campaign, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def legacy_value(campaign: Path, code: str) -> dict:
    """Run statements with the frozen modules and return the arrays they make.

    Parameters
    ----------
    campaign : pathlib.Path
        Campaign root.
    code : str
        Python statements that fill a dict ``out`` of arrays.

    Returns
    -------
    dict
        The arrays of ``out``.
    """
    target = campaign / "out.npz"
    script = campaign / "probe.py"
    script.write_text("import numpy as np\nout = {}\n" + code
                      + f"\nnp.savez({str(target)!r}, **out)\n")
    run_legacy(campaign, "probe.py")
    with np.load(target) as data:
        return {key: data[key] for key in data.files}


def model(case: str) -> tuple[kmvmesh.VelocityGrid3D, kmvmesh.FacetedSurface | None]:
    """Build the velocity of a case with kmvmesh, as ``malha3d_matriz.Modelo`` does.

    Parameters
    ----------
    case : str
        ``suavizado`` or ``verdadeiro``.

    Returns
    -------
    velocity : kmvmesh.VelocityGrid3D
        The model.
    seafloor : kmvmesh.FacetedSurface or None
        The sea floor of the true model.
    """
    with np.load(DATA / S250) as data:
        x, y, depth = (np.asarray(data[k], float) for k in ("x_km", "y_km", "depth_km"))
        vp = np.asarray(data["vp"], float)
        water_depth = data["water_depth_km"].astype(float)
    if case == "suavizado":
        return kmvmesh.VelocityGrid3D(vp, x, y, depth), None
    with np.load(DATA / TRUE) as data:
        rock = np.asarray(data["vp_rocha"], float)
        sigma, spacing = float(data["fundo_sigma_km"]), float(data["fundo_passo_km"])
    seafloor = kmvmesh.FacetedSurface.from_water_depth(water_depth, x, y, BOX, sigma, spacing)
    return kmvmesh.VelocityGrid3D(rock, x, y, depth, fixed_layer=(seafloor, WATER)), seafloor


def size_field(path: Path, velocity: kmvmesh.VelocityGrid3D) -> kmvmesh.ScatteredSize:
    """Write a varying size at scattered points as the campaign's field file.

    Parameters
    ----------
    path : pathlib.Path
        ``.npz`` with ``centroids`` and ``requested_h``.
    velocity : kmvmesh.VelocityGrid3D
        Model whose CPW size the field modulates.

    Returns
    -------
    kmvmesh.ScatteredSize
        The same field for kmvmesh.
    """
    axes = (np.arange(6.6, 8.61, 0.1), np.arange(4.1, 6.11, 0.1), -np.arange(0.0, 2.71, 0.1))
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    sizes = kmvmesh.cpw_size(velocity(*points.T), 2.0, FREQUENCY)
    sizes *= 1.0 + 0.3 * np.sin(3.0 * points[:, 0]) * np.cos(2.0 * points[:, 2])
    np.savez(path, centroids=points, requested_h=sizes)
    return kmvmesh.ScatteredSize(points, sizes)


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


@pytest.mark.parametrize("case, kind", [("suavizado", "cpw"), ("verdadeiro", "cpw"),
                                        ("suavizado", "field"), ("verdadeiro", "field")])
def test_matrix_mesh(campaign: Path, tmp_path: Path, case: str, kind: str) -> None:
    """``malha3d_matriz.py``: CPW or a size field, smoothed or true model.

    Parameters
    ----------
    campaign : pathlib.Path
        Campaign root.
    tmp_path : pathlib.Path
        Output directory.
    case : str
        Model.
    kind : str
        ``cpw`` (C = 2) or ``field``.
    """
    velocity, seafloor = model(case)
    if kind == "cpw":
        size = kmvmesh.CPWSize(velocity, 2.0, FREQUENCY)
        option = ["--C", repr(2.0)]
    else:
        size = size_field(tmp_path / "field.npz", velocity)
        option = ["--campo", str(tmp_path / "field.npz")]
    old = tmp_path / "old" / "m.msh"
    run_legacy(campaign, "code3d/malha3d_matriz.py", "--caso", case, "--frequencia",
               f"{FREQUENCY:g}", "--saida", str(old), *option)
    new = kmvmesh.generate_mesh_3d(tmp_path / "new" / "m.msh", BOX, size, seafloor)
    assert digest(new) == digest(old)
    record = json.loads(old.with_suffix(".json").read_text())
    audit = kmvmesh.audit_tetrahedra(*kmvmesh.read_tetrahedra(new), seafloor)
    crossing = audit.pop("seafloor_crossing_tetrahedra", 0)
    assert audit == record["actual_geometry"]
    assert kmvmesh.count_kmv_dofs(kmvmesh.read_tetrahedra(new)[1], 3) == record["dofs"]
    assert crossing == record.get("tetraedros_que_cruzam_o_fundo", 0) == 0


@pytest.mark.parametrize("case", ["suavizado", "verdadeiro"])
def test_uniform_mesh(campaign: Path, tmp_path: Path, case: str) -> None:
    """``malhas_uniformes.py 3d``: constant size, smoothed or true model.

    Parameters
    ----------
    campaign : pathlib.Path
        Campaign root.
    tmp_path : pathlib.Path
        Output directory.
    case : str
        Model.
    """
    velocity, seafloor = model(case)
    v_min = WATER if case == "verdadeiro" else float(velocity.values.min())
    old = tmp_path / "old.msh"
    run_legacy(campaign, "code/malhas_uniformes.py", "3d", "--caso", case, "--frequencia",
               f"{FREQUENCY:g}", "--c", "1.5", "--saida", str(old))
    new = kmvmesh.generate_mesh_3d(tmp_path / "new.msh", BOX,
                                   kmvmesh.UniformSize(v_min / (1.5 * FREQUENCY)), seafloor)
    assert digest(new) == digest(old)
    record = json.loads(old.with_suffix(".json").read_text())
    audit = kmvmesh.audit_tetrahedra(*kmvmesh.read_tetrahedra(new), seafloor)
    audit.pop("seafloor_crossing_tetrahedra", None)
    assert audit == record["geometria"]


def test_models_and_laws(campaign: Path) -> None:
    """Velocity, sea floor and the 3D size law against the campaign's.

    Parameters
    ----------
    campaign : pathlib.Path
        Campaign root.
    """
    old = legacy_value(campaign, """
import campaign_math, fundo_salt3d, config_acoustic3d as cfg
from malha3d_matriz import Modelo
for case in ("suavizado", "verdadeiro"):
    m = Modelo(case)
    out[case + "_points"], out[case + "_v"] = m.grade()
    out[case + "_extremes"] = np.array(m.extremos())
    out[case + "_probe"] = m(np.array([6.61, 7.3, 8.59]), np.array([4.2, 5.0, 6.0]),
                             np.array([-0.05, -1.0, -2.69]))
s = fundo_salt3d.superficie(cfg.MODEL_NPZ, 0.1, 0.1)
out["x"], out["y"], out["d"] = s.x, s.y, s.d
rng = np.random.default_rng(3)
h = rng.uniform(0.05, 0.2, (5, 10, 10)); rho = rng.uniform(0.1, 2.0, h.shape)
vp = h * 1.25 * 2.0 * 5 * rng.uniform(0.5, 1.0, h.shape)
out["h"], out["rho"], out["vp"] = h, rho, vp
out["law"] = campaign_math.size_law(0.8, h, rho, vp, 6.0, 2.0, 5)
# matriz3d.malha_adjunta's log form of the same law (q = 6, C_eq = 2, f = 5).
q, hmin = 6.0, np.minimum(vp / (1.25 * 2.0 * 5), h)
forma = q / (q + 3.0) * np.log(h) - np.log(rho) / (q + 3.0)
out["log_law"] = np.clip(np.exp(np.clip(np.log(0.8) + forma, -700, 700)), hmin, h)
""")
    for case in ("suavizado", "verdadeiro"):
        velocity, seafloor = model(case)
        points, values = velocity.on_grid()
        np.testing.assert_array_equal(points, old[case + "_points"])
        np.testing.assert_array_equal(values, old[case + "_v"])
        assert velocity.extremes() == tuple(old[case + "_extremes"])
        np.testing.assert_array_equal(
            velocity(np.array([6.61, 7.3, 8.59]), np.array([4.2, 5.0, 6.0]),
                     np.array([-0.05, -1.0, -2.69])), old[case + "_probe"])
    with np.load(DATA / S250) as data:
        surface = kmvmesh.FacetedSurface.from_water_depth(
            data["water_depth_km"].astype(float), data["x_km"].astype(float),
            data["y_km"].astype(float), BOX, 0.1, 0.1)
    for mine, theirs in ((surface.x, old["x"]), (surface.y, old["y"]),
                         (surface.depth, old["d"])):
        np.testing.assert_array_equal(mine, theirs)
    # The same law as the campaigns', to rounding: v4 multiplies the factors in
    # another order, matriz3d evaluates it in log form.
    law = kmvmesh.SizeLaw(old["rho"], old["h"], 0.02, order=6.0)
    law.set_floor(old["vp"], 5, 2.0, 1.25)
    np.testing.assert_allclose(law.size(0.8), old["law"], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(law.size(0.8), old["log_law"], rtol=1e-14, atol=0.0)
