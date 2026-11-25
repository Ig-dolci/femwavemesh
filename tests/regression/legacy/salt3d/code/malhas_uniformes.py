"""Malhas uniformes conformes das referências da matriz (P4): triângulos no 2D, tetraedros no 3D.

"Uniforme" = tamanho-alvo constante h = v_min/(C·f) em todo o domínio (com a camada), v_min o v_p mínimo no acústico e
o v_s mínimo no elástico. O gmsh gera uma malha não estruturada com esse tamanho.

- 2D: o gerador do spyro (``generate_mesh``), com o mesmo domínio, as mesmas interfaces conformes (fundo do mar e,
  no acústico verdadeiro, a discordância) e as mesmas marcas de contorno das malhas do laço; só o campo de tamanho é
  constante.
- 3D suavizado: a caixa do ``make_mesh3d.py`` (mesmas marcas 1–6), tamanho constante.
- 3D verdadeiro: a caixa dividida pelo fundo do mar de ``fundo_salt3d.py`` (facetas planas), água em cima e rocha
  embaixo; as faces do fundo têm a marca 7; os dois volumes ficam na marca 1.

Cada malha sai com um ``.json``: h, C, f, GL (KMV4 no 2D, com as componentes; KMV3 no 3D, escalar), arestas (mín.,
média, máx.), conformidade (nenhum elemento atravessa uma interface), tempo e memória de pico do gerador.

Uso (2D, a partir de spyro/demos/automatic_mesh_adaptation)::

    python malhas_uniformes.py 2d --caso acustico_s250 --frequencia 3.5 --c 6 --saida M.msh
    python malhas_uniformes.py 3d --caso suavizado|verdadeiro --frequencia 3.5 --c 8 --saida M.msh [--threads N]

Preparado com assistência do Claude (Anthropic), a pedido da pesquisadora.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

import config as cfg

VP_MIN_3D = {"suavizado": None, "verdadeiro": 1.5}   # suavizado: lido do modelo s250


def _pico_memoria_gib() -> float:
    """Memória residente de pico do processo (GiB; Linux em KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def malha_2d(caso: str, frequencia: float, c: float, saida: Path, semente: int = 0) -> dict:
    """Gera uma malha uniforme 2D conforme.

    Parameters
    ----------
    caso : str
        Caso da matriz (``config.problema_matriz``).
    frequencia : float
        Frequência (Hz).
    c : float
        Células por comprimento de onda na velocidade mínima.
    saida : pathlib.Path
        Arquivo ``.msh``.
    semente : int, optional
        Semente do gmsh. Default 0.

    Returns
    -------
    dict
        Metadados da malha.
    """
    import marmousi_automatic_loop as mal
    from spyro.adaptivity.mesh_generation import count_kmv_dofs, generate_mesh, read_triangulation
    from spyro.adaptivity.size_fields import VelocityGrid

    parser = argparse.ArgumentParser()
    mal.add_problem_arguments(parser)
    args = parser.parse_args(cfg.problema_matriz(caso, frequencia, 2.0))
    problem = mal.build_problem(args)
    passo = 0.005
    z, x = problem.domain.grid(passo)
    Z, X = np.meshgrid(z, x, indexing="ij")
    v_min = float(np.min(problem.velocity(Z, X)))
    h = v_min / (c * frequencia)
    t0 = time.perf_counter()
    generate_mesh(saida, problem.domain, VelocityGrid(np.full(Z.shape, h), z[0], x[0], passo, passo),
                  problem.conforming_depths, seed=semente)
    segundos = time.perf_counter() - t0
    pontos, tri = read_triangulation(saida)
    arestas = np.concatenate([np.linalg.norm(pontos[tri[:, a]] - pontos[tri[:, b]], axis=1)
                              for a, b in ((0, 1), (1, 2), (2, 0))])
    cruzam = {}
    for prof in problem.conforming_depths:
        zt = pontos[tri, 0]
        cruzam[str(prof)] = int(np.sum((zt.min(axis=1) < prof - 1e-9) & (zt.max(axis=1) > prof + 1e-9)))
    meta = {"dimensao": 2, "caso": caso, "frequencia_hz": frequencia, "C": c, "v_min_km_s": v_min, "h_km": h,
            "semente": semente, "triangulos": int(len(tri)),
            "gl": int(problem.components * count_kmv_dofs(tri, args.degree)), "grau": args.degree,
            "componentes": problem.components,
            "aresta_km": {"min": float(arestas.min()), "media": float(arestas.mean()), "max": float(arestas.max()),
                          "desvio": float(arestas.std())},
            "interfaces_conformes": list(problem.conforming_depths), "triangulos_que_cruzam": cruzam,
            "conforme": all(n == 0 for n in cruzam.values()),
            "segundos": segundos, "pico_memoria_gib": _pico_memoria_gib(),
            "sha256": hashlib.sha256(Path(saida).read_bytes()).hexdigest()}
    Path(saida).with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def malha_3d(caso: str, frequencia: float, c: float, saida: Path, threads: int = 1,
             sigma_km: float = 0.1, passo_km: float = 0.1) -> dict:
    """Gera uma malha uniforme 3D (Salt suavizado, ou verdadeiro com o fundo do mar conforme).

    Parameters
    ----------
    caso : str
        ``suavizado`` ou ``verdadeiro``.
    frequencia : float
        Frequência (Hz).
    c : float
        Células por comprimento de onda no v_p mínimo.
    saida : pathlib.Path
        Arquivo ``.msh`` (binário, formato 2.2, como os do 3D).
    threads : int, optional
        Threads do gmsh (HXT). Default 1.
    sigma_km, passo_km : float, optional
        Parâmetros do fundo do mar (``fundo_salt3d.superficie``).

    Returns
    -------
    dict
        Metadados da malha.
    """
    import gmsh
    import fundo_salt3d as fundo
    sys.path.insert(0, str(cfg.CODE3D))
    from mesh_geometry import audit_tetrahedra

    if caso == "suavizado":
        with np.load(cfg.MODELOS["salt3d_s250"]) as dados:
            v_min = float(dados["vp"].min())
        sup = None
    elif caso == "verdadeiro":
        v_min = VP_MIN_3D["verdadeiro"]
        sup = fundo.superficie(cfg.MODELOS["salt3d_s250"], sigma_km, passo_km)
    else:
        raise ValueError(f"caso 3D desconhecido: {caso}")
    h = v_min / (c * frequencia)
    t0 = time.perf_counter()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", threads)
        gmsh.model.add(Path(saida).stem)
        if sup is None:
            _caixa(gmsh)
        else:
            _caixa_com_fundo(gmsh, sup)
        for chave, valor in {"Mesh.MeshSizeFromPoints": 0, "Mesh.MeshSizeFromCurvature": 0,
                             "Mesh.MeshSizeExtendFromBoundary": 0, "Mesh.MeshSizeMin": h, "Mesh.MeshSizeMax": h,
                             "Mesh.Algorithm3D": 10, "Mesh.RandomSeed": 0, "Mesh.MshFileVersion": 2.2,
                             "Mesh.Binary": 1}.items():
            gmsh.option.setNumber(chave, valor)
        gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: h)
        gmsh.model.mesh.generate(3)
        Path(saida).parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(saida))
        tipos, _, nos = gmsh.model.mesh.getElements(3)
        if set(tipos) != {4}:
            raise RuntimeError(f"elementos 3D inesperados: {tipos}")
        tags, coords, _ = gmsh.model.mesh.getNodes()
    finally:
        gmsh.finalize()
    segundos = time.perf_counter() - t0
    coords = coords.reshape(-1, 3)
    conn = nos[0].reshape(-1, 4)
    geo = audit_tetrahedra(tags, coords, conn)
    meta = {"dimensao": 3, "caso": caso, "frequencia_hz": frequencia, "C": c, "v_min_km_s": v_min, "h_km": h,
            "geometria": geo, "gl_kmv3": geo["KMV3_dofs_predicted"], "segundos": segundos, "threads": threads,
            "pico_memoria_gib": _pico_memoria_gib(), "gmsh": gmsh.__version__,
            "sha256": hashlib.sha256(Path(saida).read_bytes()).hexdigest()}
    if sup is not None:
        ordem = np.argsort(tags)
        idx = ordem[np.searchsorted(tags[ordem], conn)]
        lados = sup.lado(coords[idx.ravel()]).reshape(-1, 4)
        cruzam = int(np.sum((lados.min(axis=1) < 0) & (lados.max(axis=1) > 0)))
        meta.update({"fundo": {"sigma_km": sup.sigma_km, "passo_km": sup.passo_km,
                               "profundidade_km": [float(sup.d.min()), float(sup.d.max())],
                               "origem_sha256": sup.origem_sha256},
                     "tetraedros_que_cruzam_o_fundo": cruzam, "conforme": cruzam == 0})
    Path(saida).with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


TOP, BOTTOM, X_MIN, X_MAX, Y_MIN, Y_MAX, FUNDO = 1, 2, 3, 4, 5, 6, 7


def _caixa(gmsh) -> None:
    """Caixa do Salt 3D com as marcas de ``make_mesh3d.py``."""
    from fundo_salt3d import XMAX, XMIN, YMAX, YMIN, ZMIN
    occ = gmsh.model.occ
    caixa = occ.addBox(XMIN, YMIN, ZMIN, XMAX - XMIN, YMAX - YMIN, -ZMIN)
    occ.synchronize()
    marcas = {}
    for dim, tag in gmsh.model.getBoundary([(3, caixa)], oriented=False):
        x, y, z = occ.getCenterOfMass(dim, tag)
        marca = (TOP if abs(z) < 1e-8 else BOTTOM if abs(z - ZMIN) < 1e-8 else X_MIN if abs(x - XMIN) < 1e-8
                 else X_MAX if abs(x - XMAX) < 1e-8 else Y_MIN if abs(y - YMIN) < 1e-8 else Y_MAX)
        marcas.setdefault(marca, []).append(tag)
    for marca, faces in marcas.items():
        gmsh.model.addPhysicalGroup(2, faces, marca)
    gmsh.model.addPhysicalGroup(3, [caixa], 1)


def _caixa_com_fundo(gmsh, sup) -> None:
    """Caixa dividida pelo fundo do mar em facetas planas (núcleo geo do gmsh).

    Parameters
    ----------
    gmsh : module
        O módulo gmsh já inicializado.
    sup : fundo_salt3d.Superficie
        O fundo do mar.
    """
    from fundo_salt3d import XMAX, XMIN, YMAX, YMIN, ZMIN
    geo = gmsh.model.geo
    nx, ny = len(sup.x), len(sup.y)
    S = [[geo.addPoint(sup.x[i], sup.y[j], -sup.d[j, i]) for i in range(nx)] for j in range(ny)]
    cantos = {(0, 0): (XMIN, YMIN), (nx - 1, 0): (XMAX, YMIN), (nx - 1, ny - 1): (XMAX, YMAX),
              (0, ny - 1): (XMIN, YMAX)}
    T = {k: geo.addPoint(x, y, 0.0) for k, (x, y) in cantos.items()}
    B = {k: geo.addPoint(x, y, ZMIN) for k, (x, y) in cantos.items()}
    linhas = {}

    def linha(a: int, b: int) -> int:
        """Linha de a a b, reaproveitada com sinal se já existe."""
        if (a, b) in linhas:
            return linhas[(a, b)]
        if (b, a) in linhas:
            return -linhas[(b, a)]
        linhas[(a, b)] = geo.addLine(a, b)
        return linhas[(a, b)]

    def face(pontos: list[int]) -> int:
        """Face plana pelo polígono fechado ``pontos``."""
        laco = geo.addCurveLoop([linha(a, b) for a, b in zip(pontos, pontos[1:] + pontos[:1])])
        return geo.addPlaneSurface([laco])

    fundo_faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            fundo_faces.append(face([S[j][i], S[j][i + 1], S[j + 1][i + 1]]))
            fundo_faces.append(face([S[j][i], S[j + 1][i + 1], S[j + 1][i]]))
    # Polilinhas do fundo em cada lado, do canto "inicial" ao "final" (sentido anti-horário visto de cima).
    lado_ymin = [S[0][i] for i in range(nx)]                       # (0,0) -> (nx-1,0)
    lado_xmax = [S[j][nx - 1] for j in range(ny)]                  # (nx-1,0) -> (nx-1,ny-1)
    lado_ymax = [S[ny - 1][i] for i in reversed(range(nx))]        # (nx-1,ny-1) -> (0,ny-1)
    lado_xmin = [S[j][0] for j in reversed(range(ny))]             # (0,ny-1) -> (0,0)
    lados = [((0, 0), (nx - 1, 0), lado_ymin, Y_MIN), ((nx - 1, 0), (nx - 1, ny - 1), lado_xmax, X_MAX),
             ((nx - 1, ny - 1), (0, ny - 1), lado_ymax, Y_MAX), ((0, ny - 1), (0, 0), lado_xmin, X_MIN)]
    topo = face([T[(0, 0)], T[(nx - 1, 0)], T[(nx - 1, ny - 1)], T[(0, ny - 1)]])
    base = face([B[(0, 0)], B[(nx - 1, 0)], B[(nx - 1, ny - 1)], B[(0, ny - 1)]])
    agua_lat, rocha_lat, marcas = [], [], {}
    for c0, c1, poli, marca in lados:
        agua = face([T[c0], T[c1], *reversed(poli)])           # topo c0->c1, desce em c1, volta pelo fundo
        rocha = face([*poli, B[c1], B[c0]])                    # fundo c0->c1, desce em c1, volta pela base
        agua_lat.append(agua)
        rocha_lat.append(rocha)
        marcas.setdefault(marca, []).extend([agua, rocha])
    agua_vol = geo.addVolume([geo.addSurfaceLoop([topo, *fundo_faces, *agua_lat])])
    rocha_vol = geo.addVolume([geo.addSurfaceLoop([base, *fundo_faces, *rocha_lat])])
    geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [topo], TOP)
    gmsh.model.addPhysicalGroup(2, [base], BOTTOM)
    for marca, faces in marcas.items():
        gmsh.model.addPhysicalGroup(2, faces, marca)
    gmsh.model.addPhysicalGroup(2, fundo_faces, FUNDO)
    gmsh.model.addPhysicalGroup(3, [agua_vol, rocha_vol], 1)


def main() -> None:
    """Linha de comando."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dimensao", choices=("2d", "3d"))
    p.add_argument("--caso", required=True)
    p.add_argument("--frequencia", type=float, required=True)
    p.add_argument("--c", type=float, required=True)
    p.add_argument("--saida", type=Path, required=True)
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--sigma-fundo", type=float, default=0.1)
    p.add_argument("--passo-fundo", type=float, default=0.1)
    a = p.parse_args()
    if a.dimensao == "2d":
        meta = malha_2d(a.caso, a.frequencia, a.c, a.saida, a.semente)
    else:
        meta = malha_3d(a.caso, a.frequencia, a.c, a.saida, a.threads, a.sigma_fundo, a.passo_fundo)
    print(json.dumps({k: meta[k] for k in ("caso", "frequencia_hz", "C", "h_km", "segundos")}))


if __name__ == "__main__":
    main()
