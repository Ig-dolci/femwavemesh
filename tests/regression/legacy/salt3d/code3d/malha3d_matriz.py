"""Malhas tetraédricas da matriz 3D: CPW (h = v/(C·f) local) ou campo de tamanho do adjunto; fundo conforme no verdadeiro.

Mesma caixa, mesmas marcas de contorno (1 topo, 2 base, 3–6 laterais) e mesmas opções do gmsh do
``make_mesh3d.py`` da v4 (HXT, semente 0, formato 2.2 binário). Diferenças:

- o modelo: ``suavizado`` (s250, interpolação trilinear da grade de 20 m, como na v4) ou ``verdadeiro`` (água com
  1,5 km/s acima do fundo de ``fundo_salt3d.py`` e ``vp_rocha`` do recorte original abaixo, o mesmo modelo do
  ``fem3d.TrueVelocity``);
- no verdadeiro, a caixa é dividida pelo fundo por facetas planas (``malhas_uniformes._caixa_com_fundo``, marca 7):
  nenhum tetraedro atravessa o fundo, como nas referências uniformes de P4;
- a frequência vem da linha de comando (a v4 usava a do ``config_acoustic3d``).

O ``.json`` de cada malha traz ``actual_geometry`` (o ``forward.py`` confere o GL do Firedrake contra ele), o C ou o
SHA-256 do campo, a conformidade com o fundo e o tempo.

Uso::

    python malha3d_matriz.py --caso suavizado|verdadeiro --frequencia F --saida M.msh (--C C | --campo campo.npz)

O campo (``.npz``) tem ``centroids`` (n × 3, km, z negativo) e ``requested_h`` (n), como na v4; entre os pontos,
média ponderada pelo inverso do quadrado da distância em log(h), com os 4 mais próximos.

Preparado com assistência do Claude (Anthropic), a pedido da pesquisadora.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
AGUA = 1.5


class Modelo:
    """Velocidade do caso em pontos e numa grade, sem Firedrake.

    Parameters
    ----------
    caso : str
        ``suavizado`` ou ``verdadeiro``.
    """

    def __init__(self, caso: str) -> None:
        import config_acoustic3d as cfg
        self.caso = caso
        with np.load(cfg.MODEL_NPZ) as d:
            self.x, self.y, self.z = (np.asarray(d[k], float) for k in ("x_km", "y_km", "depth_km"))
            vp_s250 = np.asarray(d["vp"], float)
        self.sup = None
        if caso == "suavizado":
            self.v = vp_s250
            self.origem = cfg.MODEL_NPZ
        elif caso == "verdadeiro":
            import fundo_salt3d
            self.origem = Path(os.environ.get("SALT3D_TRUE_NPZ", AQUI / "data" / "salt_true_x6p6_y4p1_d2p7.npz"))
            with np.load(self.origem) as d:
                eixos = tuple(np.asarray(d[k], float) for k in ("x_km", "y_km", "depth_km"))
                if not all(np.allclose(a, b) for a, b in zip(eixos, (self.x, self.y, self.z))):
                    raise ValueError("a grade do modelo verdadeiro não é a do s250")
                self.v = np.asarray(d["vp_rocha"], float)
                sigma, passo = float(d["fundo_sigma_km"]), float(d["fundo_passo_km"])
            self.sup = fundo_salt3d.superficie(cfg.MODEL_NPZ, sigma, passo)
        else:
            raise ValueError(f"caso 3D desconhecido: {caso}")
        self.passo = float(self.x[1] - self.x[0])

    def _trilinear(self, x, y, z) -> np.ndarray:
        """Interpolação trilinear de ``self.v`` (profundidade, y, x) em pontos; fora da grade, o valor da borda."""
        q = [np.clip((-np.asarray(z, float) - self.z[0]) / self.passo, 0, len(self.z) - 1),
             np.clip((np.asarray(y, float) - self.y[0]) / self.passo, 0, len(self.y) - 1),
             np.clip((np.asarray(x, float) - self.x[0]) / self.passo, 0, len(self.x) - 1)]
        i = [np.minimum(t.astype(int), n - 2) for t, n in zip(q, self.v.shape)]
        w = [t - j for t, j in zip(q, i)]
        valor = 0.0
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    peso = ((w[0] if dz else 1 - w[0]) * (w[1] if dy else 1 - w[1]) * (w[2] if dx else 1 - w[2]))
                    valor = valor + self.v[i[0] + dz, i[1] + dy, i[2] + dx] * peso
        return np.asarray(valor, float)

    def __call__(self, x, y, z) -> np.ndarray:
        """Velocidade (km/s) em pontos; no verdadeiro, sobre o fundo conta como água (o tamanho menor)."""
        v = self._trilinear(x, y, z)
        if self.sup is None:
            return v
        agua = -np.asarray(z, float) <= self.sup.profundidade(x, y) + 1e-9
        return np.where(agua, AGUA, v)

    def grade(self) -> tuple[np.ndarray, np.ndarray]:
        """A grade de projeto (a do modelo, 20 m): pontos ``(n, 3)`` e a velocidade com forma (prof., y, x)."""
        Z, Y, X = np.meshgrid(self.z, self.y, self.x, indexing="ij")
        pontos = np.column_stack([X.ravel(), Y.ravel(), -Z.ravel()])
        return pontos, self(pontos[:, 0], pontos[:, 1], pontos[:, 2]).reshape(Z.shape)

    def extremos(self) -> tuple[float, float]:
        """v mínimo e máximo do caso."""
        vmin, vmax = float(self.v.min()), float(self.v.max())
        return (min(vmin, AGUA), max(vmax, AGUA)) if self.sup is not None else (vmin, vmax)


def gerar(caso: str, frequencia: float, saida: Path, C: float | None = None, campo: Path | None = None,
          threads: int = 1) -> dict:
    """Gera a malha e grava ``saida`` e ``saida.json``.

    Parameters
    ----------
    caso : str
        ``suavizado`` ou ``verdadeiro``.
    frequencia : float
        Frequência (Hz) do tamanho CPW.
    saida : pathlib.Path
        Arquivo ``.msh``.
    C : float, optional
        Células por comprimento de onda local (malha CPW).
    campo : pathlib.Path, optional
        Campo de tamanho (malha do adjunto). Exatamente um de ``C`` e ``campo``.
    threads : int, optional
        Threads do gmsh.

    Returns
    -------
    dict
        Os metadados gravados.
    """
    import gmsh
    from scipy.spatial import cKDTree
    from malhas_uniformes import _caixa, _caixa_com_fundo
    from mesh_geometry import audit_tetrahedra

    if (C is None) == (campo is None):
        raise ValueError("dê C ou campo, um dos dois")
    modelo = Modelo(caso)
    vmin, vmax = modelo.extremos()
    arvore = None
    if campo is not None:
        with np.load(campo) as z:
            centros = np.asarray(z["centroids"], float)
            pedido = np.asarray(z["requested_h"], float)
        if centros.ndim != 2 or centros.shape[1] != 3 or pedido.shape != (len(centros),) or \
                not (np.isfinite(pedido).all() and (pedido > 0).all()):
            raise ValueError(f"campo inválido: {campo}")
        arvore, logh = cKDTree(centros), np.log(pedido)
        hmin, hmax = float(pedido.min()), float(pedido.max())
    else:
        hmin, hmax = vmin / (C * frequencia), vmax / (C * frequencia)

    def tamanho(x: float, y: float, z: float) -> float:
        if arvore is None:
            return float(modelo(np.array([x]), np.array([y]), np.array([z]))[0]) / (C * frequencia)
        dist, idx = arvore.query([x, y, z], k=4)
        dist, idx = np.atleast_1d(dist), np.atleast_1d(idx)
        if dist[0] < 1e-12:
            return float(pedido[idx[0]])
        w = 1.0 / np.maximum(dist, 1e-12) ** 2
        return float(np.exp(np.dot(w, logh[idx]) / w.sum()))

    t0 = time.perf_counter()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", threads)
        gmsh.model.add(Path(saida).stem)
        if modelo.sup is None:
            _caixa(gmsh)
        else:
            _caixa_com_fundo(gmsh, modelo.sup)
        for chave, valor in {"Mesh.MeshSizeFromPoints": 0, "Mesh.MeshSizeFromCurvature": 0,
                             "Mesh.MeshSizeExtendFromBoundary": 0, "Mesh.Algorithm3D": 10, "Mesh.RandomSeed": 0,
                             "Mesh.MshFileVersion": 2.2, "Mesh.Binary": 1, "Mesh.MeshSizeMin": hmin,
                             "Mesh.MeshSizeMax": hmax}.items():
            gmsh.option.setNumber(chave, valor)
        gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: tamanho(x, y, z))
        gmsh.model.mesh.generate(3)
        Path(saida).parent.mkdir(parents=True, exist_ok=True)
        temp = Path(saida).with_suffix(".tmp.msh")
        gmsh.write(str(temp))
        tipos, _, nos = gmsh.model.mesh.getElements(3)
        if set(tipos) != {4}:
            raise RuntimeError(f"elementos 3D inesperados: {tipos}")
        tags, coords, _ = gmsh.model.mesh.getNodes()
    finally:
        gmsh.finalize()
    os.replace(temp, saida)
    coords = coords.reshape(-1, 3)
    conn = nos[0].reshape(-1, 4)
    geo = audit_tetrahedra(tags, coords, conn)
    meta = {"caso": caso, "frequency_hz": frequencia, "C": C,
            "size_field": None if campo is None else str(campo),
            "size_field_sha256": None if campo is None else hashlib.sha256(Path(campo).read_bytes()).hexdigest(),
            "actual_geometry": geo, "dofs": int(geo["KMV3_dofs_predicted"]), "tetrahedra": int(len(conn)),
            "seed": 0, "threads": threads, "elapsed_s": time.perf_counter() - t0, "gmsh": gmsh.__version__,
            "model": str(modelo.origem), "model_sha256": hashlib.sha256(Path(modelo.origem).read_bytes()).hexdigest(),
            "mesh_sha256": hashlib.sha256(Path(saida).read_bytes()).hexdigest()}
    if modelo.sup is not None:
        ordem = np.argsort(tags)
        idx = ordem[np.searchsorted(tags[ordem], conn)]
        lados = modelo.sup.lado(coords[idx.ravel()]).reshape(-1, 4)
        cruzam = int(np.sum((lados.min(axis=1) < 0) & (lados.max(axis=1) > 0)))
        meta.update({"tetraedros_que_cruzam_o_fundo": cruzam, "conforme": cruzam == 0})
        if cruzam:
            raise RuntimeError(f"{cruzam} tetraedros atravessam o fundo: {saida}")
    Path(saida).with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> None:
    """Linha de comando."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    p.add_argument("--caso", required=True, choices=("suavizado", "verdadeiro"))
    p.add_argument("--frequencia", type=float, required=True)
    p.add_argument("--saida", type=Path, required=True)
    p.add_argument("--C", type=float)
    p.add_argument("--campo", type=Path)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    meta = gerar(a.caso, a.frequencia, a.saida, a.C, a.campo, a.threads)
    print(json.dumps({k: meta[k] for k in ("caso", "C", "dofs", "tetrahedra", "elapsed_s")}))


if __name__ == "__main__":
    main()
