"""Fundo do mar do Salt 3D verdadeiro: superfície por facetas planas, a mesma na malha (P4) e no modelo (P3).

A lâmina d'água do recorte (``water_depth_km``, grade de 20 m) vem em degraus de 20 m (0,06 a 0,12 km). A superfície
usada aqui é a lâmina suavizada por uma gaussiana (``sigma_km``) e amostrada numa grade regular (``passo_km``); cada
célula da grade vira dois triângulos planos, com a diagonal do canto (i, j) ao (i+1, j+1). A malha conforme (P4) usa
esses triângulos como faces, e o modelo verdadeiro (P3) põe água acima deles e rocha abaixo, com a mesma função
``profundidade``: o salto da velocidade fica exatamente numa face da malha.

Os padrões (``sigma_km`` = 0,1, ``passo_km`` = 0,1) são uma escolha de modelagem registrada no MANIFEST; mudar exige
gerar de novo malhas e modelo.

Preparado com assistência do Claude (Anthropic), a pedido da pesquisadora.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Caixa do domínio 3D (config_acoustic3d.py: janela física mais a camada de 0,3 km).
XMIN, XMAX, YMIN, YMAX, ZMIN = 6.6, 8.6, 4.1, 6.1, -2.7


@dataclass(frozen=True)
class Superficie:
    """Fundo do mar por facetas: profundidades ``d[j, i]`` (km, positivas) nos nós ``(x[i], y[j])``."""

    x: np.ndarray
    y: np.ndarray
    d: np.ndarray
    sigma_km: float
    passo_km: float
    origem_sha256: str

    def profundidade(self, x, y) -> np.ndarray:
        """Lâmina d'água (km) em pontos ``(x, y)``, plana em cada triângulo da grade.

        Parameters
        ----------
        x, y : array_like
            Coordenadas (km), dentro da caixa.

        Returns
        -------
        numpy.ndarray
            Profundidade do fundo (positiva).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        dx, dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
        fi = np.clip((x - self.x[0]) / dx, 0.0, len(self.x) - 1 - 1e-12)
        fj = np.clip((y - self.y[0]) / dy, 0.0, len(self.y) - 1 - 1e-12)
        i = np.minimum(fi.astype(int), len(self.x) - 2)
        j = np.minimum(fj.astype(int), len(self.y) - 2)
        u, v = fi - i, fj - j
        d00, d10 = self.d[j, i], self.d[j, i + 1]
        d01, d11 = self.d[j + 1, i], self.d[j + 1, i + 1]
        baixo = d00 + u * (d10 - d00) + v * (d11 - d10)      # triângulo (0,0), (1,0), (1,1): u >= v
        cima = d00 + v * (d01 - d00) + u * (d11 - d01)       # triângulo (0,0), (1,1), (0,1): u < v
        return np.where(u >= v, baixo, cima)

    def lado(self, xyz: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """+1 na rocha (abaixo do fundo), −1 na água, 0 sobre o fundo.

        Parameters
        ----------
        xyz : numpy.ndarray
            Pontos ``(n, 3)`` com z negativo para baixo.
        tol : float, optional
            Tolerância (km) para "sobre o fundo".

        Returns
        -------
        numpy.ndarray
            Sinal por ponto.
        """
        s = -xyz[:, 2] - self.profundidade(xyz[:, 0], xyz[:, 1])
        return np.where(s > tol, 1, np.where(s < -tol, -1, 0))


def superficie(modelo_s250: str | Path, sigma_km: float = 0.1, passo_km: float = 0.1) -> Superficie:
    """Constrói a superfície a partir de ``water_depth_km`` do recorte s250.

    Parameters
    ----------
    modelo_s250 : str or pathlib.Path
        ``salt_s250_x6p9_y4p4_d2p4.npz`` (traz ``water_depth_km[y, x]``, ``x_km`` e ``y_km`` em 20 m).
    sigma_km : float, optional
        Desvio da gaussiana que suaviza os degraus de 20 m. Default 0,1.
    passo_km : float, optional
        Lado das células da grade de facetas. Default 0,1 (21 × 21 nós na caixa de 2 km).

    Returns
    -------
    Superficie
        A superfície.
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter

    caminho = Path(modelo_s250)
    with np.load(caminho) as dados:
        lamina = dados["water_depth_km"].astype(float)
        xs, ys = dados["x_km"].astype(float), dados["y_km"].astype(float)
    passo_grade = float(xs[1] - xs[0])
    suave = gaussian_filter(lamina, sigma=sigma_km / passo_grade, mode="nearest")
    interp = RegularGridInterpolator((ys, xs), suave)
    nx = int(round((XMAX - XMIN) / passo_km)) + 1
    ny = int(round((YMAX - YMIN) / passo_km)) + 1
    x = np.linspace(XMIN, XMAX, nx)
    y = np.linspace(YMIN, YMAX, ny)
    Y, X = np.meshgrid(y, x, indexing="ij")
    pontos = np.column_stack([np.clip(Y.ravel(), ys[0], ys[-1]), np.clip(X.ravel(), xs[0], xs[-1])])
    d = interp(pontos).reshape(ny, nx)
    return Superficie(x=x, y=y, d=d, sigma_km=sigma_km, passo_km=passo_km,
                      origem_sha256=hashlib.sha256(caminho.read_bytes()).hexdigest())
