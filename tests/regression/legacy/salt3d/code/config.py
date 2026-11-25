"""Caminhos, problemas e parâmetros comuns da campanha P0/P3/P4 (matriz, 24/11/2026).

Tudo é relativo à raiz da campanha (variável ``CAMP``, ou a pasta acima de ``code/``). Disposição esperada no
Mintrop, a mesma das campanhas anteriores::

    $CAMP/code/      scripts desta campanha e cópias de laco_v3.py, cpw_fontes.py, validar_dt_fixo.py
    $CAMP/spyro/     snapshot do spyro (ramo e commit registrados no MANIFEST.json)
    $CAMP/code3d/    cópia do código 3D (experiments/salt3d/results/acoustic_s250_5hz/v4/code) com os
                     arquivos de P3
    $CAMP/entrada/   modelos e referências (ver ARQUIVOS.txt)
    $CAMP/results/   saídas
    $CAMP/logs/      logs de cada tarefa

Preparado com assistência do Claude (Anthropic), a pedido da pesquisadora.
"""
from __future__ import annotations

import os
from pathlib import Path

CAMP = Path(os.environ.get("CAMP", Path(__file__).resolve().parents[1]))
CODE = CAMP / "code"
CODE3D = CAMP / "code3d"
DEMO = CAMP / "spyro" / "demos" / "automatic_mesh_adaptation"
ENTRADA = CAMP / "entrada"
RESULTS = CAMP / "results"
LOGS = CAMP / "logs"

MODELOS = {
    "acustico_s250": ENTRADA / "marmousi_window_s250.npz",
    "acustico_verdadeiro": ENTRADA / "marmousi_window.npz",
    "elastico_s250": ENTRADA / "marmousi_window_elastic_s250.npz",
    "elastico_verdadeiro": ENTRADA / "marmousi_window_elastic_true.npz",
    "salt3d_s250": ENTRADA / "salt_s250_x6p9_y4p4_d2p4.npz",
    "salt3d_original": ENTRADA / "source_original.npz",
}

# Referências antigas: só para o portão do laço v4 (comparação com a v1 e a Parte G). A matriz usa as uniformes.
REFERENCIAS_PORTAO = {
    "acustico": ENTRADA / "F30_s0_ci_st_traces.npz",
    "elastico": ENTRADA / "elastic_none_C6.npz",
}
PASSOS_POR_AMOSTRA_PORTAO = {"acustico": 12, "elastico": 4}

# Problemas do portão: os mesmos da Parte G (run_parte_g_mac.sh, função problema).
PROBLEMAS_PORTAO = {
    "acustico": ["--wave", "acoustic", "--model", str(MODELOS["acustico_s250"]), "--frequency", "7",
                 "--damping-reflection", "1e-3", "--conforming-depths", "-0.45125", "--no-progress-bar",
                 "--sea-floor-step", "--discontinuous-velocity"],
    "elastico": ["--wave", "elastic", "--model", str(MODELOS["elastico_s250"]), "--frequency", "3.5",
                 "--damping-reflection", "1e-3", "--abc", "none", "--no-progress-bar",
                 "--discontinuous-velocity"],
}
# A Parte G rodou o elástico com velocidade nodal; no modelo suave as duas coincidem (Q-E, 24/11), e o portão usa a
# velocidade por elemento, como toda rodada nova (decisão de 24/11).

# Parâmetros do laço (os da v1 e da Parte G).
LACO = ["--theta", "0.8", "--initial-cpw", "auto", "--cpw-start", "1.33", "--cpw-factor", "1.5",
        "--eta-reliable", "0.3", "--p0", "1.8", "--growth-safety", "1.0", "--growth-min", "1.15",
        "--growth-max", "2", "--max-iterations", "8", "--accumulate-every", "16"]
TOLERANCIA = 0.05

# Emenda de 24/11 (EMENDA_FASE_INICIAL_2026-11-24.md, decisões da autora às 11h19 e 11h39): a fase inicial
# multiplica C por 1,5^(2/3) ≈ 1,31 por nível em todos os casos (o fator do 3D), e não por 1,5. No acústico por
# elemento, com 1,5, a escada 1,33 → 1,995 → 2,9925 pula de Ê ≈ 36% (não confiável) para Ê ≈ 1,9% (abaixo de
# θ·tol = 4%): o laço aceita a malha CPW da fase inicial sem adaptar (sementes 0 e 2 da Parte G). Regra: o degrau s
# não pode pular a janela 0,3/(θ·tol) = 7,5 em Ê; com Ê ∝ C^-q, s ≤ 7,5^(1/q) (acústico q ≈ 7,3: 1,32; elástico
# q ≈ 3,3: 1,84). No elástico a 3,5, 5 e 7 Hz o primeiro nível (C = 1,33) já é confiável e o fator não é usado.
# LACO fica como estava (V0 e o portão original); as rodadas novas usam laco_args().
FATOR_C_INICIAL = {"acustico": 1.5 ** (2.0 / 3.0), "elastico": 1.5 ** (2.0 / 3.0)}


def laco_args(caso: str) -> list[str]:
    """Parâmetros do laço de um caso, com o fator da fase inicial da emenda de 24/11.

    Parameters
    ----------
    caso : str
        ``acustico``, ``elastico`` ou um caso da matriz (``acustico_s250``, ``elastico_verdadeiro``...).

    Returns
    -------
    list of str
        ``LACO`` com ``--cpw-factor`` trocado pelo fator do caso.
    """
    base = "acustico" if caso.startswith("acustico") else "elastico"
    args = list(LACO)
    args[args.index("--cpw-factor") + 1] = f"{FATOR_C_INICIAL[base]:.6f}"
    return args


def problema_matriz(caso: str, frequencia: float, tempo_final: float) -> list[str]:
    """Argumentos de ``marmousi_automatic_loop.py`` de uma célula 2D da matriz.

    Parameters
    ----------
    caso : str
        ``acustico_s250``, ``acustico_verdadeiro``, ``elastico_s250`` ou ``elastico_verdadeiro``.
    frequencia : float
        Frequência de pico (Hz).
    tempo_final : float
        Duração do registro (s).

    Returns
    -------
    list of str
        Argumentos do problema, com velocidade por elemento (decisão de 24/11). Os verdadeiros ainda esperam a
        decisão D1 (média por célula ou não); aqui ficam sem ``--cell-average``.
    """
    comum = ["--frequency", f"{frequencia:g}", "--final-time", f"{tempo_final:g}",
             "--damping-reflection", "1e-3", "--no-progress-bar", "--discontinuous-velocity"]
    if caso == "acustico_s250":
        return ["--wave", "acoustic", "--model", str(MODELOS[caso]), "--conforming-depths", "-0.45125",
                "--sea-floor-step", *comum]
    if caso == "acustico_verdadeiro":
        return ["--wave", "acoustic", "--model", str(MODELOS[caso]), "--conforming-depths", "-0.45125",
                "-0.505625", "--sea-floor-step", *comum]
    if caso in ("elastico_s250", "elastico_verdadeiro"):
        return ["--wave", "elastic", "--model", str(MODELOS[caso]), "--abc", "none", *comum]
    raise ValueError(f"caso desconhecido: {caso}")
