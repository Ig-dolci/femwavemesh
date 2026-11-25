"""Isolated acoustic s250 5 Hz campaign. Units km, km/s, s; z negative down.

Matriz 3D (25/11/2026): a mesma configuração da campanha v4 (cópia de
experiments/salt3d/results/acoustic_s250_5hz/v4/code/config_acoustic3d.py), com três variáveis de ambiente
opcionais. Sem elas, tudo fica idêntico à v4 (o G3a confere o fem3d novo contra o arquivado nesse modo):

- ``MATRIZ3D_FREQ``: frequência de pico do Ricker (Hz; padrão 5);
- ``MATRIZ3D_T``: duração do registro (s; padrão 2,5);
- ``MATRIZ3D_FONTES``: ``v4`` (as nove fontes da v4, padrão) ou ``duas`` (as duas da campanha de 09/11,
  (7,6; 5,1) e (7,25; 4,75) km, usadas na matriz).

O modelo (suavizado ou verdadeiro) e a representação da velocidade continuam em ``SALT3D_MODEL`` e
``SALT3D_COEF_MODE`` (``fem3d.py``). O arquivo do modelo s250 é procurado em ``code3d/data`` e, se não estiver lá,
em ``../data`` (o lugar da v4).

Preparado com assistência do Claude (Anthropic), a pedido da pesquisadora.
"""
import os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
_MODELO = "salt_s250_x6p9_y4p4_d2p4.npz"
MODEL_NPZ = next((p for p in (HERE / "data" / _MODELO, HERE.parent / "data" / _MODELO) if p.exists()),
                 HERE.parent / "data" / _MODELO)

# Janela física e camada absorvente (laterais em x e y, e fundo)
X0, X1 = 6.9, 8.3
Y0, Y1 = 4.4, 5.8
Z_BOTTOM = -2.4
PAD = 0.3
XMIN, XMAX = X0 - PAD, X1 + PAD
YMIN, YMAX = Y0 - PAD, Y1 + PAD
ZMIN, ZMAX = Z_BOTTOM - PAD, 0.0

# Ricker de 5 Hz, amplitude unitária (a matriz troca por MATRIZ3D_FREQ).
FREQ = float(os.environ.get("MATRIZ3D_FREQ", "5.0"))
RICKER_DELAY = 1.5 * np.sqrt(6.0) / (np.pi * FREQ)      # mesmo atraso do spyro
# Janela temporal do novo recorte físico de 2,4 km (a matriz troca por MATRIZ3D_T).
FINAL_TIME = float(os.environ.get("MATRIZ3D_T", "2.5"))
OUT_DT = 1e-3

# Aquisição a 20/40 m; a suavização inclui a antiga coluna de água.
SOURCE_DEPTH = -0.02
RECEIVER_DEPTH = -0.04

FONTES = os.environ.get("MATRIZ3D_FONTES", "v4")
if FONTES == "v4":
    SOURCES = [(x, y, SOURCE_DEPTH) for y in (4.65, 5.05, 5.45) for x in (7.15, 7.55, 7.95)]
elif FONTES == "duas":
    SOURCES = [(7.6, 5.1, SOURCE_DEPTH), (7.25, 4.75, SOURCE_DEPTH)]
else:
    raise ValueError(f"MATRIZ3D_FONTES desconhecido: {FONTES}")
RECEIVERS = [(x, y, RECEIVER_DEPTH) for y in np.linspace(Y0,Y1,15) for x in np.linspace(X0,X1,15)]
CFL_SAFETY = 0.4  # mesma fração do pipeline KMV3 atual; dt divide OUT_DT
SEED = 0
TOLERANCE = 0.05
THETA = 0.8
INITIAL_CPW = 1.33
INITIAL_CPW_GROWTH = 1.5
INITIAL_CPW_MAX = 6.0
INITIAL_ESTIMATOR_LIMIT = 0.3
NORM_RATIO_BOUNDS = (0.5, 2.0)
NORM_RATIO_POLICY = "references_controls_or_estimator_at_most_0p3"
MAX_ITERATIONS = 8
DENSITY_SMOOTHING_KM = 0.084
DESIGN_GRID_KM = 0.02

# Alteração autorizada antes da fase 3, 10/11/2026:
# Não há fundo do mar no modelo totalmente suavizado: usar todo o tempo
# de 0 a FINAL_TIME. V3 acrescenta a exclusão espacial autorizada abaixo,
# comum ao erro, estimador e funcional adjunto; a gravação mantém a aquisição.
if FONTES == "v4" and FREQ == 5.0 and FINAL_TIME == 2.5:
    CONFIGURATION_REVISION = "v4_nine_sources_20261111"
else:
    CONFIGURATION_REVISION = f"matriz3d_{FONTES}_f{FREQ:g}_T{FINAL_TIME:g}_20261125"
OBSERVATION_MODE = "full_time_offset_mask"
MIN_HORIZONTAL_OFFSET_KM = 0.25
# Decisão da pesquisadora antes de qualquer comparação de métodos: excluir
# afastamentos horizontais <250 m; manter todos os receptores nos arquivos.
# Constantes abaixo servem APENAS ao diagnóstico da janela antiga.
WATER_VELOCITY = 1.5
MASK_MARGIN = 0.17

DEGREE = 3          # KMV3 em tetraedros (o grau máximo de KMV 3D no Firedrake)
DAMP_R = 1e-3


def water_depth_at(xs, ys):
    """Lâmina d'água (km) do recorte, no ponto de grade mais próximo."""
    d = np.load(MODEL_NPZ)
    ix = np.clip(np.round((np.asarray(xs) - d["x_km"][0]) / 0.02).astype(int), 0, len(d["x_km"]) - 1)
    iy = np.clip(np.round((np.asarray(ys) - d["y_km"][0]) / 0.02).astype(int), 0, len(d["y_km"]) - 1)
    return d["water_depth_km"][iy, ix]


def mask_start_times(source):
    """DIAGNÓSTICO LEGADO: início pós-fundo original; não usado no erro/adjunto atual."""
    xs, ys, zs = source
    r = np.array(RECEIVERS)
    wd = water_depth_at(0.5 * (xs + r[:, 0]), 0.5 * (ys + r[:, 1]))   # lâmina no ponto médio
    L = np.sqrt((r[:, 0] - xs) ** 2 + (r[:, 1] - ys) ** 2 + (2 * wd + zs + r[:, 2]) ** 2)
    return RICKER_DELAY + L / WATER_VELOCITY + MASK_MARGIN


def observation_mask(source, times=None):
    """B v3: t em [0, FINAL_TIME], afastamento horizontal >=250 m por fonte."""
    if times is None:
        times = np.arange(int(round(FINAL_TIME / OUT_DT)) + 1) * OUT_DT
    times = np.asarray(times)
    valid = (times >= -1e-12) & (times <= FINAL_TIME + 1e-12)
    return valid[:, None] & receiver_selection(source)[None, :]


def receiver_selection(source):
    """Inclusão da igualdade com tolerância apenas de arredondamento (1e-12 km)."""
    offsets = np.linalg.norm(np.asarray(RECEIVERS)[:, :2] - np.asarray(source)[:2], axis=1)
    return offsets >= MIN_HORIZONTAL_OFFSET_KM - 1e-12


def legacy_observation_mask(source, times=None):
    """Janela antiga, exclusivamente diagnóstico secundário."""
    if times is None:
        times = np.arange(int(round(FINAL_TIME / OUT_DT)) + 1) * OUT_DT
    return np.asarray(times)[:, None] >= mask_start_times(source)[None, :]


if __name__ == "__main__":
    print(f"{len(SOURCES)} fontes, {len(RECEIVERS)} receptores gravados, {FREQ:g} Hz, {FINAL_TIME:g} s; B = offset horizontal >=250 m")
    for s in SOURCES:
        t = mask_start_times(s)
        print(f"  fonte ({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f}) km: janela começa entre {t.min():.2f} e {t.max():.2f} s")

# Fase 3 aprovada em 11/11/2026, antes de qualquer resultado dos orçamentos.
PHASE3_P0 = None
PHASE3_Q = None
PHASE3_MAX_DOFS = None
COMMON_BUDGETS = ()
BUDGET_RELATIVE_TOLERANCE = 0.02
CHECKPOINT_BLOCKS = 32
PILOT_REPORT_HOURS = 24
