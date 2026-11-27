"""Utilidades de los experimentos: estilo de gráficos, paralelismo y registro de decisiones."""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from golillas.datos import RAIZ, RESULTADOS  # noqa: E402

RESULTADOS.mkdir(exist_ok=True)
REFERENCIA = RAIZ / "data" / "real" / "referencia"

# Paleta validada (dataviz, modo claro): azul = elegido / sintético, naranja = real, gris = resto
AZUL, NARANJA, AQUA, AMARILLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRIS, TINTA, TINTA2 = "#b4b2a9", "#0b0b0b", "#52514e"
GRILLA, EJE, SUPERFICIE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SECUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def estilo():
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": EJE, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRILLA, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "medium", "axes.titlelocation": "left",
        "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA,
        "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def barras_h(ax, nombres, series, etiquetas=None, colores=None, elegido=None, fmt="{:.3f}", lim=None):
    """Barras horizontales agrupadas; la opción elegida va en negrita con ★."""
    series = [np.asarray(s, float) for s in series]
    n = len(series)
    alto = 0.8 / n
    y = np.arange(len(nombres))[::-1]
    una = n == 1 and colores is None
    colores = colores or [AZUL, NARANJA, AQUA][:n]
    for k, (s, c) in enumerate(zip(series, colores)):
        yy = y + (n - 1) / 2 * alto - k * alto
        if una:  # una sola serie: gris, y la elegida en azul
            c = [AZUL if m == elegido else GRIS for m in nombres]
        ax.barh(yy, s, height=alto * 0.85, color=c, label=etiquetas[k] if etiquetas else None)
        for yi, v in zip(yy, s):
            if np.isfinite(v):
                ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=8, color=TINTA2)
    ax.set_yticks(y, [f"★ {m}" if m == elegido else m for m in nombres])
    for t in ax.get_yticklabels():
        if t.get_text().startswith("★"):
            t.set_fontweight("bold")
            t.set_color(TINTA)
    ax.grid(axis="y", visible=False)
    if lim:
        ax.set_xlim(*lim)
    if etiquetas:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=len(etiquetas), fontsize=8)


def guardar(fig, nombre):
    fig.savefig(RESULTADOS / nombre)
    plt.close(fig)
    print("  →", nombre)


def mapa(f, items, procesos=4):
    with Pool(procesos) as p:
        return p.map(f, items, chunksize=max(1, len(items) // (procesos * 6)))


def ms(t0):
    return (time.perf_counter() - t0) * 1000


DECISIONES = RESULTADOS / "decisiones.json"


def leer_decisiones() -> dict:
    return json.loads(DECISIONES.read_text()) if DECISIONES.exists() else {}


def decision(proceso: str) -> str:
    return leer_decisiones()[proceso]["opcion"]


def decidir(proceso: str, opcion: str, criterio: str, **detalle):
    d = leer_decisiones()
    d[proceso] = {"opcion": opcion, "criterio": criterio, **detalle}
    DECISIONES.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    print(f"DECISIÓN {proceso}: {opcion}  ({criterio})")


def referencia(id_: str) -> np.ndarray:
    return (cv2.imread(str(REFERENCIA / f"{id_}.png"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)


def n_agujeros(m: np.ndarray) -> int:
    from golillas.segmentacion import partes

    relleno, _ = partes(m)
    hueco = (relleno & (1 - m)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(hueco, connectivity=4)
    return int((st[1:, 4] > 20).sum())


def recorte(img, cx, cy, r=80, lado=200):
    p = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REPLICATE)
    x, y = int(round(cx)) + r, int(round(cy)) + r
    return cv2.resize(p[y - r : y + r, x - r : x + r], (lado, lado), interpolation=cv2.INTER_AREA)


def centro(m):
    mm = cv2.moments(m, binaryImage=True)
    return (mm["m10"] / mm["m00"], mm["m01"] / mm["m00"]) if mm["m00"] else (m.shape[1] / 2, m.shape[0] / 2)
