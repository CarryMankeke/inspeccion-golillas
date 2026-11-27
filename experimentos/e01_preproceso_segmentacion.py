"""N3 de los procesos 1 (preprocesado) y 2 (segmentación).

Se prueban las 8 opciones de preprocesado × las 11 de segmentación sobre
120 golillas sintéticas (con máscara exacta) y las 15 buenas reales (con la
referencia por votación). Métrica principal: IoU (intersección / unión) entre la
máscara obtenida y la verdadera; 1 es perfecto.

Decisión P1: el preprocesado con mejor IoU medio sobre TODAS las segmentaciones
(sin elegir todavía la segmentación: no se salta el nivel).
Decisión P2: con ese preprocesado, la segmentación con mejor IoU (promedio
sintético-real) entre las que aciertan la topología (1 agujero) en el 100 % de las
golillas BUENAS; en empate (<0,005) gana la más rápida.
Corrección del criterio (27 nov): la primera versión exigía topología ≥95 % sobre
todas las golillas con agujero, incluidas las malas. Todas las técnicas fallan igual
en las deformadas (el brillo corta el anillo), y eso no es un error: la pieza igual
se rechaza. Una topología rota solo hace daño en una golilla buena.
"""

import sys
import time

import cv2
import numpy as np
import pandas as pd
from comun import (
    RESULTADOS,
    SECUENCIAL,
    barras_h,
    centro,
    decidir,
    estilo,
    guardar,
    mapa,
    n_agujeros,
    recorte,
    referencia,
)
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from golillas import preproceso, segmentacion, sintetico
from golillas.datos import cargar_reales

PRE = list(preproceso.OPCIONES)
SEG = list(segmentacion.OPCIONES)
N_SINT = 120
SEMILLA = 1


def contraste(g, gt):
    """CNR global: (papel − metal) / desviación del papel en TODA la foto. Castiga la luz desigual."""
    metal = g[cv2.erode(gt, np.ones((5, 5), np.uint8)) > 0].astype(np.float32)
    papel = g[cv2.dilate(gt, np.ones((31, 31), np.uint8)) == 0].astype(np.float32)
    return float((papel.mean() - metal.mean()) / max(papel.std(), 1e-3))


def evaluar(img, gt, fuente, id_, defecto):
    filas = []
    huecos_gt = n_agujeros(gt)
    for p in PRE:
        t0 = time.perf_counter()
        g = preproceso.preprocesar(img, p)
        t_pre = (time.perf_counter() - t0) * 1000
        cnr = contraste(g, gt)
        for s in SEG:
            t1 = time.perf_counter()
            m = segmentacion.segmentar(g, s)
            t_seg = (time.perf_counter() - t1) * 1000
            filas.append(dict(fuente=fuente, id=id_, defecto=defecto, pre=p, seg=s,
                              iou=segmentacion.iou(m, gt), topo=n_agujeros(m) == huecos_gt,
                              tiene_agujero=huecos_gt == 1, cnr=cnr, ms_pre=t_pre, ms_seg=t_seg))
    return filas


def trabajo_sint(i):
    s = sintetico.generar(SEMILLA * 100_000 + i, p_mala=0.5)
    return evaluar(s.imagen, s.mascara, "sintética", f"s{i:03d}", s.defecto)


def trabajo_real(m):
    return evaluar(m.imagen(), referencia(m.id), "real", m.id, m.defecto)


def main():
    estilo()
    csv = RESULTADOS / "e01_grilla.csv"
    if csv.exists() and "--recalcular" not in sys.argv:
        df = pd.read_csv(csv)
    else:
        reales = [m for m in cargar_reales() if not m.es_mala]  # referencia confiable solo en buenas
        filas = sum(mapa(trabajo_sint, list(range(N_SINT))), []) + sum(mapa(trabajo_real, reales), [])
        df = pd.DataFrame(filas)
        df.to_csv(csv, index=False)

    # ---- P1: preprocesado, promediando todas las segmentaciones
    t1 = df.groupby(["pre", "fuente"]).agg(iou=("iou", "mean"), cnr=("cnr", "mean")).unstack("fuente")
    t1.columns = [f"{a}_{'sint' if b == 'sintética' else 'real'}" for a, b in t1.columns]
    t1["iou_prom"] = 0.5 * t1.iou_sint + 0.5 * t1.iou_real
    t1["ms"] = df.groupby("pre")["ms_pre"].median()
    t1 = t1.sort_values("iou_prom", ascending=False)
    t1.to_csv(RESULTADOS / "e01_preproceso.csv")
    pre = t1.index[0]
    decidir("preproceso", pre, "IoU medio sobre las 11 segmentaciones (promedio sintético y real)",
            iou_prom=round(float(t1.loc[pre, "iou_prom"]), 4))

    orden = list(t1.index)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    barras_h(axs[0], orden, [t1.loc[orden, "iou_sint"], t1.loc[orden, "iou_real"]],
             ["sintéticas (120)", "reales buenas (15)"], elegido=pre, lim=(0, 1.08))
    axs[0].set_title("IoU medio sobre las 11 segmentaciones")
    axs[0].set_xlabel("IoU (1 = máscara perfecta)")
    barras_h(axs[1], orden, [t1.loc[orden, "cnr_sint"], t1.loc[orden, "cnr_real"]],
             ["sintéticas", "reales"], elegido=pre, fmt="{:.1f}")
    axs[1].set_title("Contraste metal/papel (CNR global)")
    axs[1].set_xlabel("CNR (más alto = más fácil separar)")
    guardar(fig, "e01_preproceso.png")

    # ---- mapa de calor de la grilla completa
    piv = df.groupby(["pre", "seg"])["iou"].mean().unstack("seg").loc[orden]
    piv = piv[piv.mean().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cmap = LinearSegmentedColormap.from_list("azul", SECUENCIAL)
    im = ax.imshow(piv.values, cmap=cmap, vmin=0.5, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=35, ha="right")
    ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.8 else "#0b0b0b")
    fig.colorbar(im, ax=ax, label="IoU medio (sintéticas + reales)", shrink=0.85)
    ax.set_title("Corregir la luz rescata a los umbrales globales (Otsu, k-means); los adaptativos ya la compensan")
    guardar(fig, "e01_mapa_grilla.png")

    # ---- P2: segmentación con el preprocesado elegido
    d2 = df[df.pre == pre]
    t2 = d2.groupby(["seg", "fuente"])["iou"].mean().unstack("fuente")
    t2["iou_prom"] = 0.5 * t2["sintética"] + 0.5 * t2["real"]
    t2["topo"] = d2[d2.defecto == "ninguno"].groupby("seg")["topo"].mean()
    t2["topo_malas"] = d2[d2.tiene_agujero & (d2.defecto != "ninguno")].groupby("seg")["topo"].mean()
    t2["ms"] = d2.groupby("seg")["ms_seg"].median()
    t2 = t2.sort_values("iou_prom", ascending=False)
    t2.to_csv(RESULTADOS / "e02_segmentacion.csv")
    cand = t2[t2.topo >= 1.0]
    mejor = cand.iou_prom.max()
    empate = cand[cand.iou_prom >= mejor - 0.005].sort_values("ms")
    seg = empate.index[0]
    decidir("segmentacion", seg,
            "mejor IoU (promedio sintético-real) con topología 100 % en buenas; empate <0,005 → la más rápida",
            iou_prom=round(float(t2.loc[seg, "iou_prom"]), 4), ms=round(float(t2.loc[seg, "ms"]), 2))

    orden = list(t2.index)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True, gridspec_kw={"width_ratios": [2, 1, 1]})
    barras_h(axs[0], orden, [t2.loc[orden, "sintética"], t2.loc[orden, "real"]],
             ["sintéticas", "reales buenas"], elegido=seg, lim=(0, 1.1))
    axs[0].set_title(f"IoU con preprocesado «{pre}»")
    barras_h(axs[1], orden, [100 * t2.loc[orden, "topo"]], elegido=seg, fmt="{:.0f}", lim=(0, 125))
    axs[1].set_title("1 agujero en buenas (%)")
    barras_h(axs[2], orden, [t2.loc[orden, "ms"]], elegido=seg, fmt="{:.1f}")
    axs[2].set_title("Tiempo (ms)")
    guardar(fig, "e02_segmentacion.png")

    # ---- ejemplo visual con una foto real: el problema de la luz
    peor = df[(df.pre == "ninguno") & (df.seg == "otsu") & (df.fuente == "real")].sort_values("iou").id.iloc[0]
    m = next(x for x in cargar_reales() if x.id == peor)  # la foto real donde Otsu sin corregir falla
    img = m.imagen()
    g0 = preproceso.gris(img)
    g1 = preproceso.preprocesar(img, pre)
    a = segmentacion.segmentar(g0, "otsu")
    b = segmentacion.segmentar(g1, seg)
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.6))
    paneles = [(g0, "Foto en gris"), (preproceso.fondo_estimado(g0), "Papel estimado (sin golilla)"),
               (g1, f"Corregida («{pre}»)"), (_sobre(g0, a), f"Otsu SIN corregir la luz ({m.id})"),
               (recorte(_sobre(g1, b), *centro(b), 90, 300), f"«{seg}» con luz corregida")]
    for ax, (im_, t) in zip(axs, paneles):
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(im_, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t, fontsize=10)
    guardar(fig, "e01_ejemplo.png")
    print(t1.round(3).to_string())
    print(t2.round(3).to_string())


def _sobre(g, m):
    c = np.dstack([g] * 3).astype(np.float32)
    c[m > 0] = 0.55 * c[m > 0] + 0.45 * np.array([42, 120, 214])
    return c.astype(np.uint8)


if __name__ == "__main__":
    main()
