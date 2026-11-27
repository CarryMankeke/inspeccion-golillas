"""N3 del proceso 3 · Bordes: ¿qué detector ubica mejor el borde?

Con el preprocesado y la segmentación ya decididos, cada detector entrega puntos de
borde. En 80 golillas sintéticas buenas se conoce el círculo exacto, así que se mide:
- sesgo: cuánto se corren los puntos hacia adentro o afuera (px);
- dispersión: cuánto se desordenan (px);
- error de radio tras un ajuste fijo (Kåsa), exterior e interior (px).
En las 15 reales buenas no hay verdad exacta: se informa la rugosidad (desviación de
los puntos a su propio círculo) como dato secundario.

Decisión: menor suma de (error medio de radio + dispersión media de los puntos), en px;
empate (<0,05 px) → el más rápido.
Revisión del criterio (27 nov): el primero miraba solo el error de radio, y todos los
detectores de gradiente quedaron bajo 0,1 px: el radio no los distingue y ganaba
Canny por rapidez. Pero la dispersión de los puntos es el piso de ruido de los rasgos
del proceso 5 (desviación radial, Fourier): con menos ruido se ven defectos más
chicos. Por eso se suma. La rugosidad en fotos reales confirma el mismo orden.
"""

import sys
import time

import numpy as np
import pandas as pd
from comun import RESULTADOS, barras_h, decidir, decision, estilo, guardar, mapa
from matplotlib import pyplot as plt

from golillas import bordes, medicion, preproceso, segmentacion, sintetico
from golillas.datos import cargar_reales

N = 80
SEMILLA = 3


def evaluar(img, gt, fuente, id_):
    pre, seg = decision("preproceso"), decision("segmentacion")
    g = preproceso.preprocesar(img, pre)
    m = segmentacion.segmentar(g, seg)
    filas = []
    for b in bordes.OPCIONES:
        t0 = time.perf_counter()
        pe, pi = bordes.bordes(g, m, b)
        t = (time.perf_counter() - t0) * 1000
        f = dict(fuente=fuente, id=id_, borde=b, ms=t, n_ext=len(pe), n_int=len(pi))
        if len(pe) >= 5 and len(pi) >= 5:
            ce, ci = medicion.kasa(pe), medicion.kasa(pi)
            f.update(r_ext=ce.r, r_int=ci.r,
                     rug_ext=float(np.std(np.hypot(*(pe - ce.c).T) - ce.r)),
                     rug_int=float(np.std(np.hypot(*(pi - ci.c).T) - ci.r)))
            if gt is not None:
                de = np.hypot(*(pe - [gt.cx, gt.cy]).T) - gt.r_ext
                di = np.hypot(*(pi - [gt.cx, gt.cy]).T) - gt.r_int
                f.update(sesgo_ext=de.mean(), disp_ext=de.std(), sesgo_int=di.mean(), disp_int=di.std(),
                         err_r_ext=abs(ce.r - gt.r_ext), err_r_int=abs(ci.r - gt.r_int),
                         err_c=float(np.hypot(ce.cx - gt.cx, ce.cy - gt.cy)))
        filas.append(f)
    return filas


def trabajo_sint(i):
    s = sintetico.generar(SEMILLA * 100_000 + i, defecto="ninguno")
    return evaluar(s.imagen, s, "sintética", f"s{i:03d}")


def trabajo_real(m):
    return evaluar(m.imagen(), None, "real", m.id)


def main():
    estilo()
    csv = RESULTADOS / "e03_bordes_crudo.csv"
    if csv.exists() and "--recalcular" not in sys.argv:
        df = pd.read_csv(csv)
    else:
        reales = [m for m in cargar_reales() if not m.es_mala]
        df = pd.DataFrame(sum(mapa(trabajo_sint, list(range(N))), []) + sum(mapa(trabajo_real, reales), []))
        df.to_csv(csv, index=False)
    s, r = df[df.fuente == "sintética"], df[df.fuente == "real"]
    t = s.groupby("borde")[["sesgo_ext", "disp_ext", "sesgo_int", "disp_int", "err_r_ext", "err_r_int",
                            "err_c", "ms"]].mean()
    t["ms"] = s.groupby("borde")["ms"].median()
    t["err_r"] = 0.5 * (t.err_r_ext + t.err_r_int)
    t["disp"] = 0.5 * (t.disp_ext + t.disp_int)
    t["puntaje"] = t.err_r + t.disp
    t["rug_real"] = r.groupby("borde")[["rug_ext", "rug_int"]].mean().mean(axis=1)
    t["sd_r_ext_real"] = r.groupby("borde")["r_ext"].std()
    t = t.sort_values("puntaje")
    t.to_csv(RESULTADOS / "e03_bordes.csv")
    mejor = t.puntaje.min()
    elegido = t[t.puntaje <= mejor + 0.05].sort_values("ms").index[0]
    decidir("bordes", elegido,
            "menor (error de radio + dispersión de puntos) en sintéticas; empate <0,05 px → el más rápido",
            err_r_px=round(float(t.loc[elegido, "err_r"]), 3), disp_px=round(float(t.loc[elegido, "disp"]), 3),
            ms=round(float(t.loc[elegido, "ms"]), 2))

    orden = list(t.index)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    barras_h(axs[0], orden, [t.loc[orden, "err_r_ext"], t.loc[orden, "err_r_int"]],
             ["borde exterior", "borde del agujero"], elegido=elegido, fmt="{:.2f}")
    axs[0].set_title("Error de radio (px) · 80 sintéticas")
    barras_h(axs[1], orden, [t.loc[orden, "disp"], t.loc[orden, "rug_real"]],
             ["sintéticas (vs. verdad)", "reales (vs. su círculo)"], elegido=elegido, fmt="{:.2f}")
    axs[1].set_title("Dispersión de los puntos (px)")
    barras_h(axs[2], orden, [t.loc[orden, "ms"]], elegido=elegido, fmt="{:.1f}")
    axs[2].set_title("Tiempo (ms)")
    guardar(fig, "e03_bordes.png")
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
