"""N3 del proceso 4 · Medición: ¿qué ajuste de círculo es exacto y además robusto?

Dos pruebas con golillas sintéticas (círculo nominal conocido):
1. Exactitud: 80 golillas buenas → error de radio y de centro.
2. Robustez: 40 golillas por cada defecto local (muesca, mordida, borde del agujero,
   corte). Un buen ajuste mide el círculo NOMINAL sin dejarse arrastrar por el defecto,
   para que el defecto quede a la vista como residuo.

Decisión: menor promedio entre el error en buenas y el error con defecto; empate
(<0,05 px) → el más rápido.
"""

import sys
import time

import numpy as np
import pandas as pd
from comun import RESULTADOS, barras_h, decidir, decision, estilo, guardar, mapa
from matplotlib import pyplot as plt

from golillas import bordes, medicion, preproceso, segmentacion, sintetico

SEMILLA = 4
CASOS = [("ninguno", 80), ("muesca_exterior", 40), ("mordida_exterior", 40), ("borde_interior", 40),
         ("corte_radial", 40)]
CIRCULO = {"muesca_exterior": "ext", "mordida_exterior": "ext", "borde_interior": "int", "corte_radial": "int"}


def trabajo(arg):
    defecto, i = arg
    k = [c[0] for c in CASOS].index(defecto)  # semilla fija por defecto (hash() cambia entre corridas)
    s = sintetico.generar(SEMILLA * 100_000 + k * 1000 + i, defecto=defecto)
    g = preproceso.preprocesar(s.imagen, decision("preproceso"))
    m = segmentacion.segmentar(g, decision("segmentacion"))
    pe, pi = bordes.bordes(g, m, decision("bordes"))
    filas = []
    for metodo in medicion.OPCIONES:
        t0 = time.perf_counter()
        ce, ci = medicion.medir(g, m, pe, pi, metodo)
        t = (time.perf_counter() - t0) * 1000
        e_ext = abs(ce.r - s.r_ext) + np.hypot(ce.cx - s.cx, ce.cy - s.cy)
        e_int = abs(ci.r - s.r_int) + np.hypot(ci.cx - s.cx, ci.cy - s.cy)
        if defecto == "ninguno":
            err = 0.5 * (e_ext + e_int)
        else:
            err = e_ext if CIRCULO[defecto] == "ext" else e_int
        filas.append(dict(defecto=defecto, id=i, metodo=metodo, err=err if np.isfinite(err) else 99.0,
                          err_r_ext=abs(ce.r - s.r_ext), err_r_int=abs(ci.r - s.r_int), ms=t))
    return filas


def main():
    estilo()
    csv = RESULTADOS / "e04_medicion_crudo.csv"
    if csv.exists() and "--recalcular" not in sys.argv:
        df = pd.read_csv(csv)
    else:
        trabajos = [(d, i) for d, n in CASOS for i in range(n)]
        df = pd.DataFrame(sum(mapa(trabajo, trabajos), []))
        df.to_csv(csv, index=False)
    piv = df.groupby(["metodo", "defecto"])["err"].median().unstack("defecto")
    t = pd.DataFrame({
        "buenas": piv["ninguno"],
        "con_defecto": piv.drop(columns="ninguno").mean(axis=1),
        "ms": df.groupby("metodo")["ms"].median(),
    })
    t = t.join(piv.drop(columns="ninguno"))
    t["puntaje"] = 0.5 * (t.buenas + t.con_defecto)
    t = t.sort_values("puntaje")
    t.to_csv(RESULTADOS / "e04_medicion.csv")
    mejor = t.puntaje.min()
    elegido = t[t.puntaje <= mejor + 0.05].sort_values("ms").index[0]
    decidir("medicion", elegido,
            "menor promedio entre error en buenas y error con defecto (mediana, px); empate <0,05 → el más rápido",
            err_buenas_px=round(float(t.loc[elegido, "buenas"]), 3),
            err_defecto_px=round(float(t.loc[elegido, "con_defecto"]), 3))

    orden = list(t.index)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, gridspec_kw={"width_ratios": [2, 1]})
    barras_h(axs[0], orden, [t.loc[orden, "buenas"], t.loc[orden, "con_defecto"]],
             ["golillas buenas (80)", "con defecto local (160)"], elegido=elegido, fmt="{:.2f}")
    axs[0].set_xscale("log")
    axs[0].set_title("Error del círculo nominal (px, mediana; escala log)")
    barras_h(axs[1], orden, [t.loc[orden, "ms"]], elegido=elegido, fmt="{:.2f}")
    axs[1].set_title("Tiempo (ms)")
    guardar(fig, "e04_medicion.png")
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
