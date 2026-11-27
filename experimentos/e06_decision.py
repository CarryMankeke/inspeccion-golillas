"""N3 del proceso 6 · Decisión: ¿qué regla o modelo decide aprobar o retirar?

Con los 4 rasgos elegidos se comparan 9 modelos en tres pruebas:
A. Fotos reales, dejando una fuera (LOO): se entrena con 20 y se prueba la que quedó
   fuera, 21 veces. Es la prueba honesta con datos reales.
R. Lo mismo, pero se prueban las 7 versiones giradas (90°, 180°, 270° y espejos, sin
   interpolar) de la foto que quedó fuera: 147 casos. Mide si la decisión aguanta que
   la luz y la pieza queden en otra orientación.
B. De sintético a real: se entrena solo con sintéticas y se prueba en las 21 reales.
C. Sintético a sintético: entrenamiento 600 → prueba 600 (otra semilla), por defecto.

Los modelos de una clase solo ven golillas buenas (como en planta, con piezas patrón).

Decisión: entre los que detectan 6/6 malas reales en A y ≥99 % en R, el de menos falsas
alarmas (promedio de A y R); empate → mejor (detección − falsas alarmas) en C; si sigue
el empate, se prefiere uno de una clase (no necesita juntar golillas malas).
Revisión (27 nov): la primera versión miraba solo A. Al simular la línea con fotos
giradas apareció un 2–7 % de buenas retiradas (casos de borde en relleno_anillo):
se agregó la prueba R y el umbral de Hotelling para muestras chicas.
El umbral k de las tolerancias se elige ANTES, solo con sintéticas de entrenamiento
(máxima exactitud balanceada), para no mirar las etiquetas reales.
"""

import json

import numpy as np
import pandas as pd
from comun import AZUL, GRIS, NARANJA, RESULTADOS, barras_h, decidir, decision, estilo, guardar
from matplotlib import pyplot as plt
from scipy.stats import beta

from golillas import decision as dec
from golillas.datos import RAIZ


def ic95(k, n):
    """Intervalo de Clopper-Pearson al 95 % para k aciertos de n."""
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def metricas(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    return dict(deteccion=pred[y].mean(), falsas=pred[~y].mean(), n_malas=int(y.sum()), n_buenas=int((~y).sum()),
                malas_ok=int(pred[y].sum()), buenas_mal=int(pred[~y].sum()))


def entrenar(modelo, X, y):
    return modelo.fit(X[~y], X, y.astype(int))


def main():
    estilo()
    df = pd.read_csv(RESULTADOS / "e05_rasgos.csv")
    F = decision("rasgos").split(",")
    tr, te, re, rg = (df[df.split == s] for s in ("entrenamiento", "prueba", "real", "real_giro"))
    Xrg, yrg = rg[F].values, (rg.clase == "mala").values
    Xtr, ytr = tr[F].values, (tr.clase == "mala").values
    Xte, yte = te[F].values, (te.clase == "mala").values
    Xre, yre = re[F].values, (re.clase == "mala").values

    # k de las tolerancias: solo con sintéticas de entrenamiento
    ks = [2, 3, 4, 5, 6, 8, 10, 15, 20]
    barrido = []
    for k in ks:
        t = dec.Tolerancias(k).fit(Xtr[~ytr])
        m = metricas(ytr, t.predecir(Xtr))
        barrido.append(dict(k=k, **m, bal=0.5 * (m["deteccion"] + 1 - m["falsas"])))
    barrido = pd.DataFrame(barrido)
    k_elegido = int(barrido.sort_values(["bal", "k"], ascending=[False, False]).k.iloc[0])
    barrido.to_csv(RESULTADOS / "e06_barrido_k.csv", index=False)

    filas = []
    for nombre, modelo in dec.modelos().items():
        if nombre == "tolerancias":
            modelo.k = k_elegido
        # A: LOO real
        pred, pred_r = np.zeros(len(yre), bool), np.zeros(len(yrg), bool)
        for i, id_ in enumerate(re.id.values):
            fuera = np.arange(len(yre)) != i
            mod = entrenar(modelo, Xre[fuera], yre[fuera])
            pred[i] = mod.predecir(Xre[i : i + 1])[0]
            sel = (rg.id == id_).values
            pred_r[sel] = mod.predecir(Xrg[sel])
        a = metricas(yre, pred)
        rr = metricas(yrg, pred_r)
        # B: sintético → real
        b = metricas(yre, entrenar(modelo, Xtr, ytr).predecir(Xre))
        # C: sintético → sintético
        pc = entrenar(modelo, Xtr, ytr).predecir(Xte)
        c = metricas(yte, pc)
        por_def = {f"C_{d}": pc[(te.defecto == d).values].mean() for d in te.defecto.unique() if d != "ninguno"}
        filas.append(dict(modelo=nombre, una_clase=modelo.una_clase,
                          **{f"A_{k}": v for k, v in a.items()}, **{f"R_{k}": v for k, v in rr.items()},
                          **{f"B_{k}": v for k, v in b.items()},
                          **{f"C_{k}": v for k, v in c.items()}, **por_def))
    T = pd.DataFrame(filas).set_index("modelo")
    T["C_margen"] = T.C_deteccion - T.C_falsas
    T.to_csv(RESULTADOS / "e06_decision.csv")

    T["AR_falsas"] = 0.5 * (T.A_falsas + T.R_falsas)
    cand = T[(T.A_deteccion >= 1.0) & (T.R_deteccion >= 0.99)].copy()
    if len(cand):
        cand = cand.sort_values(["AR_falsas", "C_margen", "una_clase"], ascending=[True, False, False])
        criterio = ("6/6 malas en LOO real y ≥99 % en giros; menos falsas alarmas (LOO + giros); "
                    "empate → mejor margen sintético; una clase")
    else:
        cand = T.assign(bal=0.5 * (T.A_deteccion + 1 - T.A_falsas)).sort_values("bal", ascending=False)
        criterio = "ninguno detectó 6/6: mayor exactitud balanceada en LOO real"
    elegido = cand.index[0]
    fila = T.loc[elegido]
    lo, hi = ic95(int(fila.A_malas_ok), int(fila.A_n_malas))
    lo2, hi2 = ic95(int(fila.A_buenas_mal), int(fila.A_n_buenas))
    decidir("decision", elegido, criterio, k=k_elegido if elegido == "tolerancias" else None,
            loo_deteccion=f"{int(fila.A_malas_ok)}/{int(fila.A_n_malas)} (IC95 {lo:.0%}–{hi:.0%})",
            loo_falsas=f"{int(fila.A_buenas_mal)}/{int(fila.A_n_buenas)} (IC95 {lo2:.0%}–{hi2:.0%})",
            giros_deteccion=f"{int(fila.R_malas_ok)}/{int(fila.R_n_malas)}",
            giros_falsas=f"{int(fila.R_buenas_mal)}/{int(fila.R_n_buenas)}",
            sint_deteccion=round(float(fila.C_deteccion), 3), sint_falsas=round(float(fila.C_falsas), 3))

    # calibración final del inspector: el modelo elegido, entrenado con las 15 buenas reales
    final = dec.modelos()[elegido]
    if elegido == "tolerancias":
        final.k = k_elegido
    final.fit(Xre[~yre], Xre, yre.astype(int))
    explica = dec.Tolerancias(3).fit(Xre[~yre])  # para decir QUÉ rasgo se salió
    cal = dict(rasgos=F, modelo=elegido, parametros=final.parametros(),
               explicacion=dict(mediana=explica.med.tolist(), escala=explica.esc.tolist(), k=3),
               origen="15 golillas buenas reales (data/real/buenas)",
               decisiones={p: v["opcion"] for p, v in json.loads((RESULTADOS / "decisiones.json").read_text()).items()})
    (RAIZ / "golillas" / "calibracion.json").write_text(json.dumps(cal, indent=2, ensure_ascii=False))

    # gráfico
    orden = list(T.sort_values(["A_deteccion", "AR_falsas"], ascending=[False, True]).index)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, p, titulo in zip(axs, "ARC", ["A · reales, dejando una fuera (6 malas, 15 buenas)",
                                          "R · las mismas, giradas (42 malas, 105 buenas)",
                                          "C · sintético → sintético (600)"]):
        barras_h(ax, orden, [100 * T.loc[orden, f"{p}_deteccion"], 100 * T.loc[orden, f"{p}_falsas"]],
                 ["malas detectadas (%)", "buenas retiradas (%)"], colores=[AZUL, NARANJA], elegido=elegido,
                 fmt="{:.0f}", lim=(0, 118))
        ax.set_title(titulo, fontsize=10)
    guardar(fig, "e06_decision.png")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(barrido.k, 100 * barrido.deteccion, "-o", color=AZUL, lw=2, ms=5, label="malas detectadas")
    ax.plot(barrido.k, 100 * barrido.falsas, "-o", color=NARANJA, lw=2, ms=5, label="buenas retiradas")
    ax.axvline(k_elegido, color=GRIS, lw=1, ls="--")
    ax.text(k_elegido, 50, f"  k = {k_elegido}", color="#52514e")
    ax.set_xlabel("k (cuántas desviaciones se toleran)")
    ax.set_ylabel("%")
    ax.set_title("Tolerancias: elegir k con sintéticas de entrenamiento")
    ax.legend(loc="center right", fontsize=8)
    guardar(fig, "e06_barrido_k.png")
    cols = ["una_clase", "A_deteccion", "A_falsas", "R_deteccion", "R_falsas", "B_deteccion", "B_falsas",
            "C_deteccion", "C_falsas"]
    print(T[cols].round(3).to_string())
    print(T[[c for c in T.columns if c.startswith("C_") and c not in
             ("C_deteccion", "C_falsas", "C_margen") and not c.startswith("C_n") and "_ok" not in c
             and "_mal" not in c]].round(2).to_string())
    print(barrido.round(3).to_string())


if __name__ == "__main__":
    main()
