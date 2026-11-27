"""N3 del proceso 5 · Rasgos: ¿qué números separan las golillas malas de las buenas?

Se calculan los 17 rasgos de `golillas/rasgos.py` con la cadena ya decidida
(preprocesado → segmentación → bordes → RANSAC) en:
- 600 sintéticas de entrenamiento (300 buenas + 50 de cada uno de los 6 defectos),
- 600 sintéticas de prueba (otra semilla; se usan recién en el proceso 6),
- las 21 fotos reales.

Cada rasgo se convierte en un puntaje de anomalía: cuántas desviaciones (MAD) se aleja
de la mediana de las buenas. El AUC dice qué tan bien ese puntaje ordena malas sobre
buenas (1 = perfecto, 0,5 = al azar), por tipo de defecto.

Decisión: selección hacia adelante en las sintéticas de ENTRENAMIENTO. En cada paso se
agrega el rasgo que más sube el AUC medio por defecto del puntaje combinado (el
máximo de los rasgos elegidos); se para cuando mejora <0,002 o hay 6 rasgos.
`radio_ext` queda fuera: depende de la altura de la cámara (es tolerancia de tamaño,
no de forma).
"""

import sys
import time

import numpy as np
import pandas as pd
from comun import RESULTADOS, SECUENCIAL, decidir, decision, estilo, guardar, mapa
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score

from golillas import bordes, medicion, preproceso, rasgos, segmentacion, sintetico
from golillas.datos import cargar_reales

CANDIDATOS = [r for r in rasgos.NOMBRES if r != "radio_ext"]
LISTA = ["ninguno"] * 300 + [d for d in sintetico.DEFECTOS for _ in range(50)]
SEMILLAS = {"entrenamiento": 5, "prueba": 6}


def extraer(img):
    t = {}
    t0 = time.perf_counter()
    g = preproceso.preprocesar(img, decision("preproceso"))
    t["ms_pre"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    m = segmentacion.segmentar(g, decision("segmentacion"))
    t["ms_seg"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    pe, pi = bordes.bordes(g, m, decision("bordes"))
    t["ms_bordes"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    ce, ci = medicion.medir(g, m, pe, pi, decision("medicion"))
    t["ms_medicion"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    r = rasgos.calcular(g, m, pe, pi, ce, ci)
    t["ms_rasgos"] = (time.perf_counter() - t0) * 1000
    return {**r, **t}


def trabajo_sint(arg):
    split, i = arg
    s = sintetico.generar(SEMILLAS[split] * 100_000 + i, defecto=LISTA[i])
    return dict(fuente="sintética", split=split, id=f"{split[0]}{i:03d}", clase=s.clase, defecto=s.defecto,
                **extraer(s.imagen))


def trabajo_real(m):
    return dict(fuente="real", split="real", id=m.id, clase=m.clase, defecto=m.defecto, **extraer(m.imagen()))


def girar(img, k):
    """Las 8 simetrías exactas del cuadrado (giros de 90° y espejo): sin interpolar."""
    img = np.rot90(img, k % 4)
    return np.ascontiguousarray(img[:, ::-1] if k >= 4 else img)


def trabajo_giro(arg):
    m, k = arg
    return dict(fuente="real", split="real_giro", id=m.id, variante=k, clase=m.clase, defecto=m.defecto,
                **extraer(girar(m.imagen(), k)))


def z(X, buenas):
    med = np.nanmedian(buenas, axis=0)
    esc = 1.4826 * np.nanmedian(np.abs(buenas - med), axis=0)
    esc = np.maximum(esc, 1e-3 * np.maximum(np.abs(med), 1e-3))
    s = np.abs(X - med) / esc
    return np.where(np.isfinite(s), s, 1e3)


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def main():
    estilo()
    csv = RESULTADOS / "e05_rasgos.csv"
    if csv.exists() and "--recalcular" not in sys.argv:
        df = pd.read_csv(csv)
    else:
        trabajos = [(sp, i) for sp in SEMILLAS for i in range(len(LISTA))]
        reales = cargar_reales()
        df = pd.DataFrame(mapa(trabajo_sint, trabajos) + mapa(trabajo_real, reales)
                          + mapa(trabajo_giro, [(m, k) for m in reales for k in range(1, 8)]))
        df.to_csv(csv, index=False)

    tr = df[df.split == "entrenamiento"]
    re = df[df.split == "real"]
    Ztr = pd.DataFrame(z(tr[CANDIDATOS].values, tr[tr.clase == "buena"][CANDIDATOS].values),
                       columns=CANDIDATOS, index=tr.index)
    Zre = pd.DataFrame(z(re[CANDIDATOS].values, re[re.clase == "buena"][CANDIDATOS].values),
                       columns=CANDIDATOS, index=re.index)

    # AUC por rasgo y tipo de defecto
    tabla = {}
    for d in sintetico.DEFECTOS:
        sel = tr.defecto.isin(["ninguno", d])
        y = (tr.defecto[sel] == d).values.astype(int)
        tabla[d] = {f: auc(y, Ztr.loc[sel, f].values) for f in CANDIDATOS}
    y_re = (re.clase == "mala").values.astype(int)
    tabla["REAL (6 malas)"] = {f: auc(y_re, Zre[f].values) for f in CANDIDATOS}
    T = pd.DataFrame(tabla)

    # selección hacia adelante
    def objetivo(cols):
        s = Ztr[cols].max(axis=1).values
        return np.mean([auc((tr.defecto[tr.defecto.isin(["ninguno", d])] == d).values.astype(int),
                             s[tr.defecto.isin(["ninguno", d]).values]) for d in sintetico.DEFECTOS])

    elegidos, historial, actual = [], [], 0.0
    while len(elegidos) < 6:
        cand = {f: objetivo(elegidos + [f]) for f in CANDIDATOS if f not in elegidos}
        f, v = max(cand.items(), key=lambda kv: kv[1])
        if v - actual < 0.002:
            break
        elegidos.append(f)
        historial.append((f, v))
        actual = v
    T["elegido"] = [f in elegidos for f in T.index]
    T.to_csv(RESULTADOS / "e05_auc_rasgos.csv")
    pd.DataFrame(historial, columns=["rasgo", "auc_medio"]).to_csv(RESULTADOS / "e05_seleccion.csv", index=False)
    s_re = Zre[elegidos].max(axis=1).values
    decidir("rasgos", ",".join(elegidos),
            "selección hacia adelante por AUC medio por defecto (sintéticas de entrenamiento)",
            auc_sint=round(actual, 4), auc_real=round(float(auc(y_re, s_re)), 4))

    # mapa de calor
    orden = elegidos + [f for f in T.drop(columns="elegido").mean(axis=1).sort_values(ascending=False).index
                        if f not in elegidos]
    M = T.loc[orden].drop(columns="elegido")
    fig, ax = plt.subplots(figsize=(10, 6.2))
    cmap = LinearSegmentedColormap.from_list("azul", SECUENCIAL)
    im = ax.imshow(M.values, cmap=cmap, vmin=0.5, vmax=1, aspect="auto")
    ax.set_xticks(range(M.shape[1]), [c.replace("_", " ") for c in M.columns], rotation=25, ha="right")
    ax.set_yticks(range(M.shape[0]), [f"★ {f}" if f in elegidos else f for f in M.index])
    for t in ax.get_yticklabels():
        if t.get_text().startswith("★"):
            t.set_fontweight("bold")
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            ax.text(j, i, "–" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if np.isfinite(v) and v > 0.8 else "#0b0b0b")
    ax.axhline(len(elegidos) - 0.5, color="#eb6834", lw=1.5)
    fig.colorbar(im, ax=ax, label="AUC (1 = separa perfecto; 0,5 = al azar)", shrink=0.8)
    ax.set_title(f"Qué rasgo delata cada defecto · ★ = los {len(elegidos)} elegidos (sobre la línea naranja)")
    guardar(fig, "e05_rasgos.png")

    # tiempos por etapa (todas las imágenes)
    etapas = ["ms_pre", "ms_seg", "ms_bordes", "ms_medicion", "ms_rasgos"]
    tiempos = df[etapas].median()
    tiempos.to_csv(RESULTADOS / "e05_tiempos.csv")
    print(T.round(3).to_string())
    print(historial)
    print("tiempos (mediana ms):", tiempos.round(2).to_dict(), "total", round(df[etapas].sum(axis=1).median(), 1))


if __name__ == "__main__":
    main()
