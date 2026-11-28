"""Embudo de selección (ADR 0019): poda por dominancia y cota del oráculo sobre los puntajes de e13. SOLO INFORMA.

En MANTA la prueba de e13 ya se miró, así que nada de lo que salga aquí decide. Para que decida, la poda se hace
con datos de desarrollo y después va una prueba sellada nueva (ADR 0019). No usa GPU: lee resultados/e13_metodos*.csv.

Qué calcula en cada tipo de golilla:
1. **Punto ROC con 5 % de falsas o menos** de cada método: el umbral más permisivo que retira a lo más
   floor(0,05 · buenas) buenas de la prueba (7 de 148 o de 144, el criterio AIAG del ADR 0017). Es un oráculo,
   porque elige el umbral mirando la prueba: da la mejor detección posible del método con ese requisito de falsas.
2. **Dominancia (frente de Pareto)**, con los umbrales conformales de e13: A domina a B si, en las dos golillas,
   no es peor en detección ni en falsas ni en cumplimiento de latencia, y es mejor en al menos una de ellas.
   «Mejor» o «peor» en detección o falsas cuenta solo si la prueba exacta de McNemar sobre las piezas en que
   discrepan da p < 0,05 (Dietterich, 1998, Neural Computation 10(7)): no gana nadie por 1 pieza de 74. La latencia
   cuenta solo contra el requisito (≤ 100 ms o no). Todas las licencias de la lista ya pasaron el filtro del ADR 0017.
   La dominancia poda candidatos solos, no miembros de una combinación: un método dominado puede atrapar malas que
   el otro no ve.
3. **Cota del oráculo para «retira si A o B o …»**: con los umbrales del punto 1, la detección del conjunto es la
   unión de las malas que detecta cada uno. Ninguna regla «o» con 5 % de falsas o menos detecta más: sus falsas
   son al menos las de cada miembro, así que cada umbral tiene que ser igual o más estricto que el del punto 1, y
   subir un umbral solo pierde malas. Se recorre el retículo de conjuntos de menor a mayor y se poda (sin probarlo)
   todo superconjunto de uno que ya pasa de 100 ms en serie, o de 5 % de falsas con los umbrales fijos de e13.
4. **Fusiones de puntajes**: las malas que pierden todos a la vez en el punto 1 y el solapamiento (Jaccard) de las
   malas perdidas por cada par. Es un diagnóstico, no una cota: una suma de puntajes puede rescatar una pieza que
   cada método pierde por separado.

Uso:  python scripts/combinaciones.py          (MANTA, después de e13)
      python scripts/combinaciones.py --humo   (con los CSV del humo, solo para probar el script)
"""

from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"
HUMO = "--humo" in sys.argv
SUF = "_humo" if HUMO else ""
DECIDEN = ("kaggle", "edge_impulse") if HUMO else ("manta_red_washer", "manta_yellow_green_washer")
FALSAS_MAX, NO_DET_MAX, MS_MAX = 0.05, 0.02, 100.0  # criterio AIAG «aceptable» y latencia del ADR 0017
P_MCNEMAR = 0.05  # dos colas, la misma confianza (95 %) de los IC del ADR 0017


def leer():
    P = pd.read_csv(RESULTADOS / f"e13_metodos_puntajes{SUF}.csv")
    T = pd.read_csv(RESULTADOS / f"e13_metodos{SUF}.csv")
    ms = {(r.conjunto, r.metodo): r.ms_cpu for r in T.itertuples()}
    fuera = [f"{r.conjunto}·{r.metodo}: {'no aplica' if not r.aplica else 'error'}" for r in T.itertuples()
             if not r.aplica or isinstance(getattr(r, "error", None), str)]
    return P, ms, fuera


def punto_roc(s: np.ndarray, mala: np.ndarray) -> tuple[float, int, int]:
    """(umbral, malas detectadas, buenas retiradas) con el umbral más permisivo de ≤ floor(5 % · buenas) falsas.

    Un puntaje infinito (la clásica no logró medir la pieza) se retira siempre.
    """
    b = np.sort(s[~mala])[::-1]
    k = math.floor(FALSAS_MAX * len(b))
    n_inf = int(np.isinf(b).sum())
    if n_inf > k:
        return math.inf, int(np.isinf(s[mala]).sum()), n_inf
    u = b[k] if k < len(b) else -math.inf
    return float(u), int((s[mala] > u).sum()), int((b > u).sum())


def mcnemar(a_bien: np.ndarray, b_bien: np.ndarray) -> int:
    """+1 si A acierta significativamente más piezas que B, −1 si menos, 0 si no se distinguen."""
    n10, n01 = int((a_bien & ~b_bien).sum()), int((~a_bien & b_bien).sum())
    if n10 + n01 == 0 or binomtest(n10, n10 + n01).pvalue >= P_MCNEMAR:
        return 0
    return 1 if n10 > n01 else -1


def main():
    P, ms, fuera = leer()
    metodos = [m for m in P.metodo.unique() if all(((c, m) in ms) and ((P.conjunto == c) & (P.metodo == m)).any()
                                                   for c in DECIDEN)]
    R, D = {}, {}
    filas_roc, perdidas, pares = [], [], []
    for c in DECIDEN:
        Q = P[P.conjunto == c]
        piezas = Q[Q.metodo == metodos[0]].pieza.tolist()
        mala = Q[Q.metodo == metodos[0]].set_index("pieza").loc[piezas, "mala"].to_numpy(bool)
        defecto = Q[Q.metodo == metodos[0]].set_index("pieza").loc[piezas, "defecto"].to_numpy()
        S = {m: Q[Q.metodo == m].set_index("pieza").loc[piezas] for m in metodos}
        R[c] = {m: S[m].retira.to_numpy(bool) for m in metodos}  # decisión con el umbral conformal de e13
        D[c] = {}
        for m in metodos:
            u, det, fa = punto_roc(S[m].s.to_numpy(float), mala)
            D[c][m] = S[m].s.to_numpy(float) > u
            filas_roc.append(dict(conjunto=c, metodo=m, umbral_5=u, malas_5=det, falsas_5=fa, n_malas=int(mala.sum()),
                                  n_buenas=int((~mala).sum()), malas_e13=int(R[c][m][mala].sum()),
                                  falsas_e13=int(R[c][m][~mala].sum()), ms_cpu=ms[(c, m)]))
        perdida_todos = mala & ~np.any([D[c][m] for m in metodos], axis=0)
        perdidas += [dict(conjunto=c, pieza=p, defecto=d) for p, d in zip(np.array(piezas)[perdida_todos],
                                                                           defecto[perdida_todos])]
        for a, b in combinations(metodos, 2):
            pa, pb = mala & ~D[c][a], mala & ~D[c][b]
            union = int((pa | pb).sum())
            pares.append(dict(conjunto=c, a=a, b=b, perdidas_a=int(pa.sum()), perdidas_b=int(pb.sum()),
                              perdidas_ambos=int((pa & pb).sum()), jaccard=(pa & pb).sum() / union if union else 1.0))
        R[c]["_mala"], D[c]["_mala"] = mala, mala

    # ---- 2. dominancia, con los umbrales conformales de e13
    def lat_ok(m):
        return all(ms[(c, m)] <= MS_MAX for c in DECIDEN)

    dom = []
    for a in metodos:
        for b in metodos:
            if a == b:
                continue
            signos = []
            for c in DECIDEN:
                y = R[c]["_mala"]
                signos.append(mcnemar(R[c][a][y], R[c][b][y]))  # detección: acertar = retirar una mala
                signos.append(mcnemar(~R[c][a][~y], ~R[c][b][~y]))  # falsas: acertar = dejar pasar una buena
            signos.append(int(lat_ok(a)) - int(lat_ok(b)))
            if min(signos) >= 0 and max(signos) > 0:
                dom.append(dict(domina=a, dominado=b))
    dominados = sorted({d["dominado"] for d in dom})

    # ---- 3. retículo de conjuntos «A o B o …»
    def k_det(c):
        n = int(R[c]["_mala"].sum())
        return n - math.floor(NO_DET_MAX * n)  # malas que hay que detectar (73 de 74, 71 de 72)

    def evaluar(conj):
        f = {}
        for c in DECIDEN:
            y = R[c]["_mala"]
            ret_e13 = np.any([R[c][m] for m in conj], axis=0)
            ret_5 = np.any([D[c][m] for m in conj], axis=0)
            f[c] = dict(ms=sum(ms[(c, m)] for m in conj), malas_e13=int(ret_e13[y].sum()),
                        falsas_e13=int(ret_e13[~y].sum()), n_buenas=int((~y).sum()), cota_malas=int(ret_5[y].sum()),
                        n_malas=int(y.sum()), alcanza=bool(ret_5[y].sum() >= k_det(c)))
        return f

    def podado(f):
        lat = any(v["ms"] > MS_MAX for v in f.values())
        fa = any(v["falsas_e13"] > math.floor(FALSAS_MAX * v["n_buenas"]) for v in f.values())
        return lat, fa

    filas, vivos, n_podados = [], [()], 0
    for k in range(1, len(metodos) + 1):
        candidatos = sorted({tuple(sorted(set(v) | {m})) for v in vivos for m in metodos if m not in v})
        vivos_set = set(vivos)
        nuevos = []
        for conj in candidatos:
            if k > 1 and any(sub not in vivos_set for sub in combinations(conj, k - 1)):
                n_podados += 1  # contiene un conjunto que ya pasa de 100 ms o de las falsas: no se prueba
                continue
            f = evaluar(conj)
            lat, fa = podado(f)
            filas.append(dict(conjunto=" o ".join(conj), n=k, pasa_100ms=lat, pasa_falsas_e13=fa,
                              alcanza_cota=all(v["alcanza"] for v in f.values()),
                              **{f"{c}·{x}": v[x] for c, v in f.items() for x in ("ms", "cota_malas", "malas_e13",
                                                                                   "falsas_e13")}))
            if not (lat or fa):
                nuevos.append(conj)
        vivos = nuevos
        if not vivos:
            break
    C = pd.DataFrame(filas)
    todos = evaluar(tuple(metodos))  # sin tope de latencia: ¿alcanza la detección aunque el hardware fuera más rápido?

    # ---- salida
    pd.DataFrame(filas_roc).to_csv(RESULTADOS / f"e13_punto5{SUF}.csv", index=False)
    pd.DataFrame(dom).to_csv(RESULTADOS / f"e13_dominancia{SUF}.csv", index=False)
    C.to_csv(RESULTADOS / f"e13_combinaciones{SUF}.csv", index=False)
    pd.DataFrame(pares).to_csv(RESULTADOS / f"e13_perdidas_pares{SUF}.csv", index=False)
    pd.DataFrame(perdidas).to_csv(RESULTADOS / f"e13_perdidas_todos{SUF}.csv", index=False)
    caben = C[~C.pasa_100ms & ~C.pasa_falsas_e13]
    resumen = dict(
        solo_informa=True, deciden=list(DECIDEN), metodos=metodos, fuera=fuera,
        frente_pareto=[m for m in metodos if m not in dominados], dominados=dominados,
        conjuntos_probados=len(C), conjuntos_podados_sin_probar=n_podados,
        conjuntos_que_caben=caben.conjunto.tolist(),
        algun_conjunto_que_cabe_alcanza=bool(caben.alcanza_cota.any()) if len(caben) else False,
        sin_tope_de_latencia={c: f"{v['cota_malas']}/{v['n_malas']} malas (hacen falta {k_det(c)})"
                              for c, v in todos.items()},
        sin_tope_alcanza=all(v["alcanza"] for v in todos.values()),
        perdidas_por_todos={c: int(sum(p["conjunto"] == c for p in perdidas)) for c in DECIDEN},
    )
    (RESULTADOS / f"e13_combinaciones{SUF}.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False))
    print(pd.DataFrame(filas_roc).to_string(index=False))
    print(json.dumps(resumen, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
