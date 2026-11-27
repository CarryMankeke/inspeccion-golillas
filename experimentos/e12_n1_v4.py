"""N1 v4 · e12: umbral conformal y una regla de respaldo que respeta las falsas alarmas (ADR 0015).

Qué cambia frente a e11 (ADR 0014):
- Umbral conformal partido con α = 3 %: el puntaje número ⌈(n+1)(1−α)⌉ de n buenas que
  calibran y que NO ajustaron el modelo (golillas.decision.umbral_conformal). Una buena nueva
  del mismo montaje lo supera con probabilidad ≤ α.
- Regla de respaldo: si ninguno cumple 1 y 2, la mayor detección media entre los que cumplen
  2; si ninguno cumple 2, la menor tasa media de falsas alarmas. En los dos casos se informa
  «ninguno cumple» y no se adopta.
- Cada tasa va con su IC95 exacto (Clopper-Pearson).

Candidatos (las configuraciones de e11, sin búsqueda): clasica_4 (los 4 rasgos de los ADR
0010 y 0011), clasica_v3_6, clasica_v3_7, los 4 PatchCore de e11 y la fusión de la mejor
clásica con el mejor PatchCore, elegidos dentro de cada familia con la misma regla.

Partición de las buenas de cada montaje, al azar con semilla 0 (np.random.default_rng(0) para
cada montaje, sobre el orden de e11):
- sintético: 300 buenas de entrenamiento → 150 ajustan y 150 calibran; se mide en las 600 de
  validación (semilla 9).
- kaggle: 1.000 buenas de entrenamiento → 500 ajustan y 500 calibran; se mide en las 200 de prueba.
- ei: 273 buenas de entrenamiento → tres tercios de 91: ajustan, calibran y miden falsas alarmas.
- propio: 15 buenas no alcanzan para α = 3 % (hacen falta 33). Se usan las reglas de e11:
  Hotelling en la clásica y el máximo del LOO interno en PatchCore, dejando una fuera (A) y
  con los giros (R).
Ajustar = la media y la covarianza de la clásica, o el banco de PatchCore (sin giros, porque
todos los grupos que ajustan tienen 50 buenas o más). En la fusión, cada parte usa α/2 = 1,5 %
para que las dos juntas no pasen del 3 %; en el montaje propio sus partes usan las reglas de
e11, igual que la fusión de e11. Una pieza que la cadena clásica no logra medir se retira
siempre en las clásicas (z = ∞) y no ajusta ni calibra, como en e11.

Criterio (ADR 0015, fijado antes de correr; los requisitos son los del ADR 0014):
1. Propio: 6/6 en A, ≥ 95 % de malas en R y < 100 ms por pieza en CPU.
2. Falsas: ≤ 1/15 propias (A), ≤ 5 % en las 100 buenas de prueba de Kaggle y ≤ 5 % en el
   tercio de Edge Impulse que mide.
Gana la mayor detección media (malas de Kaggle y sintéticas de validación) entre los que
cumplen 1 y 2. Empate (≤ 2 puntos): clásica, luego fusión, luego PatchCore; después, la más
rápida en CPU. Respaldo, con el mismo desempate: la mayor detección media entre los que
cumplen 2; si ninguno cumple 2, la menor tasa media de falsas alarmas (el promedio de las
tres tasas del requisito 2).

Solo para informar, después de guardar la elección: el set de prueba de Edge Impulse (191
fotos, NO CIEGO porque ya se vio en e10), con el modelo del tercio que ajusta y el umbral del
tercio que calibra.
Reusa los rasgos y recortes de e11: las mismas fotos, la misma cadena y la misma regla de
polaridad (si no están, los calcula). Control: en el montaje propio, los 6 candidatos de e11
deben repetir sus números de e11, porque el código y las reglas son los mismos.
Corre en norea-node (GPU; mirar `nvidia-smi` antes). Escribe la decisión "enfoque_v4".
`--humo` corre con las muestras chicas de e11 para probar el código. Usa α = 10 %, porque con
20–30 buenas que calibran el umbral del 3 % no existe. No guarda la decisión y escribe sus
archivos con el sufijo _humo.
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from comun import AQUA, AZUL, GRIS, NARANJA, RESULTADOS, TINTA2, barras_h, decidir, estilo, guardar
from e09_ajuste_redes import DEV, auc, auc_ic, latencia
from e11_n1_v3 import CLASICAS as CLASICAS_V3
from e11_n1_v3 import (
    EXPLICA,
    F4,
    HUMO,
    PATCHCORES,
    SUF,
    ajustar_maha,
    banco,
    cadena,
    familia,
    imagen,
    latencias,
    pos,
    preparar,
    sel,
)
from matplotlib import pyplot as plt
from scipy.stats import beta, binomtest

from golillas.decision import umbral_conformal

ALFA = 0.10 if HUMO else 0.03
CLASICAS = {"clasica_4": F4, **CLASICAS_V3}
ORDEN = list(CLASICAS) + list(PATCHCORES) + ["fusion"]
CONFORMALES = (("sintetico", "sint"), ("kaggle", "kaggle"), ("ei", "ei_tercio"))
PRUEBAS = ("A", "R", "sint", "kaggle", "ei_tercio")
CLAVE = ["prueba", "id", "giro", "mala"]


# ---------------------------------------------------------------- partición y utilidades
def particion(R):
    """Papel de cada buena que calibra: 'ajusta', 'calibra' o 'mide' (este último, solo en Edge Impulse)."""
    papel = np.full(len(R), "", dtype=object)
    for montaje, _ in CONFORMALES:
        idx = np.random.default_rng(0).permutation(pos((R.montaje == montaje) & (R.grupo == "calibracion")))
        for nombre, p in zip(("ajusta", "calibra", "mide"), np.array_split(idx, 3 if montaje == "ei" else 2)):
            papel[p] = nombre
    return papel


def grupos(R, papel, montaje):
    """Índices que ajustan, que calibran y que se miden en un montaje con umbral conformal."""
    m = (R.montaje == montaje).values
    med = m & ((papel == "mide") if montaje == "ei" else (R.grupo == "medicion").values)
    return pos(m & (papel == "ajusta")), pos(m & (papel == "calibra")), pos(med)


def filas(prueba, R, idx, s, u, u2):
    """Una fila por pieza: puntaje crudo s, umbral u (α) y umbral u2 (α/2, el de la fusión)."""
    return [dict(prueba=prueba, id=R.id.values[i], giro=int(R.giro.values[i]), mala=bool(R.mala.values[i]),
                 s=float(v), u=float(u), u2=float(u2)) for i, v in zip(idx, s)]


def z_de(s, u):
    """z = s/u: se retira si z > 1. Una pieza sin medir (s = ∞) se retira siempre, aunque u = ∞."""
    s, u = np.asarray(s, float), np.asarray(u, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = s / u
    return np.where(np.isposinf(s), np.inf, z)


def umbrales(s):
    return umbral_conformal(s, ALFA), umbral_conformal(s, ALFA / 2)


# ---------------------------------------------------------------- clásica
def puntaje_cl(mod, X, ok):
    return np.where(ok, mod.puntaje(X), np.inf)  # sin medir: se retira siempre


def eval_clasica(R, cols, papel):
    X = R[cols].values.astype(float)
    ok = np.isfinite(X).all(1)
    L, U, ei = [], [], None
    re = pos((R.montaje == "propio") & (R.giro == 0))
    for i in re:  # reglas de e11: Hotelling, dejando una fuera
        mod = ajustar_maha(X[[j for j in re if j != i and not R.mala.values[j]]])
        L += filas("A", R, [i], puntaje_cl(mod, X[[i]], ok[[i]]), mod.t, mod.t)
        g = pos((R.montaje == "propio") & (R.giro > 0) & (R.id == R.id.values[i]))
        L += filas("R", R, g, puntaje_cl(mod, X[g], ok[g]), mod.t, mod.t)
    for montaje, prueba in CONFORMALES:
        aj, ca, med = grupos(R, papel, montaje)
        mod = ajustar_maha(X[aj])  # ajustar_maha deja fuera las que no se midieron
        u, u2 = umbrales(mod.puntaje(X[ca][ok[ca]]))
        L += filas(prueba, R, med, puntaje_cl(mod, X[med], ok[med]), u, u2)
        U.append(dict(montaje=montaje, n_ajusta=int(ok[aj].sum()), n_calibra=int(ok[ca].sum()), u=u, u2=u2,
                      umbral_e11=float(mod.t), sin_medir_ajusta=int((~ok[aj]).sum()),
                      sin_medir_calibra=int((~ok[ca]).sum()), sin_medir_mide=int((~ok[med]).sum())))
        if montaje == "ei":
            ei = (mod, u, u2)
    return pd.DataFrame(L), U, ei


# ---------------------------------------------------------------- PatchCore
def eval_patchcore(R, X, capas, papel):
    L, U, ei = [], [], None
    re = pos((R.montaje == "propio") & (R.giro == 0))
    for i in re:  # reglas de e11: el máximo del LOO interno, dejando una fuera
        pc = banco(X, [j for j in re if j != i and not R.mala.values[j]], capas)
        L += filas("A", R, [i], pc.puntaje(sel(X, [i])), pc.umbral, pc.umbral)
        g = pos((R.montaje == "propio") & (R.giro > 0) & (R.id == R.id.values[i]))
        L += filas("R", R, g, pc.puntaje(sel(X, g)), pc.umbral, pc.umbral)
    for montaje, prueba in CONFORMALES:
        aj, ca, med = grupos(R, papel, montaje)
        pc = banco(X, aj, capas)  # 50 buenas o más: sin giros
        u, u2 = umbrales(pc.puntaje(sel(X, ca)))
        L += filas(prueba, R, med, pc.puntaje(sel(X, med)), u, u2)
        U.append(dict(montaje=montaje, n_ajusta=len(aj), n_calibra=len(ca), u=u, u2=u2, umbral_e11=pc.umbral,
                      n_banco=pc.n_banco))
        if montaje == "ei":
            ei = (pc, u, u2)
        else:
            del pc
            torch.cuda.empty_cache()
    return pd.DataFrame(L), U, ei


# ---------------------------------------------------------------- criterio
def ic(k, n):
    """IC95 exacto (Clopper-Pearson) de una tasa k/n."""
    if not n:
        return np.nan, np.nan
    r = binomtest(int(k), int(n)).proportion_ci(0.95, method="exact")
    return float(r.low), float(r.high)


def resumen(L, ms):
    L = L.assign(z=z_de(L.s, L.u))
    r = {}
    for p in PRUEBAS:
        d = L[L.prueba == p]
        for clase, mala, n in (("malas", True, "n_malas"), ("falsas", False, "n_buenas")):
            k = int((d.z[d.mala == mala] > 1).sum())
            r[f"{p}_{clase}"], r[f"{p}_{n}"] = k, int((d.mala == mala).sum())
            r[f"{p}_{clase}_lo"], r[f"{p}_{clase}_hi"] = ic(k, r[f"{p}_{n}"])
        r[f"{p}_auc"] = auc(d.mala.values, d.z.values) if d.mala.any() and (~d.mala).any() else np.nan
    r["ms_cpu"] = ms
    r["det_media"] = (r["kaggle_malas"] / r["kaggle_n_malas"] + r["sint_malas"] / r["sint_n_malas"]) / 2
    r["falsas_media"] = float(np.mean([r[f"{p}_falsas"] / r[f"{p}_n_buenas"] for p in ("A", "kaggle", "ei_tercio")]))
    r["req1"] = bool(r["A_malas"] == r["A_n_malas"] and r["R_malas"] >= 0.95 * r["R_n_malas"] and ms < 100)
    r["req2"] = bool(r["A_falsas"] <= 1 and r["kaggle_falsas"] <= 0.05 * r["kaggle_n_buenas"]
                     and r["ei_tercio_falsas"] <= 0.05 * r["ei_tercio_n_buenas"])
    return r


def desempate(T, col, mayor):
    v = T[col]
    emp = T[v >= v.max() - 0.02] if mayor else T[v <= v.min() + 0.02]
    return emp.sort_values(["explica", "ms_cpu"]).index[0]


def elegir(T):
    """Devuelve (elegido, ¿ninguno cumple?, por qué vía)."""
    if (T.req1 & T.req2).any():
        return desempate(T[T.req1 & T.req2], "det_media", True), False, "cumple 1 y 2"
    if T.req2.any():
        return desempate(T[T.req2], "det_media", True), True, "respaldo: cumple 2, mayor detección media"
    return desempate(T, "falsas_media", False), True, "respaldo: ninguno cumple 2, menor tasa media de falsas"


def fusionar(a, b):
    """Retira si la clásica o PatchCore retiran, cada uno con su umbral de α/2 (en el montaje propio, el de e11)."""
    m = a.merge(b, on=CLAVE, suffixes=("_cl", "_pc"), validate="one_to_one")
    z = np.maximum(z_de(m.s_cl, m.u2_cl), z_de(m.s_pc, m.u2_pc))
    return m[CLAVE].assign(s=z, u=1.0, u2=1.0)


def beta_condicional(n, alfa):
    """Tasa de falsas alarmas condicionada a una calibración: Beta(ℓ, n+1−ℓ), ℓ = ⌊(n+1)α⌋ (Bates et al., Prop. 1)."""
    ell = int(np.floor((n + 1) * alfa + 1e-9))
    if ell < 1:
        return dict(n=n, ell=ell)
    b = beta(ell, n + 1 - ell)
    return dict(n=n, ell=ell, media=ell / (n + 1), p05=float(b.ppf(0.05)), p95=float(b.ppf(0.95)),
                p_sobre_5=float(b.sf(0.05)))


def control_e11(T):
    """En el montaje propio, los candidatos de e11 deben repetir sus números (mismo código y mismas reglas)."""
    f = RESULTADOS / f"e11_candidatos{SUF}.csv"
    if not f.exists():
        return "sin e11"
    E = pd.read_csv(f, index_col=0)
    cols = ["A_malas", "A_falsas", "R_malas", "R_falsas"]
    dif = {n: {c: [int(T.loc[n, c]), int(E.loc[n, c])] for c in cols if int(T.loc[n, c]) != int(E.loc[n, c])}
           for n in T.index if n in E.index and n != "fusion"}
    return {n: d for n, d in dif.items() if d} or "igual"


# ---------------------------------------------------------------- experimento
def calcular():
    t0 = time.time()
    R, C, _ = preparar("--rasgos" in sys.argv)
    X, k = C["X"], C["k"]
    papel = particion(R)
    R.assign(papel=papel).loc[papel != "", ["montaje", "grupo", "id", "papel"]].to_csv(
        RESULTADOS / f"e12_particion{SUF}.csv", index=False)
    largos, U, modelos_ei = {}, [], {}
    for n, cols in CLASICAS.items():
        largos[n], u, modelos_ei[n] = eval_clasica(R, cols, papel)
        U += [dict(candidato=n, **x) for x in u]
        print(n, f"{time.time() - t0:.0f} s", flush=True)
    for n, (capas, modo) in PATCHCORES.items():
        largos[n], u, modelos_ei[n] = eval_patchcore(R, X[modo], capas, papel)
        U += [dict(candidato=n, **x) for x in u]
        print(n, f"{time.time() - t0:.0f} s", flush=True)
    ms = latencias(R, X, k)
    img, _ = imagen(("real", 0, 0))
    ms["clasica_4"] = latencia(lambda: cadena(img, sector=False), reps=20)
    print("latencias", {a: round(b, 1) for a, b in ms.items()}, flush=True)

    T = pd.DataFrame({n: resumen(L, ms[n]) for n, L in largos.items()}).T
    T["familia"] = [familia(n) for n in T.index]
    T["explica"] = T.familia.map(EXPLICA)
    T = T.infer_objects()
    cl, _, via_cl = elegir(T[T.familia == "clasica"])
    pc, _, via_pc = elegir(T[T.familia == "patchcore"])
    largos["fusion"] = fusionar(largos[cl], largos[pc])
    ms["fusion"] = ms[cl] + ms[f"red_{pc}"]
    T.loc["fusion"] = pd.Series({**resumen(largos["fusion"], ms["fusion"]), "familia": "fusion",
                                 "explica": EXPLICA["fusion"]})
    T = T.infer_objects()
    T["componentes"] = ""
    T.loc["fusion", "componentes"] = f"{cl} + {pc}"
    elegido, ninguno, via = elegir(T)
    T["elegido"] = T.index == elegido
    T.to_csv(RESULTADOS / f"e12_candidatos{SUF}.csv")
    pd.concat([L.assign(candidato=n) for n, L in largos.items()]).to_csv(RESULTADOS / f"e12_puntajes{SUF}.csv",
                                                                         index=False)
    pd.DataFrame(U).to_csv(RESULTADOS / f"e12_umbrales{SUF}.csv", index=False)
    n_cal = {m: int(((papel == "calibra") & (R.montaje == m).values).sum()) for m, _ in CONFORMALES}
    info = dict(elegido=elegido, ninguno_cumple=ninguno, adoptado=not ninguno, via=via, alfa=ALFA,
                clasica_de_familia=cl, via_clasica=via_cl, patchcore_de_familia=pc, via_patchcore=via_pc,
                particion={m: {p: int(((papel == p) & (R.montaje == m).values).sum()) for p in ("ajusta", "calibra",
                                                                                                "mide")}
                           for m, _ in CONFORMALES},
                beta_condicional={m: dict(alfa=beta_condicional(n, ALFA), alfa_2=beta_condicional(n, ALFA / 2))
                                  for m, n in n_cal.items()},
                control_propio_e11=control_e11(T), k=k, ms=ms,
                dispositivo=torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
                minutos=round((time.time() - t0) / 60, 1))
    (RESULTADOS / f"e12_eleccion{SUF}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1,
                                                                    default=float))
    e = T.loc[elegido]
    if not HUMO:
        decidir("enfoque_v4", elegido,
                "ADR 0015: umbral conformal partido (α = 3 %; α/2 por parte en la fusión; propio con las reglas "
                "de e11); 6/6 en A, ≥95 % en R y <100 ms; falsas ≤1/15 propias, ≤5 % Kaggle y ≤5 % en el tercio "
                "de EI; mayor detección media (Kaggle + sintéticas); empate ≤2 puntos → más explicable, más "
                "rápida; respaldo: cumple 2 → mayor detección, si no → menor tasa media de falsas; respaldo = "
                "no se adopta",
                experimento="e12", ninguno_cumple=ninguno, adoptado=not ninguno, via=via,
                A=f"{e.A_malas}/{e.A_n_malas}", A_falsas=f"{e.A_falsas}/{e.A_n_buenas}",
                R=round(e.R_malas / e.R_n_malas, 3), kaggle=round(e.kaggle_malas / e.kaggle_n_malas, 3),
                kaggle_falsas=round(e.kaggle_falsas / e.kaggle_n_buenas, 3),
                ei_tercio_falsas=f"{e.ei_tercio_falsas}/{e.ei_tercio_n_buenas}",
                sint=round(e.sint_malas / e.sint_n_malas, 3), ms_cpu=round(float(e.ms_cpu), 1))
    cols = ["A_malas", "A_falsas", "R_malas", "sint_malas", "sint_falsas", "kaggle_malas", "kaggle_falsas",
            "ei_tercio_falsas", "det_media", "falsas_media", "ms_cpu", "req1", "req2", "elegido"]
    print(T[cols].round(3).to_string(), flush=True)
    print(f"elegido: {elegido} ({via}); control del montaje propio contra e11: {info['control_propio_e11']}",
          flush=True)

    # --- solo informativo, después de guardar la elección: Edge Impulse de prueba (NO CIEGO)
    med = pos((R.montaje == "ei") & (R.grupo == "prueba_no_ciega"))
    ei = {}
    for n, cols in CLASICAS.items():
        mod, u, u2 = modelos_ei[n]
        Xc = R[cols].values.astype(float)
        ei[n] = pd.DataFrame(filas("ei_prueba", R, med, puntaje_cl(mod, Xc[med], np.isfinite(Xc[med]).all(1)), u,
                                   u2))
    for n, (_capas, modo) in PATCHCORES.items():
        b, u, u2 = modelos_ei[n]
        ei[n] = pd.DataFrame(filas("ei_prueba", R, med, b.puntaje(sel(X[modo], med)), u, u2))
    ei["fusion"] = fusionar(ei[cl], ei[pc])
    F = []
    for n, L in ei.items():
        y, z = L.mala.values, z_de(L.s, L.u)
        lo, hi = auc_ic(y, z)
        km, kf = int((z[y] > 1).sum()), int((z[~y] > 1).sum())
        F.append(dict(candidato=n, malas=km, n_malas=int(y.sum()), malas_lo=ic(km, y.sum())[0],
                      malas_hi=ic(km, y.sum())[1], falsas=kf, n_buenas=int((~y).sum()), falsas_lo=ic(kf, (~y).sum())[0],
                      falsas_hi=ic(kf, (~y).sum())[1], auc=auc(y, z), auc_lo=lo, auc_hi=hi, no_ciego=True))
    E = pd.DataFrame(F).set_index("candidato")
    E.to_csv(RESULTADOS / f"e12_ei_no_ciego{SUF}.csv")
    print("Edge Impulse de prueba (NO CIEGO, solo informa):", E.round(3).to_string(), sep="\n", flush=True)
    return T


# ---------------------------------------------------------------- gráficos
def figuras(T, info):
    t = T.loc[ORDEN]
    elegido = info["elegido"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    barras_h(axs[0], ORDEN, [100 * t.A_malas / t.A_n_malas, 100 * t.kaggle_malas / t.kaggle_n_malas,
                             100 * t.sint_malas / t.sint_n_malas],
             ["propias, dejando una fuera (de 6)", "Kaggle (de 100)", "sintéticas de validación (de 300)"],
             colores=[AZUL, AQUA, TINTA2], elegido=elegido, fmt="{:.0f}", lim=(0, 118))
    axs[0].set_title("Malas detectadas (%)", fontsize=10)
    barras_h(axs[1], ORDEN, [100 * t.A_falsas / t.A_n_buenas, 100 * t.kaggle_falsas / t.kaggle_n_buenas,
                             100 * t.ei_tercio_falsas / t.ei_tercio_n_buenas],
             ["propias (de 15)", "Kaggle (de 100)", "Edge Impulse, tercio que mide (de 91)"],
             colores=[AZUL, AQUA, NARANJA], elegido=elegido, fmt="{:.0f}", lim=(0, 118))
    axs[1].axvline(5, color=TINTA2, lw=0.8, ls=":")
    axs[1].set_title("Buenas retiradas por error (%) · línea: tope del 5 %", fontsize=10)
    estrella = "★ = respaldo, ninguno cumple" if info["ninguno_cumple"] else "★ = elegido por el criterio"
    fig.suptitle(f"N1 v4 (e12): umbral conformal con α = {100 * info['alfa']:.0f} % (ADR 0015) · {estrella}",
                 x=0.01, ha="left", fontsize=12, y=1.02)
    guardar(fig, f"e12_n1_v4{SUF}.png")

    # Antes y después en Kaggle: el umbral de e11 contra el conformal, para los candidatos de e11
    e11 = pd.read_csv(RESULTADOS / f"e11_candidatos{SUF}.csv", index_col=0)
    nombres = [n for n in ORDEN if n in e11.index and n != "fusion"]
    a, b = e11.loc[nombres], T.loc[nombres]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    barras_h(axs[0], nombres, [100 * a.kaggle_malas / a.kaggle_n_malas, 100 * b.kaggle_malas / b.kaggle_n_malas],
             ["e11: Hotelling o el máximo de las buenas", f"e12: conformal, α = {100 * ALFA:.0f} %"],
             colores=[GRIS, AZUL], fmt="{:.0f}", lim=(0, 118))
    axs[0].set_title("Kaggle: malas detectadas (%, de 100)", fontsize=10)
    barras_h(axs[1], nombres, [100 * a.kaggle_falsas / a.kaggle_n_buenas,
                               100 * b.kaggle_falsas / b.kaggle_n_buenas],
             ["e11", "e12"], colores=[GRIS, AZUL], fmt="{:.0f}", lim=(0, 118), leyenda=False)
    axs[1].axvline(5, color=TINTA2, lw=0.8, ls=":")
    axs[1].set_title("Kaggle: buenas retiradas por error (%, de 100) · línea: tope del 5 %", fontsize=10)
    fig.suptitle("Mismo modelo, otro umbral: qué cambia en Kaggle de e11 a e12", x=0.01, ha="left", fontsize=12,
                 y=1.02)
    guardar(fig, f"e12_umbral_kaggle{SUF}.png")


def main():
    estilo()
    csv = RESULTADOS / f"e12_candidatos{SUF}.csv"
    if HUMO or not csv.exists() or "--recalcular" in sys.argv:
        T = calcular()
    else:
        T = pd.read_csv(csv, index_col=0)
    figuras(T, json.loads((RESULTADOS / f"e12_eleccion{SUF}.json").read_text()))


if __name__ == "__main__":
    main()
