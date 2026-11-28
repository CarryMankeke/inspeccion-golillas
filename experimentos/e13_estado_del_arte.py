"""N0 v2 y N1 v5 · e13: estado del arte, visión clásica contra IA, en dos golillas de MANTA (ADR 0017).

Protocolo (ADR 0017, fijado antes de bajar los sets de prueba), por cada tipo que decide:
1. Perfil del tipo con las buenas que ajustan (`--perfil`, en un commit antes de la corrida):
   vista de arriba, polaridad y escala para la clásica, y si la clásica aplica (mide ≥ 90 %).
2. Partición sellada (data/sellos/): las buenas de entrenamiento, mitad ajusta y mitad calibra.
3. Cada método se ajusta con las que ajustan, con su configuración por defecto.
4. Puntaje de una pieza: el máximo de sus 5 vistas en la IA; la vista de arriba en la clásica
   (sin medir = ∞: se retira siempre y no calibra).
5. Umbral conformal partido con α = 3 % sobre las que calibran; se retira si el puntaje lo supera.
6. La prueba se corre una sola vez.

Criterio: en cada tipo que decide, no detección ≤ 2 % y falsas ≤ 5 % (clase «aceptable» del
AIAG), y ≤ 100 ms por pieza en la CPU del servidor (mediana de 50 piezas que ajustan, 4 hilos,
con las 5 vistas). Entre los que cumplen, gana la mayor detección media; empate (≤ 2 puntos):
clásica, sin entrenamiento, con entrenamiento, texto; después, el más rápido. Si ninguno cumple,
«ninguno cumple», no se adopta nada, y la tabla se publica igual con la clase AIAG de cada uno.
Guarda la decisión «enfoque_v5».

Uso (en norea-node, con el entorno de anomalib; mirar `nvidia-smi` antes):
    python experimentos/e13_estado_del_arte.py --perfil     # paso 1 (solo las que ajustan)
    python experimentos/e13_estado_del_arte.py --control    # el PatchCore propio repite e12 en Kaggle
    python experimentos/e13_estado_del_arte.py --humo       # prueba de humo, SOLO con sets de desarrollo
    python experimentos/e13_estado_del_arte.py              # la corrida única que decide
    python experimentos/e13_estado_del_arte.py --informa    # después: Roboflow, Kaggle y Edge Impulse
    python experimentos/e13_estado_del_arte.py --pocas      # después: la curva de pocas muestras
    python experimentos/e13_estado_del_arte.py --figuras    # rehace los gráficos con lo guardado
`--humo` usa Kaggle y Edge Impulse como si decidieran, con 40 piezas por grupo y clase, 20 pasos de
Dinomaly y 5 piezas de latencia; escribe con el sufijo _humo y no guarda la decisión.
"""

import gc
import json
import statistics
import sys
import time

import numpy as np
import pandas as pd
import torch
from comun import AZUL, NARANJA, RESULTADOS, TINTA2, barras_h, decidir, estilo, guardar
from e09_ajuste_redes import auc, auc_ic
from matplotlib import pyplot as plt

from golillas import conjuntos
from golillas.datos import RAIZ
from golillas.decision import clase_aiag, ic_exacto, umbral_conformal
from golillas.metodos import EXPLICA, FAMILIA, METODOS, crear
from golillas.metodos.clasica import leer_perfil, perfil

HUMO = "--humo" in sys.argv
SUF = "_humo" if HUMO else ""
ALFA = 0.03
MS_MAX = 100.0
DECIDEN = ("kaggle", "edge_impulse") if HUMO else ("manta_red_washer", "manta_yellow_green_washer")
INFORMAN = ("roboflow_saksit_metal_washer", "roboflow_renuka_washer_defect", "roboflow_washer_classifier", "kaggle",
            "edge_impulse")
NOMBRE = {"manta_red_washer": "MANTA roja", "manta_yellow_green_washer": "MANTA amarillo-verde",
          "kaggle": "Kaggle", "edge_impulse": "Edge Impulse"}
N_HUMO = 40 if HUMO else None
N_LATENCIA = 5 if HUMO else 50
OPCIONES = {"dinomaly": dict(pasos=20)} if HUMO else {}
POCAS_N, POCAS_SEMILLAS = (1, 4, 16, 64), range(5)
SIN_ENTRENAMIENTO = [m for m in METODOS if FAMILIA[m] == "sin_entrenamiento"]


def log(*a):
    print(*a, flush=True)


def cargar(conjunto, grupo):
    return conjuntos.cargar(conjunto, grupo, n=N_HUMO)


# ---------------------------------------------------------------- medidas
def mediana_ms(f, piezas):
    for p in piezas[:3]:
        f(p)
    t = []
    for p in piezas:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        f(p)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1000)
    return statistics.median(t)


def latencias(met, nombre, piezas):
    """(ms por pieza en CPU con 4 hilos, ms en la GPU): medianas con las piezas que ajustan."""
    gpu = None if FAMILIA[nombre] == "clasica" or not torch.cuda.is_available() else mediana_ms(met.una_pieza, piezas)
    hilos = torch.get_num_threads()
    torch.set_num_threads(4)
    cpu = met.en_cpu() if hasattr(met, "en_cpu") else met
    ms = mediana_ms(cpu.una_pieza, piezas)
    torch.set_num_threads(hilos)
    del cpu
    gc.collect()
    return ms, gpu


def metricas(y, s, u):
    retira = s > u
    tp, fn = int((retira & y).sum()), int((~retira & y).sum())
    fp, tn = int((retira & ~y).sum()), int((~retira & ~y).sum())
    nm, nb = tp + fn, fp + tn
    lo_nd, hi_nd = ic_exacto(fn, nm)
    lo_fa, hi_fa = ic_exacto(fp, nb)
    a_lo, a_hi = auc_ic(y, s)
    prec = tp / (tp + fp) if tp + fp else np.nan
    sens = tp / nm
    return dict(malas=tp, n_malas=nm, falsas=fp, n_buenas=nb, no_deteccion=fn / nm, no_deteccion_lo=lo_nd,
                no_deteccion_hi=hi_nd, falsa_alarma=fp / nb, falsa_alarma_lo=lo_fa, falsa_alarma_hi=hi_fa,
                sensibilidad=sens, especificidad=tn / nb, precision=prec,
                f1=2 * prec * sens / (prec + sens) if tp else 0.0, efectividad=(tp + tn) / (nm + nb),
                clase_aiag=clase_aiag(fn, nm, fp, nb), auc=auc(y, s), auc_lo=float(a_lo), auc_hi=float(a_hi))


# ---------------------------------------------------------------- un conjunto
def evaluar(conjunto, per, decide, metodos=METODOS):
    """Ajusta, calibra y mide cada método en un conjunto.

    Devuelve las filas de resumen, por pieza, por defecto y por vista.
    """
    t0 = time.time()
    aj, ca = cargar(conjunto, "ajusta"), cargar(conjunto, "calibra")
    pr = cargar(conjunto, "prueba")
    log(f"== {conjunto}: {len(aj)} ajustan, {len(ca)} calibran, prueba {sum(not p.mala for p in pr)} buenas y "
        f"{sum(p.mala for p in pr)} malas ({time.time() - t0:.0f} s de carga)")
    y = np.array([p.mala for p in pr])
    T, P, D, V = [], [], [], []
    for nombre in metodos:
        fam = FAMILIA[nombre]
        base = dict(conjunto=conjunto, metodo=nombre, familia=fam, explica=EXPLICA[fam], decide=decide)
        if fam == "clasica" and not per["clasica_aplica"]:
            T.append({**base, "aplica": False})
            log(f"  {nombre}: no aplica (mide {per['mide']:.0%} de las que ajustan)")
            continue
        t1 = time.time()
        try:
            met = crear(nombre, per, **OPCIONES.get(nombre, {}))
            met.ajustar(aj)
            u = umbral_conformal(met.puntajes_vistas(ca).max(1), ALFA)
            S = met.puntajes_vistas(pr)
            ms_cpu, ms_gpu = latencias(met, nombre, aj[:N_LATENCIA]) if decide else (np.nan, np.nan)
        except Exception as e:  # noqa: BLE001 — un método que falla se informa y no cumple
            T.append({**base, "aplica": True, "error": repr(e)[:400]})
            log(f"  {nombre}: ERROR {e!r}")
            met = None
            gc.collect()
            torch.cuda.empty_cache()
            continue
        s = S.max(1)
        info = dict(getattr(met, "info", {}))
        if fam == "clasica":
            info["sin_medir_ajusta"] = met.sin_medir
        T.append({**base, "aplica": True, "umbral": u, **metricas(y, s, u), "ms_cpu": ms_cpu, "ms_gpu": ms_gpu,
                  "piezas_min_cpu": 60_000 / ms_cpu if ms_cpu else np.nan, "s_total": time.time() - t1,
                  "info": json.dumps(info, default=float)})
        for p, si, fila in zip(pr, s, S):
            P.append(dict(conjunto=conjunto, metodo=nombre, pieza=p.id, mala=p.mala, defecto=p.defecto, s=float(si),
                          umbral=u, retira=bool(si > u), **{f"s_v{k}": float(v) for k, v in enumerate(fila)}))
        for d in sorted({p.defecto for p in pr if p.mala}):
            m = np.array([p.defecto == d for p in pr])
            D.append(dict(conjunto=conjunto, metodo=nombre, defecto=d, n=int(m.sum()),
                          detectadas=int((s[m] > u).sum())))
        if S.shape[1] > 1:
            V += [dict(conjunto=conjunto, metodo=nombre, vista=k, auc=auc(y, S[:, k])) for k in range(S.shape[1])]
        r = T[-1]
        log(f"  {nombre}: malas {r['malas']}/{r['n_malas']}, falsas {r['falsas']}/{r['n_buenas']} "
            f"({r['clase_aiag']}), AUC {r['auc']:.3f}, {ms_cpu:.0f} ms CPU, {time.time() - t1:.0f} s")
        del met
        gc.collect()
        torch.cuda.empty_cache()
    return T, P, D, V


# ---------------------------------------------------------------- criterio
def elegir(T):
    """Devuelve (tabla por método, elegido o None)."""
    filas = []
    for m, g in T[T.conjunto.isin(DECIDEN)].groupby("metodo", sort=False):
        ok = [bool(r.aplica and pd.isna(getattr(r, "error", np.nan)) and r.clase_aiag == "aceptable"
                   and r.ms_cpu <= MS_MAX)
              for r in g.itertuples()]
        filas.append(dict(metodo=m, familia=FAMILIA[m], explica=EXPLICA[FAMILIA[m]],
                          cumple=bool(len(g) == len(DECIDEN) and all(ok)),
                          det_media=float(g.sensibilidad.mean()), ms_media=float(g.ms_cpu.mean())))
    M = pd.DataFrame(filas).set_index("metodo")
    c = M[M.cumple]
    if c.empty:
        return M, None
    emp = c[c.det_media >= c.det_media.max() - 0.02]
    return M, emp.sort_values(["explica", "ms_media"]).index[0]


def guardar_tablas(nombre, T, P, D, V):
    for sufijo, filas in (("", T), ("_puntajes", P), ("_por_defecto", D), ("_por_vista", V)):
        pd.DataFrame(filas).to_csv(RESULTADOS / f"e13_{nombre}{sufijo}{SUF}.csv", index=False)


def corrida():
    import anomalib

    t0 = time.time()
    log(f"e13 {'HUMO' if HUMO else ''} · torch {torch.__version__} · anomalib {anomalib.__version__} · "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    T, P, D, V = [], [], [], []
    for c in DECIDEN:
        per = perfil(c, cargar(c, "ajusta")) if HUMO else leer_perfil(c)
        t, p, d, v = evaluar(c, per, decide=True)
        T, P, D, V = T + t, P + p, D + d, V + v
        guardar_tablas("metodos", T, P, D, V)  # se guarda tras cada tipo por si la corrida se corta
    T = pd.DataFrame(T)
    M, elegido = elegir(T)
    M.to_csv(RESULTADOS / f"e13_eleccion{SUF}.csv")
    info = dict(elegido=elegido, ninguno_cumple=elegido is None, adoptado=elegido is not None, alfa=ALFA,
                deciden=list(DECIDEN), ms_max=MS_MAX, minutos=round((time.time() - t0) / 60, 1),
                dispositivo=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    (RESULTADOS / f"e13_eleccion{SUF}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    log(M.round(3).to_string())
    log(f"elegido: {elegido or 'ninguno cumple'} · {info['minutos']} min")
    if not HUMO:
        detalle = {f"{NOMBRE[r.conjunto]}·{r.metodo}": (f"{r.malas}/{r.n_malas} malas, {r.falsas}/{r.n_buenas} falsas, "
                                                         f"{r.clase_aiag}, {r.ms_cpu:.0f} ms")
                   for r in T.itertuples() if r.aplica and pd.isna(getattr(r, "error", np.nan))}
        decidir("enfoque_v5", elegido or "ninguno",
                "ADR 0017: en cada golilla de MANTA, no detección ≤ 2 % y falsas ≤ 5 % (AIAG aceptable) con umbral "
                "conformal α = 3 %, y ≤ 100 ms por pieza en la CPU del servidor; gana la mayor detección media; "
                "empate ≤ 2 puntos → clásica, sin entrenamiento, con entrenamiento, texto; después, la más rápida",
                experimento="e13", adoptado=elegido is not None, ninguno_cumple=elegido is None, detalle=detalle)
    return T


# ---------------------------------------------------------------- después de decidir
def informa():
    T, P, D, V = [], [], [], []
    for c in INFORMAN:
        t, p, d, v = evaluar(c, leer_perfil(c), decide=False)
        T, P, D, V = T + t, P + p, D + d, V + v
        guardar_tablas("informa", T, P, D, V)


def pocas():
    """AUC con 1, 4, 16 y 64 piezas que ajustan (5 semillas), y WinCLIP sin ninguna."""
    F = []
    for c in DECIDEN:
        aj, pr = cargar(c, "ajusta"), cargar(c, "prueba")
        y = np.array([p.mala for p in pr])
        met = crear("winclip").ajustar(aj[:1])
        F.append(dict(conjunto=c, metodo="winclip", n=0, semilla=0, auc=auc(y, met.puntajes_vistas(pr).max(1))))
        for m in SIN_ENTRENAMIENTO:
            for n in POCAS_N:
                for sem in POCAS_SEMILLAS:
                    met = crear(m, leer_perfil(c)).ajustar(conjuntos.submuestra(aj, n, sem))
                    F.append(dict(conjunto=c, metodo=m, n=n, semilla=sem, auc=auc(y, met.puntajes_vistas(pr).max(1))))
                    del met
                    torch.cuda.empty_cache()
                log(c, m, n, np.mean([f["auc"] for f in F if f["metodo"] == m and f["n"] == n and f["conjunto"] == c]))
            pd.DataFrame(F).to_csv(RESULTADOS / "e13_pocas.csv", index=False)


# ---------------------------------------------------------------- control contra e12
def control():
    """Con la configuración de e12 (recortes de e11, partición de e12), el PatchCore propio repite e12 en Kaggle."""
    from golillas.metodos.patchcore_propio import CAPAS, PatchCore

    R = pd.read_csv(RESULTADOS / "e11_rasgos.csv")
    X = torch.load(RAIZ / "cache" / "e11_recortes.pt", weights_only=False)["X"]["fijo"]
    P = pd.read_csv(RESULTADOS / "e12_particion.csv")
    papel = dict(zip(P[P.montaje == "kaggle"].id, P[P.montaje == "kaggle"].papel))
    kg = (R.montaje == "kaggle").values
    cal = kg & (R.grupo == "calibracion").values
    idx = {q: np.where(cal & (R.id.map(papel) == q).values)[0] for q in ("ajusta", "calibra")}
    med = np.where(kg & (R.grupo == "medicion").values)[0]

    def sel(i):
        return X[torch.as_tensor(i, dtype=torch.long)]

    pc = PatchCore(CAPAS["l2+l3"], 0.01).ajustar(sel(idx["ajusta"]), giros=1)
    u = umbral_conformal(pc.puntaje(sel(idx["calibra"])), ALFA)
    s, y = pc.puntaje(sel(med)), R.mala.values[med].astype(bool)
    E = pd.read_csv(RESULTADOS / "e12_candidatos.csv", index_col=0).loc["patchcore_l2+l3_fijo"]
    r = dict(e13=dict(malas=int((s[y] > u).sum()), falsas=int((s[~y] > u).sum())),
             e12=dict(malas=int(E.kaggle_malas), falsas=int(E.kaggle_falsas)),
             n=dict(ajusta=len(idx["ajusta"]), calibra=len(idx["calibra"]), mide=len(med)))
    r["igual"] = r["e13"] == r["e12"]
    (RESULTADOS / "e13_control.json").write_text(json.dumps(r, indent=1))
    log("control contra e12 (PatchCore propio l2+l3 fijo, Kaggle):", r)
    if not r["igual"]:
        sys.exit("el control NO repite e12: revisar antes de seguir")


# ---------------------------------------------------------------- gráficos
def figuras():
    T = pd.read_csv(RESULTADOS / f"e13_metodos{SUF}.csv")
    info = json.loads((RESULTADOS / f"e13_eleccion{SUF}.json").read_text())
    T = T[T.aplica & T.get("error", pd.Series(np.nan, index=T.index)).isna()]
    orden = [m for m in METODOS if m in set(T.metodo)]
    tab = {c: T[T.conjunto == c].set_index("metodo").reindex(orden) for c in DECIDEN}
    colores = [AZUL, NARANJA]
    fig, axs = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    barras_h(axs[0], orden, [100 * tab[c].sensibilidad for c in DECIDEN], [NOMBRE[c] for c in DECIDEN],
             colores=colores, elegido=info["elegido"], fmt="{:.0f}", lim=(0, 118))
    axs[0].axvline(98, color=TINTA2, lw=0.8, ls=":")
    axs[0].set_title("Malas detectadas (%) · línea: 98 % (AIAG)", fontsize=10)
    barras_h(axs[1], orden, [100 * tab[c].falsa_alarma for c in DECIDEN], [NOMBRE[c] for c in DECIDEN],
             colores=colores, elegido=info["elegido"], fmt="{:.1f}", leyenda=False)
    axs[1].axvline(5, color=TINTA2, lw=0.8, ls=":")
    axs[1].set_title("Buenas retiradas por error (%) · línea: 5 % (AIAG)", fontsize=10)
    barras_h(axs[2], orden, [tab[c].ms_cpu for c in DECIDEN], [NOMBRE[c] for c in DECIDEN], colores=colores,
             elegido=info["elegido"], fmt="{:.0f}", leyenda=False)
    axs[2].set_xscale("log")
    axs[2].axvline(MS_MAX, color=TINTA2, lw=0.8, ls=":")
    axs[2].set_title("ms por pieza en la CPU (5 vistas) · línea: 100 ms", fontsize=10)
    estrella = "★ = elegido" if info["elegido"] else "ninguno cumple: no se adopta nada"
    fig.suptitle(f"N1 v5 (e13): clásica contra IA, umbral conformal α = 3 % (ADR 0017) · {estrella}", x=0.01,
                 ha="left", fontsize=12, y=1.02)
    guardar(fig, f"e13_estado_del_arte{SUF}.png")


def main():
    estilo()
    if "--perfil" in sys.argv:
        for c in (*DECIDEN, *INFORMAN):
            p = perfil(c, conjuntos.cargar(c, "ajusta"), guardar=True)
            log(c, {k: p[k] for k in ("vista_arriba", "invertir", "escala", "mide", "clasica_aplica")})
    elif "--control" in sys.argv:
        control()
    elif "--informa" in sys.argv:
        informa()
    elif "--pocas" in sys.argv:
        pocas()
    elif "--figuras" in sys.argv:
        figuras()
    else:
        corrida()
        figuras()


if __name__ == "__main__":
    main()
