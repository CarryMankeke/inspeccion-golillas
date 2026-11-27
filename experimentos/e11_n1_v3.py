"""N1 v3 · e11: elegir el enfoque con varios montajes (ADR 0014).

Candidatos, con configuraciones fijas y sin búsqueda:
- clasica_v3_6: los 4 rasgos de e05 + razón de radios + radio exterior; Mahalanobis + Hotelling.
- clasica_v3_7: lo mismo + relleno mínimo por sector (cortes radiales finos).
- patchcore_<capas>_<recorte>: ResNet-18 de ImageNet, coreset 1 %, umbral por LOO interno.
  Capas layer2+3 o layer1+2+3. Recorte de 160 px fijo (el de e08) o proporcional: lado =
  k · r_eq, con r_eq = sqrt(área del relleno exterior / π) de la segmentación y k tal que la
  mediana de las 15 buenas propias dé 160 px.
- fusion: retira si la clásica elegida o el PatchCore elegido retiran. Su puntaje es el
  máximo de los dos puntajes divididos por su umbral.

Montajes que eligen. Todas las fotos pasan por la regla de polaridad P1 de e10, que no usa
etiquetas:
- propio: 21 fotos (15 buenas, 6 malas). Prueba A dejando una fuera y prueba R con sus 7
  giros, calibrando con las buenas de las otras 20.
- sintético: calibra con las 300 buenas de entrenamiento (semilla 5) y se mide en las 600
  de validación de e09 (semilla 9: 300 buenas y 300 malas).
- kaggle: calibra con las 1.000 buenas de entrenamiento y se mide en las 200 de prueba.
- ei_buenas: las 273 buenas de entrenamiento de Edge Impulse, dejando una fuera; solo mide
  falsas alarmas. En PatchCore, la buena i se compara con el umbral de las otras (el máximo
  de los puntajes LOO internos sin ella): por construcción da ≈ 1/273, y se informa igual.
Bancos de PatchCore: con menos de 50 buenas se agregan las 8 simetrías (como e09); con 50 o
más, sin giros (como e10). Las fotos que la cadena clásica no logra medir no calibran, y en
la medición se retiran.

Criterio (ADR 0014, fijado antes de correr):
1. Propio: 6/6 en A, ≥ 95 % de malas en R y < 100 ms por pieza en CPU.
2. Falsas: ≤ 1/15 propias (A), ≤ 5 % en las 100 buenas de prueba de Kaggle y ≤ 5 % en ei_buenas.
3. Entre los que cumplen, gana la mayor detección media: el promedio de la detección en las
   malas de Kaggle y en las sintéticas de validación.
4. Empate (≤ 2 puntos): clásica, luego fusión, luego PatchCore; después, la más rápida en CPU.
Si ninguno cumple 1 y 2, gana el de mayor detección media entre los que cumplen la latencia.
Primero se elige dentro de cada familia (clásica y PatchCore) con la misma regla; con esas
dos se arma la fusión, y al final se elige entre los 7 candidatos.

Solo para informar, después de guardar la elección: el set de prueba de Edge Impulse (191
fotos, NO CIEGO porque ya se vio en e10), calibrando con sus 273 buenas. Sus rasgos y
recortes se calculan al principio (los procesos paralelos deben correr antes de usar la
GPU), pero no se leen hasta después de guardar la elección.
Corre en norea-node (GPU; mirar `nvidia-smi` antes). Escribe la decisión "enfoque_v3".
`--humo` corre todo con muestras chicas para probar el código: no guarda la decisión y
escribe sus archivos con el sufijo _humo.
"""

import json
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from comun import AQUA, AZUL, NARANJA, RESULTADOS, TINTA2, barras_h, decidir, decision, estilo, guardar, mapa
from e05_rasgos import LISTA, girar
from e09_ajuste_redes import CAPAS, DEV, PatchCore, auc, auc_ic, latencia, u8
from e10_prueba_ciega import cargar as cargar_ei
from e10_prueba_ciega import golilla_clara, montar
from matplotlib import pyplot as plt

from golillas import bordes, medicion, preproceso, rasgos, segmentacion, sintetico
from golillas import decision as dec
from golillas.datos import RAIZ, cargar_reales

KG = RAIZ / "data" / "publico" / "kaggle_simplified_washers" / "simplified-washers-for-anomaly-detection"
HUMO = "--humo" in sys.argv
SUF = "_humo" if HUMO else ""
CACHE = RAIZ / "cache" / f"e11_recortes{SUF}.pt"
F4 = decision("rasgos").split(",")
CLASICAS = {"clasica_v3_6": F4 + ["razon_radios", "radio_ext"],
            "clasica_v3_7": F4 + ["razon_radios", "radio_ext", "relleno_sector_min"]}
PATCHCORES = {f"patchcore_{c}_{r}": (c, r) for c in ("l2+l3", "l1+l2+l3") for r in ("fijo", "prop")}
CORESET, LADO_FIJO, POCAS = 0.01, 160, 50
SEMILLA_ENT, SEMILLA_VAL = 5, 9
EXPLICA = {"clasica": 0, "fusion": 1, "patchcore": 2}
MODOS = ("fijo", "prop")
_REALES = None


# ---------------------------------------------------------------- imágenes y cadena clásica
def reales():
    global _REALES
    if _REALES is None:
        _REALES = cargar_reales()
    return _REALES


def imagen(fuente):
    """Devuelve (foto montada con la regla P1, ¿se invirtió?)."""
    tipo = fuente[0]
    if tipo == "real":
        img = reales()[fuente[1]].imagen()
        img = girar(img, fuente[2]) if fuente[2] else img
    elif tipo == "sint":
        img = sintetico.generar(fuente[1] * 100_000 + fuente[2], defecto=LISTA[fuente[2]]).imagen
    else:
        img = cv2.imread(fuente[1])
    return montar(img, "P1"), golilla_clara(img)


def centro(m, forma):
    """Centroide de la segmentación (como e08) y radio equivalente del relleno exterior."""
    mm = cv2.moments(m, binaryImage=True)
    if not mm["m00"]:
        return forma[1] / 2, forma[0] / 2, np.nan
    relleno, _ = segmentacion.partes(m)
    return mm["m10"] / mm["m00"], mm["m01"] / mm["m00"], float(np.sqrt(relleno.sum() / np.pi))


def segmentar(img):
    g = preproceso.preprocesar(img, decision("preproceso"))
    return g, segmentacion.segmentar(g, decision("segmentacion"))


def cadena(img, sector=True):
    """La cadena clásica completa (P1–P5) más el rasgo por sector y el centro para el recorte."""
    g, m = segmentar(img)
    pe, pi = bordes.bordes(g, m, decision("bordes"))
    ce, ci = medicion.medir(g, m, pe, pi, decision("medicion"))
    r = rasgos.calcular(g, m, pe, pi, ce, ci)
    r["relleno_sector_min"] = rasgos.relleno_sector_min(m, ce, ci) if sector else np.nan
    cx, cy, req = centro(m, img.shape)
    return r, cx, cy, req


def recortar(img, cx, cy, lado):
    """Recorte cuadrado de `lado` px alrededor del centro, llevado a 224 px (con lado 160 = e08)."""
    r = max(8, int(round(lado / 2)))
    p = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT)
    c = p[int(cy) : int(cy) + 2 * r, int(cx) : int(cx) + 2 * r]
    return cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA)[..., ::-1].copy()  # RGB


def trabajo_rasgos(fuente):
    img, inv = imagen(fuente)
    r, cx, cy, req = cadena(img)
    return dict(**r, cx=cx, cy=cy, r_eq=req, invertida=inv)


def trabajo_recortes(arg):
    fuente, cx, cy, lado = arg
    img, _ = imagen(fuente)
    return recortar(img, cx, cy, LADO_FIJO), recortar(img, cx, cy, lado)


# ---------------------------------------------------------------- datos
def especificaciones():
    S = []
    for i, m in enumerate(reales()):
        for k in range(8):
            S.append(dict(montaje="propio", grupo="real" if k == 0 else "giro", id=m.id, giro=k, mala=m.es_mala,
                          fuente=("real", i, k)))
    for i in range(300):  # LISTA[:300] son buenas
        S.append(dict(montaje="sintetico", grupo="calibracion", id=f"e{i:03d}", giro=0, mala=False,
                      fuente=("sint", SEMILLA_ENT, i)))
    for i, d in enumerate(LISTA):
        S.append(dict(montaje="sintetico", grupo="medicion", id=f"v{i:03d}", giro=0, mala=d != "ninguno",
                      fuente=("sint", SEMILLA_VAL, i)))
    for p in sorted((KG / "train" / "conform").glob("*.png")):
        S.append(dict(montaje="kaggle", grupo="calibracion", id=p.stem, giro=0, mala=False, fuente=("ruta", str(p))))
    for clase, mala in (("conform", False), ("anomaly", True)):
        for p in sorted((KG / "test" / clase).glob("*.png")):
            S.append(dict(montaje="kaggle", grupo="medicion", id=p.stem, giro=0, mala=mala, fuente=("ruta", str(p))))
    ei = cargar_ei()
    for r in ei.itertuples():
        grupo = "calibracion" if r.split == "calibracion" else "prueba_no_ciega"
        S.append(dict(montaje="ei", grupo=grupo, id=r.nombre, giro=0, mala=bool(r.mala), fuente=("ruta", r.ruta)))
    assert not any(s["mala"] for s in S if s["montaje"] == "ei" and s["grupo"] == "calibracion")
    if HUMO:  # muestras chicas por (montaje, grupo, clase); las propias van completas
        cuenta = {}

        def deja(s):
            clave = (s["montaje"], s["grupo"], s["mala"])
            cuenta[clave] = cuenta.get(clave, 0) + 1
            return s["montaje"] == "propio" or cuenta[clave] <= 60

        S = [s for s in S if deja(s)]
    return S


def preparar(recalcular):
    """Rasgos de todas las fotos (CSV) y sus dos recortes (caché). Todo en CPU, antes de la GPU."""
    S = especificaciones()
    csv = RESULTADOS / f"e11_rasgos{SUF}.csv"
    if csv.exists() and CACHE.exists() and not recalcular:
        R = pd.read_csv(csv)
        return R, torch.load(CACHE, weights_only=False), S
    t0 = time.time()
    R = pd.DataFrame([{k: v for k, v in s.items() if k != "fuente"} for s in S])
    R = pd.concat([R, pd.DataFrame(mapa(trabajo_rasgos, [s["fuente"] for s in S]))], axis=1)
    print(f"rasgos de {len(R)} fotos en {time.time() - t0:.0f} s", flush=True)
    propias = R[(R.montaje == "propio") & (R.giro == 0) & ~R.mala]
    k = LADO_FIJO / float(np.median(propias.r_eq))
    lados = np.where(np.isfinite(R.r_eq), k * R.r_eq, LADO_FIJO)
    X = {m: [] for m in MODOS}
    for a in range(0, len(S), 600):  # por tandas, para no juntar todo en memoria dos veces
        rec = mapa(trabajo_recortes, [(s["fuente"], R.cx[i], R.cy[i], lados[i]) for i, s in
                                      enumerate(S[a : a + 600], start=a)])
        for j, m in enumerate(MODOS):
            X[m].append(u8([r[j] for r in rec]))
    X = {m: torch.cat(v) for m, v in X.items()}
    R["lado_prop"] = lados
    R.to_csv(csv, index=False)
    CACHE.parent.mkdir(exist_ok=True)
    torch.save(dict(X=X, k=k), CACHE)
    print(f"recortes listos en {time.time() - t0:.0f} s (k = {k:.3f})", flush=True)
    return R, dict(X=X, k=k), S


def pos(mask):
    return np.where(np.asarray(mask))[0]


def sel(X, idx):
    return X[torch.as_tensor(np.asarray(idx), dtype=torch.long)]


def filas_z(prueba, R, idx, z):
    return [dict(prueba=prueba, id=R.id.values[i], giro=int(R.giro.values[i]), mala=bool(R.mala.values[i]),
                 z=float(v)) for i, v in zip(idx, z)]


# ---------------------------------------------------------------- clásica
def ajustar_maha(X):
    X = np.asarray(X, float)
    ok = np.isfinite(X).all(1)
    return dec.modelos()[decision("decision")].fit(X[ok])


def z_maha(mod, X):
    return mod.puntaje(np.asarray(X, float)) / mod.t


def eval_clasica(R, cols):
    filas = []
    re = pos((R.montaje == "propio") & (R.giro == 0))
    for i in re:
        cal = [j for j in re if j != i and not R.mala.values[j]]
        mod = ajustar_maha(R[cols].values[cal])
        filas += filas_z("A", R, [i], z_maha(mod, R[cols].values[[i]]))
        g = pos((R.montaje == "propio") & (R.giro > 0) & (R.id == R.id.values[i]))
        filas += filas_z("R", R, g, z_maha(mod, R[cols].values[g]))
    for montaje, prueba in (("sintetico", "sint"), ("kaggle", "kaggle")):
        cal = pos((R.montaje == montaje) & (R.grupo == "calibracion"))
        med = pos((R.montaje == montaje) & (R.grupo == "medicion"))
        filas += filas_z(prueba, R, med, z_maha(ajustar_maha(R[cols].values[cal]), R[cols].values[med]))
    ei = pos((R.montaje == "ei") & (R.grupo == "calibracion"))
    X = R[cols].values[ei]
    z = [z_maha(ajustar_maha(np.delete(X, i, 0)), X[i : i + 1])[0] for i in range(len(ei))]
    filas += filas_z("ei_loo", R, ei, z)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- PatchCore
def banco(X, idx, capas):
    return PatchCore(CAPAS[capas], CORESET).ajustar(sel(X, idx), giros=8 if len(idx) < POCAS else 1)


def eval_patchcore(R, X, capas):
    filas = []
    re = pos((R.montaje == "propio") & (R.giro == 0))
    for i in re:
        pc = banco(X, [j for j in re if j != i and not R.mala.values[j]], capas)
        filas += filas_z("A", R, [i], pc.puntaje(sel(X, [i])) / pc.umbral)
        g = pos((R.montaje == "propio") & (R.giro > 0) & (R.id == R.id.values[i]))
        filas += filas_z("R", R, g, pc.puntaje(sel(X, g)) / pc.umbral)
    for montaje, prueba in (("sintetico", "sint"), ("kaggle", "kaggle")):
        pc = banco(X, pos((R.montaje == montaje) & (R.grupo == "calibracion")), capas)
        med = pos((R.montaje == montaje) & (R.grupo == "medicion"))
        filas += filas_z(prueba, R, med, pc.puntaje(sel(X, med)) / pc.umbral)
        del pc
        torch.cuda.empty_cache()
    ei = pos((R.montaje == "ei") & (R.grupo == "calibracion"))
    pc_ei = banco(X, ei, capas)
    s = pc_ei.loo
    otros = np.array([np.delete(s, i).max() for i in range(len(s))])
    filas += filas_z("ei_loo", R, ei, s / otros)
    return pd.DataFrame(filas), pc_ei


# ---------------------------------------------------------------- latencia en CPU (montaje propio)
def latencias(R, X, k):
    img, _ = imagen(("real", 0, 0))
    ms = {"clasica_v3_6": latencia(lambda: cadena(img, sector=False), reps=20),
          "clasica_v3_7": latencia(lambda: cadena(img, sector=True), reps=20)}
    ms["segmentacion"] = latencia(lambda: centro(segmentar(img)[1], img.shape), reps=20)
    buenas = pos((R.montaje == "propio") & (R.giro == 0) & ~R.mala)
    _, cx, cy, req = cadena(img)
    for n, (capas, modo) in PATCHCORES.items():
        cpu = banco(X[modo], buenas, capas).a("cpu")
        lado = LADO_FIJO if modo == "fijo" else k * req
        ms[f"red_{n}"] = latencia(lambda c=cpu, lado=lado: c.puntaje(u8([recortar(img, cx, cy, lado)])), reps=5)
        ms[n] = ms["segmentacion"] + ms[f"red_{n}"]
    return ms


# ---------------------------------------------------------------- criterio
def tasa(d, mala):
    d = d[d.mala == mala]
    return int((d.z > 1).sum()), len(d)


def resumen(L, ms):
    r = {}
    for p in ("A", "R", "sint", "kaggle"):
        d = L[L.prueba == p]
        r[f"{p}_malas"], r[f"{p}_n_malas"] = tasa(d, True)
        r[f"{p}_falsas"], r[f"{p}_n_buenas"] = tasa(d, False)
        r[f"{p}_auc"] = auc(d.mala.values, d.z.values)
    r["ei_falsas"], r["ei_n_buenas"] = tasa(L[L.prueba == "ei_loo"], False)
    r["ms_cpu"] = ms
    r["det_media"] = (r["kaggle_malas"] / r["kaggle_n_malas"] + r["sint_malas"] / r["sint_n_malas"]) / 2
    r["req1"] = bool(r["A_malas"] == r["A_n_malas"] and r["R_malas"] >= 0.95 * r["R_n_malas"] and ms < 100)
    r["req2"] = bool(r["A_falsas"] <= 1 and r["kaggle_falsas"] <= 0.05 * r["kaggle_n_buenas"]
                     and r["ei_falsas"] <= 0.05 * r["ei_n_buenas"])
    return r


def elegir(T):
    ok = T[T.req1 & T.req2]
    ninguno = ok.empty
    if ninguno:
        ok = T[T.ms_cpu < 100] if (T.ms_cpu < 100).any() else T
    emp = ok[ok.det_media >= ok.det_media.max() - 0.02]
    return emp.sort_values(["explica", "ms_cpu"]).index[0], ninguno


def familia(n):
    return "clasica" if n.startswith("clasica") else "patchcore" if n.startswith("patchcore") else "fusion"


def fusionar(a, b):
    m = a.merge(b, on=["prueba", "id", "giro", "mala"], suffixes=("_cl", "_pc"), validate="one_to_one")
    return m.assign(z=np.maximum(m.z_cl, m.z_pc))[["prueba", "id", "giro", "mala", "z"]]


def calcular(recalcular):
    t0 = time.time()
    R, C, _ = preparar(recalcular)
    X, k = C["X"], C["k"]
    print(f"invertidas por la regla P1: {R.groupby('montaje').invertida.sum().to_dict()}", flush=True)
    largos, bancos_ei = {}, {}
    for n, cols in CLASICAS.items():
        largos[n] = eval_clasica(R, cols)
        print(n, f"{time.time() - t0:.0f} s", flush=True)
    for n, (capas, modo) in PATCHCORES.items():
        largos[n], bancos_ei[n] = eval_patchcore(R, X[modo], capas)
        print(n, f"{time.time() - t0:.0f} s", flush=True)
    ms = latencias(R, X, k)
    print("latencias", {a: round(b, 1) for a, b in ms.items()}, flush=True)

    T = pd.DataFrame({n: resumen(L, ms[n]) for n, L in largos.items()}).T
    T["familia"] = [familia(n) for n in T.index]
    T["explica"] = T.familia.map(EXPLICA)
    T = T.infer_objects()
    cl, _ = elegir(T[T.familia == "clasica"])
    pc, _ = elegir(T[T.familia == "patchcore"])
    largos["fusion"] = fusionar(largos[cl], largos[pc])
    ms["fusion"] = ms[cl] + ms[f"red_{pc}"]
    T.loc["fusion"] = pd.Series({**resumen(largos["fusion"], ms["fusion"]), "familia": "fusion",
                                 "explica": EXPLICA["fusion"]})
    T = T.infer_objects()
    T["componentes"] = ""
    T.loc["fusion", "componentes"] = f"{cl} + {pc}"
    elegido, ninguno = elegir(T)
    T["elegido"] = T.index == elegido
    T.to_csv(RESULTADOS / f"e11_candidatos{SUF}.csv")
    pd.concat([L.assign(candidato=n) for n, L in largos.items()]).to_csv(RESULTADOS / f"e11_puntajes{SUF}.csv",
                                                                         index=False)
    e = T.loc[elegido]
    info = dict(elegido=elegido, ninguno_cumple=ninguno, clasica_de_familia=cl, patchcore_de_familia=pc, k=k,
                ms=ms, invertidas=R.groupby("montaje").invertida.sum().astype(int).to_dict(),
                sin_medir_calibracion={m: int((~np.isfinite(R[CLASICAS["clasica_v3_7"]].values).all(1)
                                               & ((R.montaje == m) & (R.grupo == "calibracion")).values).sum())
                                       for m in ("sintetico", "kaggle", "ei")},
                dispositivo=torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
                minutos=round((time.time() - t0) / 60, 1))
    (RESULTADOS / f"e11_eleccion{SUF}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1,
                                                                    default=float))
    if not HUMO:
        decidir("enfoque_v3", elegido,
                "ADR 0014: 6/6 en A, ≥95 % en R y <100 ms; falsas ≤1/15 propias, ≤5 % Kaggle y ≤5 % EI (LOO); "
                "mayor detección media (Kaggle + sintéticas); empate ≤2 puntos → más explicable, más rápida",
                experimento="e11", ninguno_cumple=ninguno, A=f"{e.A_malas}/{e.A_n_malas}",
                A_falsas=f"{e.A_falsas}/{e.A_n_buenas}", R=round(e.R_malas / e.R_n_malas, 3),
                kaggle=round(e.kaggle_malas / e.kaggle_n_malas, 3), sint=round(e.sint_malas / e.sint_n_malas, 3),
                ms_cpu=round(float(e.ms_cpu), 1))
    print(T.drop(columns=["componentes"]).round(3).to_string(), flush=True)

    # --- solo informativo, después de guardar la elección: Edge Impulse de prueba (NO CIEGO)
    cal = pos((R.montaje == "ei") & (R.grupo == "calibracion"))
    med = pos((R.montaje == "ei") & (R.grupo == "prueba_no_ciega"))
    ei = {}
    for n, cols in CLASICAS.items():
        ei[n] = pd.DataFrame(filas_z("ei_prueba", R, med, z_maha(ajustar_maha(R[cols].values[cal]),
                                                                  R[cols].values[med])))
    for n, (_capas, modo) in PATCHCORES.items():
        b = bancos_ei[n]
        ei[n] = pd.DataFrame(filas_z("ei_prueba", R, med, b.puntaje(sel(X[modo], med)) / b.umbral))
    ei["fusion"] = fusionar(ei[cl], ei[pc])
    filas = []
    for n, L in ei.items():
        y, z = L.mala.values, L.z.values
        lo, hi = auc_ic(y, z)
        filas.append(dict(candidato=n, malas=int((z[y] > 1).sum()), n_malas=int(y.sum()),
                          falsas=int((z[~y] > 1).sum()), n_buenas=int((~y).sum()), auc=auc(y, z), auc_lo=lo,
                          auc_hi=hi, no_ciego=True))
    E = pd.DataFrame(filas).set_index("candidato")
    E.to_csv(RESULTADOS / f"e11_ei_no_ciego{SUF}.csv")
    print("Edge Impulse de prueba (NO CIEGO, solo informa):", E.round(3).to_string(), sep="\n", flush=True)
    return T, E


def main():
    estilo()
    if HUMO:
        calcular(True)
        return
    csv = RESULTADOS / "e11_candidatos.csv"
    if csv.exists() and "--recalcular" not in sys.argv:
        T = pd.read_csv(csv, index_col=0)
    else:
        T, _ = calcular("--recalcular" in sys.argv)

    orden = list(CLASICAS) + list(PATCHCORES) + ["fusion"]
    t = T.loc[orden]
    elegido = t.index[t.elegido.astype(bool)][0]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    barras_h(axs[0], orden, [100 * t.A_malas / t.A_n_malas, 100 * t.kaggle_malas / t.kaggle_n_malas,
                             100 * t.sint_malas / t.sint_n_malas],
             ["propias, dejando una fuera (de 6)", "Kaggle (de 100)", "sintéticas de validación (de 300)"],
             colores=[AZUL, AQUA, TINTA2], elegido=elegido, fmt="{:.0f}", lim=(0, 118))
    axs[0].set_title("Malas detectadas (%)", fontsize=10)
    barras_h(axs[1], orden, [100 * t.A_falsas / t.A_n_buenas, 100 * t.kaggle_falsas / t.kaggle_n_buenas,
                             100 * t.ei_falsas / t.ei_n_buenas],
             ["propias (de 15)", "Kaggle (de 100)", "Edge Impulse, buenas de calibración (de 273)"],
             colores=[AZUL, AQUA, NARANJA], elegido=elegido, fmt="{:.0f}", lim=(0, 118))
    axs[1].axvline(5, color=TINTA2, lw=0.8, ls=":")
    axs[1].set_title("Buenas retiradas por error (%) · línea: tope del 5 %", fontsize=10)
    fig.suptitle("N1 v3 (e11): elegir con varios montajes · ★ = elegido por el criterio del ADR 0014",
                 x=0.01, ha="left", fontsize=12, y=1.02)
    guardar(fig, "e11_n1_v3.png")
    cols = ["A_malas", "A_falsas", "R_malas", "R_n_malas", "sint_malas", "kaggle_malas", "kaggle_falsas",
            "ei_falsas", "det_media", "ms_cpu", "req1", "req2", "elegido"]
    print(t[cols].round(3).to_string())


if __name__ == "__main__":
    main()
