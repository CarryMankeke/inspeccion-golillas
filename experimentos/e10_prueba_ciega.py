"""Prueba ciega con datos públicos (ADR 0013): lo elegido hasta e09, medido una sola vez.

Datos: Edge Impulse «Flat washers» (BSD 3-Clause Clear). Sus 273 buenas de entrenamiento
CALIBRAN, como las piezas patrón en planta. Las 191 de prueba (103 malas, 88 buenas) no
eligen nada: ni técnicas, ni umbrales, ni hiperparámetros. Se bajan con
`python scripts/descargar_datos.py edge_impulse_flat_washers`.

Protocolo, fijado antes de correr y mirando solo las buenas de calibración:
- Montaje. La cadena supone una golilla oscura sobre papel claro. Aquí hay fondo blanco
  (golilla más oscura) y cuero negro o gris (golilla más clara).
  P0 = la foto tal cual.
  P1 = polaridad automática, sin etiquetas: si la cola clara de grises (p99,8) se aleja
       más de la mediana que la oscura (p0,2), la golilla es clara y se invierte la foto.
  Se informan ambos; P1 es el principal.
- Clásica: la cadena elegida (P1–P5 de los ADR 0006–0010) y Mahalanobis + Hotelling
  recalibrado con los rasgos de las 273 buenas (las que la cadena no logra medir no
  calibran; se informa cuántas). Además, «sin recalibrar»: la
  calibración de las 15 buenas propias (golillas/calibracion.json), tal cual.
- Red supervisada: la configuración elegida en e09 (preentrenada con sintéticas) y
  ajustada con las 21 reales propias con la receta fija de e09. No hay malas de este
  dataset para entrenarla. Dos umbrales: 0,5 de fábrica, y «calibrado con buenas» = el
  mayor puntaje de las 273 buenas (la misma regla que el umbral de PatchCore).
- PatchCore: la configuración elegida en e09; banco con las 273 buenas, sin giros
  (ya vienen en posiciones al azar); umbral por LOO interno.
- Las redes reciben el recorte de e08 (160 px alrededor del centroide de la
  segmentación clásica) de la foto montada.
Métricas: malas detectadas, buenas retiradas, AUC con IC95 (bootstrap), fallas de medición
y desglose por fondo (claro u oscuro, según la regla de polaridad). No escribe decisiones.
Corre en norea-node (GPU para las redes; mirar `nvidia-smi` antes).

Bloque 2: Roboflow «Washer Defects» de MassMEP (CC BY 4.0, COCO, copia manual verificada con
`python scripts/descargar_datos.py --verificar roboflow_massmep_washers`). Protocolo fijado en
el traspaso: las imágenes sin cajas serían las buenas que calibran, y los tres enfoques se
medirían una sola vez como arriba. Primero se hace el censo (sin correr ningún enfoque): si no
hay buenas, el bloque no puede calibrar ni medir falsas alarmas, y se informa solo el censo.
"""

import json
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from comun import AQUA, AZUL, NARANJA, RESULTADOS, TINTA2, barras_h, decision, estilo, guardar, mapa
from e05_rasgos import extraer
from e08_enfoques import recorte
from e09_ajuste_redes import CAPAS, FINO, PatchCore, auc, auc_ic, datos, entrenar, nueva_red, prob, u8
from matplotlib import pyplot as plt

from golillas import decision as dec
from golillas.datos import RAIZ
from golillas.inspector import CALIBRACION

EI = RAIZ / "data" / "publico" / "edge_impulse_flat_washers"
RF = RAIZ / "data" / "publico" / "roboflow_massmep_washers"
PROTOCOLOS = ("P0", "P1")


def golilla_clara(img):
    """Regla de polaridad sin etiquetas: ¿la cola clara se aleja más de la mediana que la oscura?"""
    lo, med, hi = np.percentile(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), [0.2, 50, 99.8])
    return bool(hi - med > med - lo)


def montar(img, protocolo):
    return 255 - img if protocolo == "P1" and golilla_clara(img) else img


def cargar():
    d = json.loads((EI / "info.labels").read_text())
    return pd.DataFrame([dict(nombre=f["name"], ruta=str(EI / f["path"]), mala=f["label"]["label"] == "anomaly",
                              split="calibracion" if f["category"] == "training" else "prueba") for f in d["files"]])


def trabajo(arg):
    ruta, protocolo = arg
    img = cv2.imread(ruta)
    return dict(ruta=ruta, protocolo=protocolo, golilla_clara=golilla_clara(img), **extraer(montar(img, protocolo)))


def fila(enfoque, protocolo, y, s, umbral, fondo_osc, fallas=np.nan):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    p = s > umbral
    lo, hi = auc_ic(y, s)
    out = dict(enfoque=enfoque, protocolo=protocolo, deteccion=p[y].mean(), falsas=p[~y].mean(),
               malas_ok=int(p[y].sum()), n_malas=int(y.sum()), buenas_mal=int(p[~y].sum()), n_buenas=int((~y).sum()),
               auc=auc(y, s), auc_lo=lo, auc_hi=hi, fallas_medicion=fallas)
    for nombre, sel in (("oscuro", fondo_osc), ("claro", ~fondo_osc)):
        out[f"det_{nombre}"] = p[y & sel].mean() if (y & sel).any() else np.nan
        out[f"falsas_{nombre}"] = p[~y & sel].mean() if (~y & sel).any() else np.nan
    return out


def calcular():
    t0 = time.time()
    meta = cargar()
    cal, pru = meta[meta.split == "calibracion"], meta[meta.split == "prueba"]
    print(f"calibración {len(cal)} buenas; prueba {pru.mala.sum()} malas + {(~pru.mala).sum()} buenas")
    F = decision("rasgos").split(",")
    nombre_dec = decision("decision")
    par_propia = json.loads(CALIBRACION.read_text())["parametros"]
    configs = pd.read_csv(RESULTADOS / "e09_configs.csv")
    sup = configs[(configs.red == "supervisada") & configs.elegida.astype(bool)].iloc[0]
    pcc = configs[(configs.red == "patchcore") & configs.elegida.astype(bool)].iloc[0]

    # --- clásica: rasgos de las 464 fotos en P0 y P1 (CPU, 4 procesos)
    R = pd.DataFrame(mapa(trabajo, [(r, p) for r in meta.ruta for p in PROTOCOLOS])).merge(meta, on="ruta")
    print(f"rasgos listos en {time.time() - t0:.0f} s")
    filas, puntajes = [], []
    for p in PROTOCOLOS:
        rc, rp = R[(R.protocolo == p) & (R.split == "calibracion")], R[(R.protocolo == p) & (R.split == "prueba")]
        y, osc = rp.mala.values, rp.golilla_clara.values
        fallas = int((~np.isfinite(rp[F].values)).any(axis=1).sum())
        medibles = np.isfinite(rc[F].values).all(axis=1)
        print(f"{p}: {int((~medibles).sum())} de {len(rc)} buenas de calibración sin medir; {fallas} fallas en prueba")
        rc = rc[medibles]
        mod = dec.modelos()[nombre_dec].fit(rc[F].values, rc[F].values, np.zeros(len(rc), int))
        s = mod.puntaje(rp[F].values)
        filas.append(dict(fila("clasica", p, y, s, mod.t, osc, fallas), calibracion_sin_medir=int((~medibles).sum())))
        s0 = dec.mahalanobis_desde(par_propia, rp[F].values)
        filas.append(fila("clasica_sin_recalibrar", p, y, s0, par_propia["umbral"], osc, fallas))
        puntajes.append(pd.DataFrame(dict(nombre=rp.nombre.values, protocolo=p, mala=y, fondo_oscuro=osc,
                                          clasica=s, clasica_umbral=mod.t, clasica_sin_recalibrar=s0)))
        print(filas[-2])

    # --- redes: la supervisada se entrena una vez (sintéticas + 21 reales propias)
    D = datos()
    lr, ep = float(sup.lr), int(sup.epocas)
    est = entrenar(nueva_red(), D["entrenamiento"], D["ys"], lr, ep, copias_en=(ep,))[ep]
    net = nueva_red(est)
    entrenar(net, D["Xr"], D["yr"], lr / FINO["div_lr"], FINO["epocas"], lote=FINO["lote"])
    print(f"red supervisada lista ({sup.config}) en {time.time() - t0:.0f} s")
    for k, p in enumerate(PROTOCOLOS):
        rec = {r: u8([recorte(montar(cv2.imread(r), p))]) for r in meta.ruta}
        Xc = torch.cat([rec[r] for r in cal.ruta])
        Xp = torch.cat([rec[r] for r in pru.ruta])
        y, osc = pru.mala.values, puntajes[k].fondo_oscuro.values
        sp, sc = prob(net, Xp), prob(net, Xc)
        filas.append(fila("red_supervisada_umbral_0.5", p, y, sp, 0.5, osc))
        filas.append(fila("red_supervisada_umbral_buenas", p, y, sp, float(sc.max()), osc))
        pc = PatchCore(CAPAS[pcc.capas], float(pcc.coreset)).ajustar(Xc, giros=1)
        spc = pc.puntaje(Xp)
        filas.append(fila("anomalias_red", p, y, spc, pc.umbral, osc))
        puntajes[k] = puntajes[k].assign(red_supervisada=sp, red_umbral_buenas=float(sc.max()), patchcore=spc,
                                         patchcore_umbral=pc.umbral)
        print(filas[-3:], f"{time.time() - t0:.0f} s")

    T = pd.DataFrame(filas)
    T.to_csv(RESULTADOS / "e10_prueba_ciega.csv", index=False)
    pd.concat(puntajes).to_csv(RESULTADOS / "e10_puntajes.csv", index=False)
    (RESULTADOS / "e10_info.json").write_text(json.dumps(dict(
        dispositivo=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        red_supervisada=sup.config, patchcore=pcc.config, minutos=round((time.time() - t0) / 60, 1)),
        ensure_ascii=False))
    return T, pd.concat(puntajes)


def ejemplos(P):
    """Hasta 8 fotos de prueba (P1): primero los errores de la clásica, luego aciertos."""
    q = P[P.protocolo == "P1"].copy()
    q["retirada"] = q.clasica > q.clasica_umbral
    q["error"] = q.retirada != q.mala
    sel = pd.concat([q[q.error].head(4), q[~q.error & q.mala].head(2), q[~q.error & ~q.mala].head(2)]).head(8)
    meta = cargar().drop_duplicates("nombre").set_index("nombre")
    tiles = []
    for _, r in sel.iterrows():
        img = cv2.resize(cv2.imread(meta.loc[r.nombre, "ruta"]), (240, 240))
        ok = not r.error
        color = (60, 160, 40) if ok else (40, 60, 220)
        cv2.rectangle(img, (0, 0), (239, 239), color, 6)
        txt = f"{'mala' if r.mala else 'buena'} -> {'retirada' if r.retirada else 'aprobada'}"
        cv2.rectangle(img, (0, 212), (239, 239), (255, 255, 255), -1)
        cv2.putText(img, txt, (8, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % 4:
        tiles.append(np.full((240, 240, 3), 252, np.uint8))
    filas = [np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(RAIZ / "docs" / "figuras" / "e10_ejemplos.jpg"), np.vstack(filas), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("  → e10_ejemplos.jpg")


def censo_roboflow():
    """Bloque 2: cuenta imágenes, cajas por clase, imágenes sin cajas y días de captura."""
    if not RF.exists():
        print("Roboflow: no está en data/publico (copia manual; ver scripts/descargar_datos.py)")
        return None
    imgs, cajas, por_img = 0, {}, []
    dias = {}
    for split in ("train", "valid", "test"):
        j = json.loads((RF / split / "_annotations.coco.json").read_text())
        nombre = {c["id"]: c["name"] for c in j["categories"]}
        n = {}
        for a in j["annotations"]:
            cajas[nombre[a["category_id"]]] = cajas.get(nombre[a["category_id"]], 0) + 1
            n[a["image_id"]] = n.get(a["image_id"], 0) + 1
        for i in j["images"]:
            imgs += 1
            por_img.append(dict(split=split, archivo=i["file_name"], cajas=n.get(i["id"], 0),
                                tamano=f"{i['width']}x{i['height']}"))
            dia = i["extra"]["name"][4:12]  # WIN_AAAAMMDD_hh_mm_ss_Pro.jpg (cámara web)
            dias[dia] = dias.get(dia, 0) + 1
    sin_cajas = [r for r in por_img if r["cajas"] == 0]
    c = dict(imagenes=imgs, cajas_por_clase=cajas, imagenes_sin_cajas=len(sin_cajas),
             sin_cajas=[r["archivo"] for r in sin_cajas], dias_de_captura=dias,
             tamanos=sorted({r["tamano"] for r in por_img}),
             medible=False,
             motivo=("0 buenas: las imágenes sin cajas también muestran rayones a ojo (anotación incompleta); "
                     "sin buenas no se calibra ni se miden falsas alarmas"))
    (RESULTADOS / "e10_roboflow_censo.json").write_text(json.dumps(c, ensure_ascii=False, indent=1))
    print(f"Roboflow: {imgs} imágenes, cajas {cajas}, {len(sin_cajas)} sin cajas, días {dias} → no medible")
    return c


def main():
    estilo()
    censo_roboflow()
    csv = RESULTADOS / "e10_prueba_ciega.csv"
    if csv.exists() and "--recalcular" not in sys.argv:
        T, P = pd.read_csv(csv), pd.read_csv(RESULTADOS / "e10_puntajes.csv")
    else:
        T, P = calcular()

    orden = ["clasica", "clasica_sin_recalibrar", "red_supervisada_umbral_0.5", "red_supervisada_umbral_buenas",
             "anomalias_red"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 7.6), sharey=True, gridspec_kw=dict(width_ratios=[1.4, 1]))
    for fila_ax, p, nombre in zip(axs, PROTOCOLOS, ["P0 · foto tal cual", "P1 · polaridad automática"]):
        t = T[T.protocolo == p].set_index("enfoque").loc[orden]
        barras_h(fila_ax[0], orden, [100 * t.deteccion, 100 * t.falsas],
                 ["malas detectadas (%, de 103)", "buenas retiradas por error (%, de 88)"], colores=[AZUL, NARANJA],
                 fmt="{:.0f}", lim=(0, 118), leyenda=p == "P1")
        fila_ax[0].set_title(f"{nombre}: punto de operación", fontsize=10)
        barras_h(fila_ax[1], orden, [t.auc], ["AUC"], colores=[AQUA], fmt="", lim=(0.3, 1.15), leyenda=False)
        yy = np.arange(len(orden))[::-1]
        fila_ax[1].errorbar(t.auc, yy, xerr=[t.auc - t.auc_lo, t.auc_hi - t.auc], fmt="none", ecolor=TINTA2,
                            capsize=3, lw=1)
        for yi, v, hi in zip(yy, t.auc, t.auc_hi):
            fila_ax[1].text(hi + 0.015, yi, f"{v:.2f}", va="center", fontsize=8, color=TINTA2)
        fila_ax[1].axvline(0.5, color=TINTA2, lw=0.8, ls=":")
        fila_ax[1].set_title(f"{nombre}: AUC con IC95 (0,5 = al azar)", fontsize=10)
    fig.suptitle("Prueba ciega · Edge Impulse «Flat washers» (191 fotos de otro laboratorio, calibrando con sus "
                 "273 buenas)", x=0.01, ha="left", fontsize=12, y=1.01)
    guardar(fig, "e10_prueba_ciega.png")
    ejemplos(P)
    cols = ["enfoque", "protocolo", "malas_ok", "n_malas", "buenas_mal", "n_buenas", "auc", "auc_lo", "auc_hi",
            "fallas_medicion", "det_oscuro", "falsas_oscuro", "det_claro", "falsas_claro"]
    print(T[cols].round(3).to_string())


if __name__ == "__main__":
    main()
