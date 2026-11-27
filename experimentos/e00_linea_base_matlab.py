"""N0 · Línea base: las reglas del proyecto MATLAB original, portadas a Python y medidas igual.

Qué hacía el original (repo privado whashers-clasic-artificial-vision, classifyWasher.m):
1. Gris → recorte: binarización adaptativa (imbinarize 'adaptive', objeto oscuro,
   sensibilidad 0,7), rellenar huecos, quitar manchas < 0,5 %, caja de la mayor + 5 px.
2. En el recorte, la misma binarización → circularidad 4πA/P², solidez y número de Euler
   de la mayor mancha.
3. Bordes Sobel + cierre (disco 3) → Hough de círculos: exterior con radio en [50, 54] px;
   agujero con radio en [0,35; 0,5]·R dentro de 0,8·R del centro.
4. Aprobada si: una sola mancha, Euler = 1, circularidad en [0,8; 1,2], solidez ≥ 0,95 y
   exactamente 1 agujero.
Esos umbrales se fijaron mirando estas mismas 21 fotos (informó 21/21, sin validación aparte).

Port: binarización Bradley sobre la oscuridad (ventana 2·⌊lado/16⌋+1, como adaptthresh), Sobel con el umbral
automático de MATLAB (√(4·media(|∇|²))) y adelgazado, y un Hough propio por votos con los
mismos rangos de radio. La "sensibilidad" de imfindcircles no tiene equivalente exacto: se
reemplaza por un soporte mínimo (votos / 2πr) para el exterior y otro para el agujero,
elegidos para reproducir los veredictos correctos en las fotos reales (como el original).
**No verificado contra MATLAB** (norea no tiene MATLAB): es un port de las reglas, no del binario.

Pruebas, las mismas del proceso 6 y de e08:
- A: 21 reales dejando una fuera (los dos soportes se eligen con las otras 20),
- R: las 7 variantes exactas (giros de 90° y espejo) de cada real,
- C: las 600 sintéticas de prueba (semilla 6) y el subconjunto con el radio de las reales.
"""

import sys
import time

import cv2
import numpy as np
import pandas as pd
from comun import AZUL, GRIS, RESULTADOS, TINTA2, barras_h, decidir, estilo, guardar, mapa
from matplotlib import pyplot as plt

from golillas import sintetico
from golillas.datos import cargar_reales

LISTA = ["ninguno"] * 300 + [d for d in sintetico.DEFECTOS for _ in range(50)]
CFG = dict(oscuridad=1.1, min_area=0.005, margen=5, disco=3, circ=(0.8, 1.2), solidez=0.95,
           r_ext=(50, 54), f_int=(0.35, 0.5))
SOPORTES = np.round(np.arange(0.1, 4.01, 0.1), 2)


# ------------------------------------------------------------------ piezas del port
def binarizar(g):
    """imbinarize(g,'adaptive','ForegroundPolarity','dark'): objeto oscuro = 1.

    Bradley sobre la oscuridad (255 − g) con la ventana de adaptthresh. La constante 1,1
    se eligió para parecerse a las figuras del MATLAB (Golilla 10: circularidad 0,85 y
    solidez 0,98; el port da 0,88 y 0,99). El recorte sale más chico que el del MATLAB
    (~110 px contra ~150 px): el original incluía parte de la sombra."""
    kh, kw = 2 * (g.shape[0] // 16) + 1, 2 * (g.shape[1] // 16) + 1
    d = 255 - g.astype(np.float32)
    media = cv2.boxFilter(d, -1, (kw, kh), borderType=cv2.BORDER_REPLICATE)
    return (d > CFG["oscuridad"] * media).astype(np.uint8)


def rellenar(bw):
    """imfill(bw,'holes'): todo lo que no toca el borde de la imagen pasa a 1."""
    h, w = bw.shape
    inv = (1 - bw).astype(np.uint8)
    marco = np.ones((h + 2, w + 2), np.uint8)  # marco de fondo para que el relleno entre por todo el borde
    marco[1:-1, 1:-1] = inv
    cv2.floodFill(marco, None, (0, 0), 2)
    return np.where(marco[1:-1, 1:-1] == 2, 0, 1).astype(np.uint8)


def quitar_chicas(bw, frac):
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= round(bw.size * frac)
    return keep[lab].astype(np.uint8), int(keep.sum())


def recortar(g):
    bw, n = quitar_chicas(rellenar(binarizar(g)), CFG["min_area"])
    if n == 0:
        return g
    n_, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = st[i, :4]
    m = CFG["margen"]
    return g[max(y - m, 0) : y + h + m, max(x - m, 0) : x + w + m]


def sobel_matlab(g):
    """edge(I,'sobel'): umbral automático √(4·media(b)) con b = gx²+gy² y adelgazado."""
    f = g.astype(np.float32) / 255
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3) / 8
    b = gx * gx + gy * gy
    corte = 4 * b.mean()
    p = np.pad(b, 1, mode="edge")
    max_x = (b > p[1:-1, :-2]) & (b >= p[1:-1, 2:])
    max_y = (b > p[:-2, 1:-1]) & (b >= p[2:, 1:-1])
    fino = np.where(np.abs(gx) >= np.abs(gy), max_x, max_y)
    return ((b > corte) & fino).astype(np.uint8)


def hough(bordes, radios, mascara=None):
    """Acumulador de centros por radio; devuelve el máximo sobre radios del soporte (votos/2πr)
    y el radio que lo da, por píxel."""
    ys, xs = np.nonzero(bordes if mascara is None else bordes & mascara)
    h, w = bordes.shape
    sop = np.zeros((h, w), np.float32)
    rad = np.zeros((h, w), np.float32)
    if xs.size == 0:
        return sop, rad
    for r in radios:
        ang = np.linspace(0, 2 * np.pi, max(32, int(round(2 * np.pi * r))), endpoint=False)  # ~1 px por paso
        c, s = np.cos(ang), np.sin(ang)
        cx = np.rint(xs[:, None] - r * c).astype(np.int32)
        cy = np.rint(ys[:, None] - r * s).astype(np.int32)
        ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        votos = np.bincount((cy * w + cx)[ok], minlength=h * w).reshape(h, w).astype(np.float32)
        # votos a ±1 px del centro, por píxel de circunferencia: ~1 = círculo completo de 1 px de grosor
        votos = cv2.boxFilter(votos, -1, (3, 3), normalize=False) / (2 * np.pi * r)
        mejor = votos > sop
        sop[mejor], rad[mejor] = votos[mejor], r
    return sop, rad


def picos(sop, ventana):
    """Máximos locales (no supresión en una ventana) con su soporte, de mayor a menor."""
    k = max(3, int(ventana) | 1)
    dil = cv2.dilate(sop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ys, xs = np.nonzero((sop >= dil) & (sop > 0))
    v = sop[ys, xs]
    o = np.argsort(-v)
    return v[o], xs[o], ys[o]


def medir(img):
    """Todo lo que usan las reglas; los umbrales de soporte se aplican después."""
    t0 = time.perf_counter()
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    c = recortar(g)
    bw, n_manchas = quitar_chicas(rellenar(binarizar(c)), CFG["min_area"])
    out = dict(n_manchas=n_manchas)
    if n_manchas == 0:
        return {**out, "circularidad": np.nan, "solidez": np.nan, "euler": 0, "sop_ext": 0.0, "r_hough": np.nan,
                **{f"int{i}": 0.0 for i in range(5)}, "ms": (time.perf_counter() - t0) * 1000}
    n_, lab, st, cen = cv2.connectedComponentsWithStats(bw, connectivity=8)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    blob = (lab == i).astype(np.uint8)
    cont, jer = cv2.findContours(blob, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    ext = max(cont, key=cv2.contourArea)
    area = float(blob.sum())
    per = cv2.arcLength(ext, True)
    hull = cv2.contourArea(cv2.convexHull(ext))
    n_huecos = sum(1 for k in range(len(cont)) if jer[0][k][3] >= 0)
    out.update(circularidad=4 * np.pi * area / per**2 if per else np.nan,
               solidez=cv2.contourArea(ext) / hull if hull else np.nan,
               euler=1 - n_huecos)
    # Hough exterior sobre Sobel + cierre + limpieza
    e = sobel_matlab(c)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CFG["disco"] + 1,) * 2))
    e, _ = quitar_chicas(e, 0.001)
    sop, rad = hough(e, range(CFG["r_ext"][0], CFG["r_ext"][1] + 1))
    v, xs, ys = picos(sop, 7)
    out["sop_ext"] = float(v[0]) if v.size else 0.0
    R = float(rad[ys[0], xs[0]]) if v.size else np.nan
    out["r_hough"] = R
    # Hough del agujero: dentro de 0,8·R del centroide de la mancha
    if np.isfinite(R):
        X, Y = np.meshgrid(np.arange(c.shape[1]), np.arange(c.shape[0]))
        cx, cy = cen[i]
        roi = ((X - cx) ** 2 + (Y - cy) ** 2 <= (0.8 * R) ** 2).astype(np.uint8)
        r0, r1 = np.round(R * np.array(CFG["f_int"])).astype(int)
        si, ri = hough(e, range(max(r0, int(np.ceil(0.2 * R))), r1 + 1), roi)
        vi, _, _ = picos(si, r0)
    else:
        vi = np.zeros(0)
    vi = np.concatenate([vi[:5], np.zeros(max(0, 5 - vi.size))])
    out.update({f"int{k}": float(vi[k]) for k in range(5)})
    out["ms"] = (time.perf_counter() - t0) * 1000
    return out


def forma_ok(df):
    return ((df.n_manchas == 1) & (df.euler == 1) & df.circularidad.between(*CFG["circ"])
            & (df.solidez >= CFG["solidez"])).values


def aprobar(df, t_ext, t_int):
    picos_int = df[[f"int{k}" for k in range(5)]].values
    un_agujero = (picos_int >= t_int).sum(axis=1) == 1
    return forma_ok(df) & (df.sop_ext.values >= t_ext) & un_agujero


def reglas(df):
    """Qué regla bota cada pieza (la primera que falla, en el orden del MATLAB)."""
    return np.select([df.n_manchas != 1, df.euler != 1, ~df.circularidad.between(*CFG["circ"]),
                      df.solidez < CFG["solidez"]], ["manchas", "euler", "circularidad", "solidez"], "hough")


def calibrar(df):
    """Par de soportes que acierta más veredictos; empate → el centro de la zona que empata."""
    y = df.clase.eq("mala").values
    ext_ok = df.sop_ext.values[None, :] >= SOPORTES[:, None]
    n_int = (df[[f"int{k}" for k in range(5)]].values[None] >= SOPORTES[:, None, None]).sum(2)
    apr = forma_ok(df)[None, None] & ext_ok[:, None] & (n_int == 1)[None]
    T = (apr == y).mean(2)  # error: aprobó una mala o retiró una buena
    ok = T <= T.min() + 1e-9
    ia, ib = np.nonzero(ok)
    k = np.argmin((ia - ia.mean()) ** 2 + (ib - ib.mean()) ** 2)
    return SOPORTES[ia[k]], SOPORTES[ib[k]], float(1 - T.min()), T


# ------------------------------------------------------------------ trabajos
def girar(img, k):
    img = np.rot90(img, k % 4)
    return np.ascontiguousarray(img[:, ::-1] if k >= 4 else img)


def trabajo_real(m):
    return dict(split="real", id=m.id, variante=0, clase=m.clase, defecto=m.defecto, r_real=np.nan,
                **medir(m.imagen()))


def trabajo_giro(arg):
    m, k = arg
    return dict(split="real_giro", id=m.id, variante=k, clase=m.clase, defecto=m.defecto, r_real=np.nan,
                **medir(girar(m.imagen(), k)))


def trabajo_sint(i):
    s = sintetico.generar(6 * 100_000 + i, defecto=LISTA[i])  # las 600 de prueba de e05/e06
    return dict(split="sintetica", id=f"p{i:03d}", variante=0, clase=s.clase, defecto=s.defecto, r_real=s.r_ext,
                **medir(s.imagen))


def metricas(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)  # p = retirar
    return float(p[y].mean()) if y.any() else np.nan, float(p[~y].mean()) if (~y).any() else np.nan


def main():
    estilo()
    csv = RESULTADOS / "e00_matlab.csv"
    if csv.exists() and "--recalcular" not in sys.argv:
        df = pd.read_csv(csv)
    else:
        reales = cargar_reales()
        df = pd.DataFrame(mapa(trabajo_real, reales) + mapa(trabajo_giro, [(m, k) for m in reales for k in range(1, 8)])
                          + mapa(trabajo_sint, list(range(len(LISTA)))))
        df.to_csv(csv, index=False)

    re = df[df.split == "real"].reset_index(drop=True)
    gi = df[df.split == "real_giro"]
    si = df[df.split == "sintetica"]
    yre = re.clase.eq("mala").values

    # calibración como el original (las 21) y prueba honesta dejando una fuera
    t_ext, t_int, acierto, _ = calibrar(re)
    pa = np.zeros(len(re), bool)
    for i in range(len(re)):
        a, b, _, _ = calibrar(re.drop(index=i))
        pa[i] = ~aprobar(re.iloc[[i]], a, b)[0]
    retirar = {n: ~aprobar(d, t_ext, t_int) for n, d in (("real", re), ("giro", gi), ("sint", si))}
    print(f"soportes elegidos: exterior {t_ext}, agujero {t_int}; acierto en las 21 (resustitución) {acierto:.3f}")

    rr = pd.read_csv(RESULTADOS / "e05_rasgos.csv")
    r_reales = rr[rr.split == "real"].radio_ext
    rango = (r_reales.min() - 1, r_reales.max() + 1)
    en_rango = si.r_real.between(*rango).values
    filas = {}
    filas["resustitucion"] = metricas(yre, retirar["real"])
    filas["A"] = metricas(yre, pa)
    filas["R"] = metricas(gi.clase.eq("mala").values, retirar["giro"])
    filas["C"] = metricas(si.clase.eq("mala").values, retirar["sint"])
    filas["C_rango"] = metricas(si.clase.eq("mala").values[en_rango], retirar["sint"][en_rango])
    M = pd.DataFrame(filas, index=["deteccion", "falsas"]).T
    M.to_csv(RESULTADOS / "e00_matlab_metricas.csv")

    # por qué se bota cada sintética buena (falsas alarmas) y qué defectos se escapan
    sb = si[si.clase == "buena"]
    motivo = pd.Series(np.where(retirar["sint"][si.clase.eq("buena").values], reglas(sb), "aprobada")).value_counts()
    por_def = si[si.clase == "mala"].assign(ret=retirar["sint"][si.clase.eq("mala").values]).groupby("defecto").ret.mean()
    motivo.to_csv(RESULTADOS / "e00_matlab_motivos.csv")
    por_def.to_csv(RESULTADOS / "e00_matlab_defectos.csv")

    nueva = pd.read_csv(RESULTADOS / "e06_decision.csv").set_index("modelo").loc["mahalanobis_hotelling"]
    decidir("linea_base", "reglas_matlab",
            "port de classifyWasher.m; soportes de Hough elegidos para reproducir los veredictos (como el original)",
            soporte_ext=float(t_ext), soporte_int=float(t_int),
            resustitucion=f"{acierto:.3f}", A=dict(deteccion=round(M.loc['A', 'deteccion'], 3),
                                                   falsas=round(M.loc['A', 'falsas'], 3)),
            R=dict(deteccion=round(M.loc['R', 'deteccion'], 3), falsas=round(M.loc['R', 'falsas'], 3)),
            C=dict(deteccion=round(M.loc['C', 'deteccion'], 3), falsas=round(M.loc['C', 'falsas'], 3)),
            C_rango=dict(deteccion=round(M.loc['C_rango', 'deteccion'], 3),
                         falsas=round(M.loc['C_rango', 'falsas'], 3), radio_px=[round(x, 1) for x in rango]),
            ms_mediana=round(float(df.ms.median()), 1))

    # ---------------- figura: reglas MATLAB vs cadena nueva
    pruebas = ["A", "R", "C"]
    nombres_p = {"A": "A · 21 reales,\ndejando una fuera", "R": "R · reales\ngiradas", "C": "C · 600\nsintéticas"}
    fig, axs = plt.subplots(1, 3, figsize=(16, 4), gridspec_kw=dict(width_ratios=[1, 1, 1.3], wspace=0.45))
    nombres, colores = ["cadena nueva", "reglas MATLAB"], [AZUL, GRIS]
    x = np.arange(len(pruebas))
    for ax, col, titulo in ((axs[0], "deteccion", "Malas detectadas (%) · más es mejor"),
                            (axs[1], "falsas", "Buenas retiradas por error (%) · menos es mejor")):
        vals = [[100 * nueva[f"{p}_{col}"] for p in pruebas], [100 * M.loc[p, col] for p in pruebas]]
        for k in range(2):
            xx = x + (k - 0.5) * 0.38
            ax.bar(xx, vals[k], 0.34, color=colores[k], label=nombres[k])
            for xi, v in zip(xx, vals[k]):
                ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=TINTA2)
        ax.set_xticks(x, [nombres_p[p] for p in pruebas], fontsize=8)
        ax.set_ylim(0, 115 if col == "deteccion" else max(10, max(max(v) for v in vals) * 1.25))
        ax.set_title(titulo, fontsize=10)
        ax.grid(axis="x", visible=False)
    defectos = list(por_def.sort_values().index)
    barras_h(axs[2], [d.replace("_", " ") for d in defectos],
             [100 * nueva[[f"C_{d}" for d in defectos]].values.astype(float), 100 * por_def[defectos].values],
             nombres, colores=colores, fmt="{:.0f}", lim=(0, 118), leyenda=False)
    axs[2].set_title("% detectado por tipo de defecto (600 sintéticas)", fontsize=10)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Línea base · las reglas a mano del MATLAB aciertan {acierto * 21:.0f}/21 en las fotos con que se "
                 "ajustaron, pero fallan fuera de ellas", x=0.01, ha="left", fontsize=12, y=1.04)
    guardar(fig, "e00_matlab.png")
    print(M.round(3).to_string())
    print("motivos de falsa alarma (sintéticas buenas):", motivo.to_dict())
    print("detección por defecto:", por_def.round(2).to_dict())
    print("radio Hough en reales:", re.r_hough.min(), "–", re.r_hough.max(), "· ms mediana",
          round(float(df.ms.median()), 1))


if __name__ == "__main__":
    main()
