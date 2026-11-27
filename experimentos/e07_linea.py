"""N3 del proceso 7 · Señal de retiro en una línea simulada.

La visión es la real: cada golilla que pasa es una de las 21 fotos, girada 90°/180°/270°
o espejada (8 versiones exactas), juzgada por un inspector calibrado SIN esa foto (para
no hacer trampa) y con su tiempo de cómputo medido en norea-node.

Se comparan las 3 estrategias de seguimiento (temporizador, encoder, encoder+sensor)
a 9 velocidades de cinta y, a 500 mm/s, con 4 niveles de resbalamiento.

Decisión: la estrategia más simple que, con resbale nominal (0–4 %), retire ≥99 % de
las malas que la visión marcó y arrastre ≤1 % de buenas, en todas las velocidades en
que la visión alcanza a procesar. Orden de simpleza: temporizador < encoder <
encoder+sensor (cada una agrega un sensor).
"""

import json
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from comun import AQUA, AZUL, NARANJA, RESULTADOS, decidir, estilo, guardar, mapa
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from golillas import decision as dec
from golillas.datos import cargar_reales
from golillas.inspector import Inspector
from golillas.linea import ESTRATEGIAS, Parametros, posicion, simular

VELOCIDADES = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
RESBALES = [(0, 0.02), (0, 0.04), (0, 0.08), (0, 0.12)]
ROTACIONES = 8
COLORES = {"temporizador": NARANJA, "encoder": AQUA, "encoder+sensor": AZUL}


def calibracion_sin(id_fuera):
    """Calibración de Mahalanobis con las buenas reales, menos `id_fuera`."""
    df = pd.read_csv(RESULTADOS / "e05_rasgos.csv")
    d = json.loads((RESULTADOS / "decisiones.json").read_text())
    F = d["rasgos"]["opcion"].split(",")
    b = df[(df.split == "real") & (df.clase == "buena") & (df.id != id_fuera)][F].values
    nombre = d["decision"]["opcion"]
    m = dec.modelos()[nombre]
    if nombre == "tolerancias":
        m.k = d["decision"]["k"]
    m.fit(b)
    e = dec.Tolerancias(3).fit(b)
    return dict(rasgos=F, modelo=nombre, parametros=m.parametros(),
                explicacion=dict(mediana=e.med.tolist(), escala=e.esc.tolist(), k=3),
                decisiones={p: v["opcion"] for p, v in d.items()})


def variante(arg):
    """Gira o espeja una foto real (exacto, sin interpolar) y la juzga sin haberla visto."""
    k, (id_, ruta, mala) = arg
    img = cv2.imread(ruta)
    j = k % ROTACIONES
    img = np.rot90(img, j % 4)
    img = np.ascontiguousarray(img[:, ::-1] if j >= 4 else img)
    r = Inspector(calibracion_sin(id_)).inspeccionar(img)
    return dict(id=id_, variante=k, mala=mala, dice_mala=not r.aprobada, ms=r.ms_total, puntaje=r.puntaje)


def main():
    estilo()
    csv = RESULTADOS / "e07_vision_variantes.csv"
    if csv.exists() and "--recalcular" not in sys.argv:
        V = pd.read_csv(csv)
    else:
        fotos = [(m.id, str(m.ruta), m.es_mala) for m in cargar_reales()]
        args = [(i * ROTACIONES + j, f) for i, f in enumerate(fotos) for j in range(ROTACIONES)]
        V = pd.DataFrame(mapa(variante, args))
        V.to_csv(csv, index=False)
    malas, buenas = V[V.mala], V[~V.mala]
    print(f"visión en variantes rotadas (sin ver la foto): malas {malas.dice_mala.mean():.1%}, "
          f"buenas retiradas {buenas.dice_mala.mean():.1%}, ms mediana {V.ms.median():.1f}, p95 {V.ms.quantile(.95):.1f}")

    def vision(p, rng):
        fila = (malas if p.mala else buenas).iloc[int(rng.integers(len(malas if p.mala else buenas)))]
        p.imagen = int(fila.variante)
        return bool(fila.dice_mala), float(fila.ms)

    filas = []
    for v0 in VELOCIDADES:
        for e in ESTRATEGIAS:
            par = Parametros(v0=v0, dt=min(0.0005, 0.25 / v0))
            dur = 1000 * 86 / v0  # ~1000 piezas por corrida
            filas.append(simular(par, e, vision, duracion=dur, semilla=v0).resumen())
    for res in RESBALES:
        for e in ESTRATEGIAS:
            par = Parametros(v0=500, resbale=res, dt=0.0005)
            filas.append(dict(simular(par, e, vision, duracion=172, semilla=7).resumen(), resbale=res[1]))
    T = pd.DataFrame(filas)
    T["retiro_marcadas"] = (T.malas_retiradas) / (1 - T.malas_no_vistas)
    T.to_csv(RESULTADOS / "e07_linea.csv", index=False)
    vel = T[T.resbale.isna()] if "resbale" in T else T
    capacidad = vel[vel.malas_tarde < 0.005].v0.unique()
    ok = {}
    for e in ESTRATEGIAS:
        d = vel[(vel.estrategia == e) & vel.v0.isin(capacidad)]
        ok[e] = bool((d.retiro_marcadas >= 0.99).all() and (d.buenas_arrastradas <= 0.01).all())
    elegida = next((e for e in ESTRATEGIAS if ok[e]), "encoder+sensor")
    decidir("linea", elegida, "la más simple con ≥99 % de malas marcadas retiradas y ≤1 % de buenas arrastradas "
            "en todas las velocidades que la visión alcanza (resbale 0–4 %)",
            velocidad_max_mm_s=int(max(capacidad)), cumple=ok)

    # gráfico 1: velocidad
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for e in ESTRATEGIAS:
        d = vel[vel.estrategia == e]
        axs[0].plot(d.v0, 100 * d.malas_retiradas, "-o", color=COLORES[e], lw=2, ms=5, label=e)
        axs[1].plot(d.v0, 100 * d.buenas_retiradas, "-o", color=COLORES[e], lw=2, ms=5, label=e)
    vis = 100 * (1 - malas.dice_mala.mean())
    axs[0].axhline(100 - vis, color="#52514e", lw=0.8, ls="--")
    axs[0].text(VELOCIDADES[0], 100 - vis - 6, "  techo: lo que la visión marca", color="#52514e", fontsize=8)
    axs[0].set_title("Malas retiradas (%)")
    axs[1].set_title("Buenas retiradas por error (%)")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("velocidad de la cinta (mm/s)")
        ax.set_xticks(VELOCIDADES, [str(v) for v in VELOCIDADES])
    axs[0].set_ylim(0, 105)
    axs[0].legend(loc="lower left", fontsize=8)
    guardar(fig, "e07_linea_velocidad.png")

    # gráfico 2: resbalamiento
    r = T[T.resbale.notna()]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for e in ESTRATEGIAS:
        d = r[r.estrategia == e]
        ax.plot(100 * d.resbale, 100 * d.retiro_marcadas, "-o", color=COLORES[e], lw=2, ms=5, label=e)
    ax.set_xlabel("resbalamiento máximo de la pieza sobre la cinta (%)")
    ax.set_ylabel("malas marcadas que salen de la cinta (%)")
    ax.set_ylim(0, 105)
    ax.set_title("A 500 mm/s: el 2.º sensor paga cuando la pieza resbala")
    ax.legend(loc="lower left", fontsize=8)
    guardar(fig, "e07_linea_resbale.png")

    animar(elegida, vision)
    print(T.round(3).to_string())


def animar(estrategia, vision, v0=400, segundos=8.0, fps=20):
    par = Parametros(v0=v0, p_mala=0.25)
    sim = simular(par, estrategia, vision, duracion=segundos + 2, semilla=11)
    esc, x_min, x_max, H = 2.2, -90, 320, 250
    W = int((x_max - x_min) * esc)
    y_cinta = 120

    def px(x):
        return int(round((x - x_min) * esc))

    from matplotlib import font_manager

    ttf = font_manager.findfont("DejaVu Sans")
    fuente, chica = ImageFont.truetype(ttf, 15), ImageFont.truetype(ttf, 12)
    cuadros = []
    for t in np.arange(1.0, 1.0 + segundos, 1 / fps):
        S = np.interp(t, sim.t, sim.S)
        img = np.full((H, W, 3), 252, np.uint8)
        cv2.rectangle(img, (0, y_cinta - 35), (W, y_cinta + 35), (214, 212, 205), -1)
        for xm in np.arange(-400, 400, 25):  # marcas de la cinta (se mueven con ella)
            xx = px((xm + S) % 425 - 100)
            cv2.line(img, (xx, y_cinta - 35), (xx, y_cinta - 28), (180, 178, 170), 1)
        cv2.rectangle(img, (px(-12), y_cinta - 48), (px(12), y_cinta + 48), (214, 120, 42), 1)
        if estrategia == "encoder+sensor":  # segundo sensor
            xs2 = px(par.L - par.d_sensor2)
            cv2.line(img, (xs2, y_cinta - 38), (xs2, y_cinta + 38), (160, 160, 160), 1)
        cv2.rectangle(img, (px(par.L) - 6, y_cinta - 62), (px(par.L) + 6, y_cinta - 40), (80, 80, 80), -1)
        if any(a <= t <= b for a, b in sim.soplos):
            for k in range(3):
                cv2.line(img, (px(par.L) - 4 + 4 * k, y_cinta - 38), (px(par.L) - 6 + 6 * k, y_cinta + 30),
                         (235, 180, 120), 2)
        cont = dict(n=0, mr=0, m=0, br=0)
        for p in sim.piezas:
            x = posicion(p, S)
            y = y_cinta
            if p.retirada and t >= p.t_retiro:
                y = y_cinta + (t - p.t_retiro) * 900  # sale volando hacia abajo
                x = sim.x(p, p.t_retiro)
            if np.isfinite(p.t_foto) and p.t_foto <= t:
                cont["n"] += 1
                cont["m"] += p.mala
            if p.retirada and t >= p.t_retiro:
                cont["mr"] += p.mala
                cont["br"] += not p.mala
            if not (x_min - 10 < x < x_max + 10) or y > H + 20:
                continue
            decidido = np.isfinite(p.t_decision) and p.t_decision <= t
            color = (150, 150, 150) if not decidido else ((52, 104, 235) if p.dice_mala else (122, 175, 27))
            c = (px(x), int(y))
            cv2.circle(img, c, int(8 * esc), color, -1, cv2.LINE_AA)
            cv2.circle(img, c, int(3.3 * esc), (214, 212, 205) if y == y_cinta else (252, 252, 252), -1,
                       cv2.LINE_AA)
            if p.mala:  # la verdad: las malas llevan una muesca
                cv2.rectangle(img, (c[0] + int(4 * esc), c[1] - 3), (c[0] + int(8.5 * esc), c[1] + 3),
                              (214, 212, 205) if y == y_cinta else (252, 252, 252), -1)
        pil = Image.fromarray(img[..., ::-1])
        dw = ImageDraw.Draw(pil)
        dw.text((8, 6), f"{estrategia} · {v0} mm/s · {int(60 * v0 / 86)} piezas/min aprox.", font=fuente,
                fill=(11, 11, 11))
        dw.text((8, 28), f"inspeccionadas {cont['n']} · malas retiradas {cont['mr']}/{cont['m']} · "
                         f"buenas retiradas {cont['br']}", font=chica, fill=(82, 81, 78))
        dw.text((px(-12) - 4, y_cinta + 52), "cámara", font=chica, fill=(42, 120, 214))
        dw.text((px(par.L) - 22, y_cinta - 80), "soplador", font=chica, fill=(82, 81, 78))
        dw.text((8, H - 20), "gris = sin revisar · verde = aprobada · naranja = retirar · muesca = mala de verdad",
                font=chica, fill=(82, 81, 78))
        cuadros.append(np.array(pil))
    imageio.mimsave(RESULTADOS / "e07_linea.gif", cuadros, duration=1 / fps, loop=0)
    print("  → e07_linea.gif", len(cuadros), "cuadros")


if __name__ == "__main__":
    main()
