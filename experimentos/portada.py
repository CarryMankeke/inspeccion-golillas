"""Imagen de portada del README: 8 golillas reales con el veredicto del inspector.

Cada foto la juzga un inspector calibrado SIN esa foto (dejando una fuera), así la
portada muestra lo mismo que mide la prueba A y no una resustitución.
"""

import cv2
import numpy as np
from comun import AQUA, NARANJA, TINTA, TINTA2, centro, estilo, guardar
from e07_linea import calibracion_sin
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from golillas.datos import cargar_reales
from golillas.inspector import Inspector

ELEGIDAS = ["b01", "b06", "b10", "b14", "m01", "m04", "m05", "m06"]
LADO = 75  # medio lado del recorte (px)


def main():
    estilo()
    reales = {m.id: m for m in cargar_reales()}
    fig, axs = plt.subplots(2, 4, figsize=(12, 7.2))
    for ax, id_ in zip(axs.ravel(), ELEGIDAS):
        m = reales[id_]
        img = m.imagen()
        r = Inspector(calibracion_sin(id_)).inspeccionar(img)
        cx, cy = (r.circulo_ext.cx, r.circulo_ext.cy) if r.circulo_ext and np.isfinite(r.circulo_ext.cx) \
            else centro(r.mascara)
        x0, y0 = int(round(cx)) - LADO, int(round(cy)) - LADO
        p = cv2.copyMakeBorder(img, LADO, LADO, LADO, LADO, cv2.BORDER_REPLICATE)
        rec = p[y0 + LADO : y0 + 3 * LADO, x0 + LADO : x0 + 3 * LADO, ::-1]
        ax.imshow(rec)
        color = AQUA if r.aprobada else NARANJA
        for c in (r.circulo_ext, r.circulo_int):
            if c is not None and np.isfinite(c.r):
                ax.add_patch(Circle((c.cx - x0, c.cy - y0), c.r, fill=False, lw=1.4, color=color))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(color)
            s.set_linewidth(3)
        ax.set_title(("✓ APROBADA" if r.aprobada else "✗ RETIRAR") + f"  ·  {id_}", fontsize=11, color=TINTA,
                     fontweight="bold", loc="left")
        razon = r.razones[0].split(" (")[0] if r.razones else "forma dentro de lo normal"
        etiqueta = "real: " + ("buena" if m.clase == "buena" else m.defecto.replace("_", " "))
        ax.text(0, -0.04, f"{razon}\n{etiqueta}", transform=ax.transAxes, va="top", fontsize=8.5, color=TINTA2)
    fig.suptitle("Fotos reales juzgadas por el inspector (cada una sin haberla visto al calibrar) · "
                 "círculos = medición subpíxel", x=0.01, ha="left", fontsize=12)
    fig.subplots_adjust(hspace=0.45, wspace=0.08)
    guardar(fig, "portada.png")


if __name__ == "__main__":
    main()
