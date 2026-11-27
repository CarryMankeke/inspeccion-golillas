"""Uso: python -m golillas foto1.jpg [foto2.jpg ...]"""

import sys

import cv2

from .inspector import Inspector


def main(rutas):
    if not rutas:
        print(__doc__)
        return 1
    insp = Inspector()
    for ruta in rutas:
        img = cv2.imread(ruta)
        if img is None:
            print(f"{ruta}: no se pudo leer")
            continue
        r = insp.inspeccionar(img)
        estado = "APROBADA" if r.aprobada else "RETIRAR "
        print(f"{estado}  {ruta}  (puntaje {r.puntaje:.1f} / umbral {r.umbral:.1f}, {r.ms_total:.0f} ms)")
        for razon in r.razones:
            print(f"          · {razon}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
