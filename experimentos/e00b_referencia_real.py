"""Máscaras de referencia para las 21 fotos reales (no hay etiquetado a mano).

Cada píxel queda como golilla si la mayoría de 16 segmentaciones distintas lo marca
(8 técnicas × 2 preprocesados). Después se revisa a ojo el mosaico
`resultados/e00b_referencia_real.jpg`. Limitación: si la mayoría se equivoca igual,
la referencia también; por eso la segmentación se mide además con sintéticas.
"""

import cv2
import numpy as np
from comun import REFERENCIA, RESULTADOS, centro, recorte

from golillas.datos import cargar_reales
from golillas.preproceso import preprocesar
from golillas.segmentacion import limpiar, segmentar

PRE = ("gauss", "fondo_div+gauss")
SEG = ("otsu", "triangulo", "adapt_gauss", "sauvola", "canny_relleno", "grabcut", "watershed", "chan_vese")


def main():
    REFERENCIA.mkdir(parents=True, exist_ok=True)
    tiles = []
    for m in cargar_reales():
        img = m.imagen()
        votos = np.zeros(img.shape[:2], np.float32)
        for p in PRE:
            g = preprocesar(img, p)
            for s in SEG:
                votos += segmentar(g, s)
        ref = limpiar((votos > len(PRE) * len(SEG) / 2).astype(np.uint8))
        cv2.imwrite(str(REFERENCIA / f"{m.id}.png"), ref * 255)
        over = img.copy()
        cnts, _ = cv2.findContours(ref, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(over, cnts, -1, (0, 0, 255), 1)
        t = recorte(over, *centro(ref), r=75, lado=220)
        cv2.putText(t, m.id, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 0), 1)
        tiles.append(t)
    while len(tiles) % 7:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(RESULTADOS / "e00b_referencia_real.jpg"),
                np.vstack([np.hstack(tiles[i : i + 7]) for i in range(0, len(tiles), 7)]))
    print("referencias:", len(list(REFERENCIA.glob("*.png"))))


if __name__ == "__main__":
    main()
