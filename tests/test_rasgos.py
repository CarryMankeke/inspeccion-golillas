"""Prueba del rasgo de relleno mínimo por sector (ADR 0014): ve un corte radial fino."""

import cv2
import numpy as np

from golillas.medicion import Circulo
from golillas.rasgos import relleno_sector_min


def _anillo(corte_px=0):
    m = np.zeros((200, 200), np.uint8)
    cv2.circle(m, (100, 100), 60, 1, -1)
    cv2.circle(m, (100, 100), 26, 0, -1)
    if corte_px:
        m[100 - corte_px // 2 : 100 + (corte_px + 1) // 2, 100:] = 0  # corte radial hacia la derecha
    return m


def test_anillo_sano_queda_cerca_de_1_y_el_corte_lo_baja():
    ce, ci = Circulo(100, 100, 60), Circulo(100, 100, 26)
    sano = relleno_sector_min(_anillo(), ce, ci)
    cortado = relleno_sector_min(_anillo(3), ce, ci)
    assert sano > 0.97
    assert cortado < sano - 0.03


def test_sin_circulo_exterior_no_mide():
    assert np.isnan(relleno_sector_min(_anillo(), Circulo(np.nan, np.nan, np.nan), Circulo(100, 100, 26)))
