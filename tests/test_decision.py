"""Pruebas del umbral conformal (ADR 0015)."""

import numpy as np

from golillas.decision import umbral_conformal


def test_con_33_buenas_y_alfa_3_por_ciento_el_umbral_es_la_mayor():
    s = np.arange(1, 34, dtype=float)  # 33 puntajes
    assert umbral_conformal(s, 0.03) == 33.0


def test_con_32_buenas_y_alfa_3_por_ciento_no_hay_umbral():
    assert umbral_conformal(np.arange(32, dtype=float), 0.03) == float("inf")


def test_el_rango_es_ceil_de_n_mas_1_por_1_menos_alfa():
    s = np.random.default_rng(1).permutation(np.arange(1, 151, dtype=float))  # 150 buenas
    assert umbral_conformal(s, 0.03) == 147.0  # ⌈151 · 0,97⌉ = 147
    assert umbral_conformal(s, 0.015) == 149.0  # ⌈151 · 0,985⌉ = 149


def test_los_puntajes_no_finitos_no_calibran():
    s = np.r_[np.arange(1, 34, dtype=float), np.nan, np.inf]
    assert umbral_conformal(s, 0.03) == 33.0


def test_la_tasa_de_falsas_alarmas_no_pasa_de_alfa_en_promedio():
    rng = np.random.default_rng(0)
    falsas = []
    for _ in range(4000):
        cal, nueva = rng.normal(size=50), rng.normal()
        falsas.append(nueva > umbral_conformal(cal, 0.1))
    assert np.mean(falsas) <= 0.1 + 0.015  # garantía ≤ α, con margen para el ruido de 4.000 sorteos
