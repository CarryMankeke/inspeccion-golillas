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


def test_clase_aiag_con_los_conteos_de_manta():
    from golillas.decision import clase_aiag

    assert clase_aiag(1, 74, 7, 148) == "aceptable"  # 1,4 % y 4,7 %
    assert clase_aiag(2, 74, 7, 148) == "marginal"  # 2,7 % de no detección
    assert clase_aiag(1, 72, 8, 144) == "marginal"  # 5,6 % de falsas
    assert clase_aiag(4, 72, 15, 144) == "inaceptable"  # 10,4 % de falsas


def test_ic_exacto_contiene_la_tasa_y_da_la_cota_del_adr():
    from golillas.decision import ic_exacto

    lo, hi = ic_exacto(1, 74)
    assert lo < 1 / 74 < hi
    assert abs(hi - 0.0730) < 0.001  # IC95 bilateral; la cota unilateral del ADR (6,3 %) es menor
    assert all(np.isnan(ic_exacto(0, 0)))
