"""Embudo del ADR 0019: punto ROC con ≤ 5 % de falsas y McNemar exacto (scripts/combinaciones.py)."""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "combinaciones", Path(__file__).resolve().parents[1] / "scripts" / "combinaciones.py")
comb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comb)


def test_punto_roc_deja_a_lo_mas_5_por_ciento_de_falsas():
    buenas = np.arange(40, dtype=float)  # floor(0,05 · 40) = 2 falsas permitidas
    malas = np.array([38.5, 37.5, 10.0])
    s = np.concatenate([buenas, malas])
    y = np.array([False] * 40 + [True] * 3)
    u, det, fa = comb.punto_roc(s, y)
    assert fa == 2 and u == 37.0 and det == 1  # solo la de 38,5 supera el umbral


def test_punto_roc_con_piezas_sin_medir():
    s = np.array([np.inf, np.inf, np.inf, 1.0, 2.0, 5.0, 9.0])
    y = np.array([False, False, False, False, False, True, True])
    u, det, fa = comb.punto_roc(s, y)  # 3 buenas sin medir > floor(0,05 · 5) = 0: no hay umbral que cumpla
    assert u == np.inf and fa == 3 and det == 0


def test_mcnemar_no_distingue_por_una_pieza():
    a = np.ones(74, bool)
    b = a.copy()
    b[0] = False
    assert comb.mcnemar(a, b) == 0
    b[:10] = False
    assert comb.mcnemar(a, b) == 1 and comb.mcnemar(b, a) == -1
