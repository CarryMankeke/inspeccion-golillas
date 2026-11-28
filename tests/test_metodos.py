"""Pruebas de la clásica como método del ADR 0017 y del perfil de un tipo (sin PyTorch)."""

import cv2
import numpy as np

from golillas import sintetico
from golillas.conjuntos import Pieza
from golillas.metodos import EXPLICA, FAMILIA, METODOS
from golillas.metodos.clasica import Clasica, montar, perfil


def _piezas(n, semilla=0, defecto="ninguno", escala=1.0, clara=False):
    out = []
    for i in range(n):
        img = sintetico.generar(semilla * 1000 + i, defecto=defecto).imagen
        img = montar(img, clara, escala)  # golillas «claras» y más grandes: el perfil debe deshacerlo
        out.append(Pieza(f"{defecto}{i}", [np.full_like(img, 255), img], defecto != "ninguno", defecto, "ajusta"))
    return out


def test_familias_y_desempate_del_adr():
    assert len(METODOS) == 8
    assert {FAMILIA[m] for m in METODOS} == set(EXPLICA)
    assert EXPLICA["clasica"] < EXPLICA["sin_entrenamiento"] < EXPLICA["con_entrenamiento"] < EXPLICA["texto"]


def test_el_perfil_encuentra_la_vista_la_polaridad_y_la_escala():
    p = perfil("falso", _piezas(12, escala=1.6, clara=True))
    assert p["vista_arriba"] == 1  # la vista 0 es blanca: no mide nada
    assert p["invertir"] is True
    assert 0.5 < p["escala"] < 0.8  # deshace el 1,6
    assert p["clasica_aplica"]


def test_la_clasica_retira_lo_que_no_mide_y_puntua_las_buenas_bajo():
    aj = _piezas(40, semilla=1)
    per = perfil("falso", aj)
    m = Clasica("clasica_v3_6", per).ajustar(aj)
    buenas = m.puntajes_vistas(_piezas(10, semilla=2))[:, 0]
    blanca = Pieza("x", [np.full((480, 640, 3), 255, np.uint8)] * 2, True, "x", "prueba")
    assert np.isfinite(buenas).all()
    assert np.isposinf(m.puntajes_vistas([blanca])[0, 0])


def test_montar_invierte_y_escala():
    img = np.zeros((100, 200, 3), np.uint8)
    out = montar(img, True, 0.5)
    assert out.shape == (50, 100, 3) and out.max() == 255
    assert cv2.countNonZero(cv2.cvtColor(montar(img, False, 1.0), cv2.COLOR_BGR2GRAY)) == 0
