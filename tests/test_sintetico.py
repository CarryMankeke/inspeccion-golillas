import numpy as np

from golillas import sintetico


def test_area_de_la_mascara_coincide_con_el_circulo():
    for semilla in range(5):
        s = sintetico.generar(semilla, defecto="ninguno")
        area = np.pi * (s.r_ext**2 - s.r_int**2)
        assert abs(s.mascara.sum() - area) / area < 0.01


def test_es_reproducible_y_los_defectos_quitan_o_cambian_material():
    a, b = sintetico.generar(7, defecto="muesca_exterior"), sintetico.generar(7, defecto="muesca_exterior")
    assert np.array_equal(a.imagen, b.imagen)
    assert a.clase == "mala"
    area = np.pi * (a.r_ext**2 - a.r_int**2)
    assert a.mascara.sum() < area
