"""Pruebas de regresión del inspector con las 21 fotos reales.

Ojo: la calibración se hizo con estas mismas 15 buenas, así que esto NO mide
exactitud (eso está en experimentos/e06_decision.py, dejando una fuera). Sirve para
detectar si un cambio de código rompe algo que funcionaba.
"""

import numpy as np

from golillas.datos import cargar_reales
from golillas.inspector import Inspector


def test_las_21_fotos_reales_se_clasifican_como_su_etiqueta():
    insp = Inspector()
    errores, tiempos = [], []
    for m in cargar_reales():
        r = insp.inspeccionar(m.imagen())
        tiempos.append(r.ms_total)
        if r.aprobada == m.es_mala:
            errores.append(m.id)
        if not r.aprobada:
            assert r.razones, m.id
    assert errores == []
    assert np.median(tiempos) < 250  # holgado: los runners de CI son más lentos que norea-node
