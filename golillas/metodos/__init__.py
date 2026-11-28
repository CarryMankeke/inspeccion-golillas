"""Métodos del ADR 0017 con una interfaz común.

    m = crear("padim", perfil)
    m.ajustar(piezas_buenas)               # aprende solo de buenas
    S = m.puntajes_vistas(piezas)          # matriz piezas × vistas; más alto = más raro
    s = S.max(axis=1)                      # puntaje de cada pieza

Los métodos de IA puntúan cada vista y la pieza toma el máximo. La clásica mide solo la vista de
arriba del perfil del tipo, y una pieza que no logra medir recibe ∞ (se retira siempre).
"""

from __future__ import annotations

FAMILIA = {
    "clasica_4": "clasica",
    "clasica_v3_6": "clasica",
    "patchcore_propio": "sin_entrenamiento",
    "patchcore": "sin_entrenamiento",
    "padim": "sin_entrenamiento",
    "anomalydino": "sin_entrenamiento",
    "dinomaly": "con_entrenamiento",
    "winclip": "texto",
}
EXPLICA = {"clasica": 0, "sin_entrenamiento": 1, "con_entrenamiento": 2, "texto": 3}  # desempate del ADR 0017
METODOS = tuple(FAMILIA)


def crear(nombre: str, perfil: dict | None = None, **opciones):
    if FAMILIA[nombre] == "clasica":
        from golillas.metodos.clasica import Clasica

        return Clasica(nombre, perfil)
    if nombre == "patchcore_propio":
        from golillas.metodos.patchcore_propio import PatchCorePropio

        return PatchCorePropio(**opciones)
    from golillas.metodos.ia import Anomalib

    return Anomalib(nombre, **opciones)
