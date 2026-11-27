"""Carga de las 21 fotos reales y sus etiquetas."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
DATOS_REALES = RAIZ / "data" / "real"
RESULTADOS = RAIZ / "resultados"


@dataclass
class Muestra:
    id: str
    ruta: Path
    clase: str  # "buena" | "mala"
    defecto: str  # "ninguno" | tipo de defecto

    @property
    def es_mala(self) -> bool:
        return self.clase == "mala"

    def imagen(self) -> np.ndarray:
        img = cv2.imread(str(self.ruta), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.ruta)
        return img


def cargar_reales() -> list[Muestra]:
    """Devuelve las fotos reales en orden fijo (b01…b15, m01…m06)."""
    muestras = []
    with open(DATOS_REALES / "etiquetas.csv", newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            ruta = DATOS_REALES / fila["archivo"]
            muestras.append(Muestra(ruta.stem, ruta, fila["clase"], fila["defecto"]))
    return muestras
