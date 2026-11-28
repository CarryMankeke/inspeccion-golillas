"""Cargadores del ADR 0017: entregan piezas con sus vistas.

Una pieza es una golilla: un id, sus vistas (imágenes BGR uint8), si es mala, su defecto y su
grupo (ajusta, calibra o prueba). En MANTA cada PNG de 1.280 × 256 trae 5 vistas de 256 × 256
lado a lado; los demás sets traen una vista por pieza.

- `sellado`: MANTA y Roboflow, desde lo sellado en data/sellos/. Cada archivo se comprueba contra
  su huella antes de abrirlo (golillas.sello.cargar).
- `desarrollo`: Kaggle y Edge Impulse, que solo informan, con la partición de e12
  (resultados/e12_particion.csv) para que sus números se puedan comparar.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from golillas import sello
from golillas.datos import RESULTADOS

LADO_MAX = 1024  # px: las fotos más grandes (renuka-p trae 4.024 × 3.036) se reducen al cargarlas (INTER_AREA)
VISTAS_MANTA = 5
KAGGLE = sello.PUBLICO / "kaggle_simplified_washers" / "simplified-washers-for-anomaly-detection"
EDGE_IMPULSE = sello.PUBLICO / "edge_impulse_flat_washers"


@dataclass
class Pieza:
    id: str
    vistas: list[np.ndarray]
    mala: bool
    defecto: str
    grupo: str


def leer(ruta: Path, reducir: bool = True) -> np.ndarray:
    """Lee una foto BGR; si `reducir`, la lleva a LADO_MAX px de lado largo (MANTA no se reduce: trae 5 vistas)."""
    img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(ruta)
    f = LADO_MAX / max(img.shape[:2]) if reducir else 1.0
    if f < 1:
        img = cv2.resize(img, (round(img.shape[1] * f), round(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
    return img


def vistas_manta(img: np.ndarray) -> list[np.ndarray]:
    """Corta las 5 vistas cuadradas que MANTA pone lado a lado."""
    alto, ancho = img.shape[:2]
    if ancho != VISTAS_MANTA * alto:
        raise ValueError(f"se esperaban {VISTAS_MANTA} vistas de {alto} × {alto} y la imagen mide {ancho} × {alto}")
    return [img[:, k * alto : (k + 1) * alto].copy() for k in range(VISTAS_MANTA)]


def submuestra(filas: list, n: int | None, semilla: int = 0) -> list:
    """n filas al azar (semilla fija), en su orden original; todas si n es None o no alcanza."""
    if n is None or n >= len(filas):
        return filas
    return [filas[i] for i in sorted(np.random.default_rng(semilla).choice(len(filas), n, replace=False))]


def raiz_de(conjunto: str) -> Path:
    return sello.PUBLICO / (sello.MANTA if conjunto.startswith("manta_") else conjunto)


def sellado(conjunto: str, grupo: str, n: int | None = None, semilla: int = 0,
            raiz: Path | None = None, destino: Path = sello.SELLOS) -> list[Pieza]:
    """Piezas selladas de un conjunto («manta_red_washer», «roboflow_saksit_metal_washer», …)."""
    filas = submuestra(sello.cargar(conjunto, grupo, raiz or raiz_de(conjunto), destino), n, semilla)
    manta = conjunto.startswith("manta_")
    return [Pieza(r["pieza"], vistas_manta(leer(r["ruta"], reducir=False)) if manta else [leer(r["ruta"])],
                  r["clase"] == "mala",
                  r["defecto"], grupo) for r in filas]


# ---------------------------------------------------------------- sets de desarrollo (solo informan)
def _particion_e12(montaje: str) -> dict[str, str]:
    with open(RESULTADOS / "e12_particion.csv", newline="", encoding="utf-8") as f:
        return {r["id"]: r["papel"] for r in csv.DictReader(f) if r["montaje"] == montaje}


def _kaggle() -> list[tuple[str, Path, bool, str]]:
    papel = _particion_e12("kaggle")
    filas = [(p.stem, p, False, papel[p.stem]) for p in sorted((KAGGLE / "train" / "conform").glob("*.png"))]
    for clase, mala in (("conform", False), ("anomaly", True)):
        filas += [(p.stem, p, mala, "prueba") for p in sorted((KAGGLE / "test" / clase).glob("*.png"))]
    return filas


def _edge_impulse() -> list[tuple[str, Path, bool, str]]:
    """Entrenamiento con la partición de e12 (ajusta, calibra y mide) y la prueba de Edge Impulse, NO CIEGA (e10)."""
    papel = _particion_e12("ei")
    d = json.loads((EDGE_IMPULSE / "info.labels").read_text())
    filas = []
    for f in d["files"]:
        mala = f["label"]["label"] == "anomaly"
        grupo = papel.get(f["name"], "") if f["category"] == "training" else "prueba"
        filas.append((f["name"], EDGE_IMPULSE / f["path"], mala, grupo))
    return filas


DESARROLLO = {"kaggle": _kaggle, "edge_impulse": _edge_impulse}


def desarrollo(conjunto: str, grupo: str, n: int | None = None, semilla: int = 0) -> list[Pieza]:
    """Piezas de un set de desarrollo; con n, una submuestra por clase (para la prueba de humo)."""
    filas = [f for f in DESARROLLO[conjunto]() if f[3] == grupo]
    filas = submuestra([f for f in filas if not f[2]], n, semilla) + submuestra([f for f in filas if f[2]], n, semilla)
    return [Pieza(i, [leer(r)], mala, "mala" if mala else "ninguno", grupo) for i, r, mala, _ in filas]


def cargar(conjunto: str, grupo: str, n: int | None = None, semilla: int = 0) -> list[Pieza]:
    if conjunto in DESARROLLO:
        return desarrollo(conjunto, grupo, n, semilla)
    return sellado(conjunto, grupo, n, semilla)
