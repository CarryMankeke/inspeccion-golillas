"""La visión clásica (P1–P6) como método del ADR 0017, más el perfil de cada tipo de golilla.

El perfil se arma solo con las buenas que ajustan y fija, para la clásica, qué vista mirar, la
polaridad y la escala. Así la misma cadena del N0 v1 (golillas oscuras de radio ≈ 50 px) mide
golillas de otro color y tamaño.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from golillas import bordes, medicion, preproceso, rasgos, segmentacion
from golillas import decision as dec

CAL = json.loads((Path(__file__).resolve().parents[1] / "calibracion.json").read_text(encoding="utf-8"))
D = CAL["decisiones"]
F4 = list(CAL["rasgos"])
COLUMNAS = {"clasica_4": F4, "clasica_v3_6": F4 + ["razon_radios", "radio_ext"]}
RADIO_OBJETIVO = 50.0  # px: el radio exterior de las fotos del N0 v1
MIDE_MIN = 0.90  # la clásica «no aplica» a un tipo si mide menos del 90 % de las que ajustan (ADR 0017)
TIPOS = Path(__file__).resolve().parents[1] / "tipos"


def golilla_clara(img: np.ndarray) -> bool:
    """Regla de polaridad sin etiquetas de e10: ¿la cola clara se aleja más de la mediana que la oscura?"""
    lo, med, hi = np.percentile(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), [0.2, 50, 99.8])
    return bool(hi - med > med - lo)


def montar(img: np.ndarray, invertir: bool, escala: float) -> np.ndarray:
    img = 255 - img if invertir else img
    if abs(escala - 1) < 1e-6:
        return img
    interp = cv2.INTER_AREA if escala < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(8, round(img.shape[1] * escala)), max(8, round(img.shape[0] * escala))),
                      interpolation=interp)


def radio_equivalente(img: np.ndarray) -> float:
    """sqrt(área del relleno exterior / π) de la segmentación; nan si no segmenta."""
    try:
        m = segmentacion.segmentar(preproceso.preprocesar(img, D["preproceso"]), D["segmentacion"])
        relleno, _ = segmentacion.partes(m)
        a = float(relleno.sum())
    except Exception:  # noqa: BLE001 — una vista que no segmenta no cuenta
        return float("nan")
    return float(np.sqrt(a / np.pi)) if a > 0 else float("nan")


def cadena(img: np.ndarray) -> dict:
    """Los rasgos de la cadena clásica completa; todos nan si alguna etapa falla."""
    try:
        g = preproceso.preprocesar(img, D["preproceso"])
        m = segmentacion.segmentar(g, D["segmentacion"])
        pe, pi = bordes.bordes(g, m, D["bordes"])
        ce, ci = medicion.medir(g, m, pe, pi, D["medicion"])
        return rasgos.calcular(g, m, pe, pi, ce, ci)
    except Exception:  # noqa: BLE001 — sin medir = se retira
        return {c: float("nan") for c in rasgos.NOMBRES}


def medida(r: dict, cols: list[str]) -> bool:
    return all(np.isfinite(r.get(c, np.nan)) for c in cols)


def perfil(conjunto: str, piezas: list, guardar: bool = False) -> dict:
    """Perfil de un tipo con las buenas que ajustan (paso 1 y 2 del protocolo del ADR 0017).

    Por cada vista: polaridad por mayoría de la regla de e10, escala que lleva la mediana del radio
    equivalente a 50 px, y cuántas piezas mide la cadena con los rasgos de clasica_v3_6. La vista
    de arriba es la que más mide; si empatan, la de menor excentricidad mediana. La clásica aplica
    si mide el 90 % o más de las piezas en esa vista.
    """
    cols = COLUMNAS["clasica_v3_6"]
    vistas = []
    for k in range(len(piezas[0].vistas)):
        imgs = [p.vistas[k] for p in piezas]
        invertir = sum(golilla_clara(v) for v in imgs) > len(imgs) / 2
        radios = np.array([radio_equivalente(montar(v, invertir, 1.0)) for v in imgs])
        r_med = float(np.nanmedian(radios)) if np.isfinite(radios).any() else float("nan")
        escala = RADIO_OBJETIVO / r_med if np.isfinite(r_med) and r_med > 0 else 1.0
        R = [cadena(montar(v, invertir, escala)) for v in imgs]
        ok = [medida(r, cols) for r in R]
        exc = [r["excentricidad"] for r, o in zip(R, ok) if o]
        vistas.append(dict(vista=k, invertir=bool(invertir), radio_mediano_px=r_med, escala=escala,
                           mide=float(np.mean(ok)), excentricidad_mediana=float(np.median(exc)) if exc else np.inf))
    arriba = min(vistas, key=lambda v: (-v["mide"], v["excentricidad_mediana"]))
    p = dict(conjunto=conjunto, n_ajusta=len(piezas), vistas=len(piezas[0].vistas), vista_arriba=arriba["vista"],
             invertir=arriba["invertir"], escala=arriba["escala"], mide=arriba["mide"],
             clasica_aplica=bool(arriba["mide"] >= MIDE_MIN), por_vista=vistas)
    if guardar:
        TIPOS.mkdir(exist_ok=True)
        (TIPOS / f"{conjunto}.json").write_text(json.dumps(p, indent=1, ensure_ascii=False, default=float) + "\n")
    return p


def leer_perfil(conjunto: str) -> dict:
    return json.loads((TIPOS / f"{conjunto}.json").read_text(encoding="utf-8"))


class Clasica:
    """Mahalanobis² de los rasgos de forma de la vista de arriba, aprendida solo con buenas."""

    def __init__(self, nombre: str, perfil: dict):
        self.nombre, self.cols, self.perfil = nombre, COLUMNAS[nombre], perfil

    def rasgos(self, piezas) -> np.ndarray:
        p = self.perfil
        R = [cadena(montar(x.vistas[p["vista_arriba"]], p["invertir"], p["escala"])) for x in piezas]
        return np.array([[r.get(c, np.nan) for c in self.cols] for r in R], float)

    def ajustar(self, piezas):
        X = self.rasgos(piezas)
        ok = np.isfinite(X).all(1)
        self.sin_medir = int((~ok).sum())
        self.mod = dec.modelos()[D["decision"]].fit(X[ok])
        return self

    def puntajes_vistas(self, piezas) -> np.ndarray:
        X = self.rasgos(piezas)
        ok = np.isfinite(X).all(1)
        return np.where(ok, self.mod.puntaje(X), np.inf)[:, None]  # sin medir: ∞, se retira siempre

    def una_pieza(self, pieza) -> float:
        """Lo que hace la línea con una pieza (para medir la latencia)."""
        return float(self.puntajes_vistas([pieza])[0, 0])
