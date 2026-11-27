"""Proceso 1 · Preprocesado: de la foto en color a una imagen gris pareja.

El problema real es la luz: el papel no es blanco parejo y la golilla ocupa
solo ~3 % de la foto, así que un umbral global se confunde con las sombras.
"""

from __future__ import annotations

import cv2
import numpy as np


def gris(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def fondo_estimado(g: np.ndarray) -> np.ndarray:
    """Papel sin golilla: un cierre morfológico grande borra lo oscuro y deja el papel."""
    chica = cv2.resize(g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))  # ~68 px a escala real
    bg = cv2.morphologyEx(chica, cv2.MORPH_CLOSE, k)
    bg = cv2.GaussianBlur(cv2.medianBlur(bg, 5), (0, 0), 2)
    return cv2.resize(bg, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_CUBIC)


def _fondo_div(g):
    bg = fondo_estimado(g).astype(np.float32)
    return np.clip(g.astype(np.float32) / np.maximum(bg, 1) * 200, 0, 255).astype(np.uint8)


def _fondo_resta(g):
    bg = fondo_estimado(g).astype(np.float32)
    return np.clip(200 - (bg - g.astype(np.float32)), 0, 255).astype(np.uint8)


OPCIONES = {
    "ninguno": lambda g: g,
    "gauss": lambda g: cv2.GaussianBlur(g, (5, 5), 0),
    "mediana": lambda g: cv2.medianBlur(g, 5),
    "bilateral": lambda g: cv2.bilateralFilter(g, 7, 30, 7),
    "clahe": lambda g: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g),
    "fondo_resta": _fondo_resta,
    "fondo_div": _fondo_div,
    "fondo_div+gauss": lambda g: cv2.GaussianBlur(_fondo_div(g), (5, 5), 0),
}


def preprocesar(img: np.ndarray, metodo: str) -> np.ndarray:
    return OPCIONES[metodo](gris(img))
