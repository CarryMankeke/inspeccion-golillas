"""Proceso 5 · Rasgos: números que describen la forma y el aspecto de la pieza.

Cada rasgo apunta a un tipo de defecto. Todos son relativos al radio, así que no
dependen de la distancia de la cámara (salvo `radio_ext`, que sí la necesita fija).
"""

from __future__ import annotations

import cv2
import numpy as np

from .medicion import Circulo
from .segmentacion import partes

K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

NOMBRES = (
    "circularidad",
    "solidez",
    "relacion_ejes",
    "n_agujeros",
    "desv_ext_max",
    "desv_ext_p95",
    "desv_int_max",
    "desv_int_p95",
    "excentricidad",
    "razon_radios",
    "fourier_ext",
    "fourier_int",
    "relleno_anillo",
    "textura_cv",
    "brillo_frac",
    "hu1",
    "radio_ext",
)


def _fourier(p: np.ndarray, c: Circulo, bins=128):
    """Energía de los armónicos 2..16 del perfil radial r(θ), relativa al radio medio."""
    if len(p) < 20 or not np.isfinite(c.r):
        return np.nan
    th = np.arctan2(p[:, 1] - c.cy, p[:, 0] - c.cx)
    r = np.hypot(p[:, 0] - c.cx, p[:, 1] - c.cy)
    idx = ((th + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    perfil = np.full(bins, np.nan)
    for k in range(bins):
        sel = r[idx == k]
        if sel.size:
            perfil[k] = np.median(sel)
    ok = np.isfinite(perfil)
    if ok.sum() < bins // 2:
        return np.nan
    perfil = np.interp(np.arange(bins), np.nonzero(ok)[0], perfil[ok], period=bins)
    F = np.abs(np.fft.rfft(perfil)) / bins
    return float(np.sqrt((F[2:17] ** 2).sum()) / F[0])


def _desv(p, c):
    if len(p) < 5 or not np.isfinite(c.r):
        return np.nan, np.nan
    d = np.abs(np.hypot(p[:, 0] - c.cx, p[:, 1] - c.cy) - c.r) / c.r
    return float(d.max()), float(np.percentile(d, 95))


def calcular(g: np.ndarray, mascara: np.ndarray, pe, pi, ce: Circulo, ci: Circulo) -> dict:
    relleno, agujero = partes(mascara)
    cs, _ = cv2.findContours(relleno, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    r = dict.fromkeys(NOMBRES, np.nan)
    if not cs:
        return r
    cnt = max(cs, key=cv2.contourArea)
    A, P = cv2.contourArea(cnt), cv2.arcLength(cnt, True)
    r["circularidad"] = 4 * np.pi * A / P**2 if P else np.nan
    r["solidez"] = A / max(cv2.contourArea(cv2.convexHull(cnt)), 1)
    if len(cnt) >= 5:
        (_, _), (a, b), _ = cv2.fitEllipse(cnt)
        r["relacion_ejes"] = min(a, b) / max(a, b)
    hueco = (relleno & (1 - mascara)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(hueco, connectivity=4)
    r["n_agujeros"] = float(sum(st[1:, 4] > 20))
    r["desv_ext_max"], r["desv_ext_p95"] = _desv(pe, ce)
    r["desv_int_max"], r["desv_int_p95"] = _desv(pi, ci)
    if np.isfinite(ce.r) and np.isfinite(ci.r):
        r["excentricidad"] = float(np.hypot(*(ce.c - ci.c)) / ce.r)
        r["razon_radios"] = ci.r / ce.r
        r["relleno_anillo"] = float(mascara.sum() / (np.pi * (ce.r**2 - ci.r**2)))
    elif np.isfinite(ce.r):
        r["razon_radios"] = 0.0
        r["relleno_anillo"] = float(mascara.sum() / (np.pi * ce.r**2))
    r["fourier_ext"] = _fourier(pe, ce)
    r["fourier_int"] = _fourier(pi, ci)
    interior = cv2.erode(mascara, K5, iterations=1) > 0
    if interior.sum() > 50:
        v = g[interior].astype(np.float32)
        r["textura_cv"] = float(v.std() / max(v.mean(), 1))
        papel_zona = (cv2.dilate(relleno, K5, iterations=6) > 0) & ~(cv2.dilate(relleno, K5, 2) > 0)
        papel = float(np.median(g[papel_zona])) if papel_zona.any() else 255.0
        r["brillo_frac"] = float((v > 0.5 * (np.median(v) + papel)).mean())
    hu = cv2.HuMoments(cv2.moments(mascara, binaryImage=True)).ravel()
    r["hu1"] = float(-np.sign(hu[0]) * np.log10(abs(hu[0]) + 1e-12))
    r["radio_ext"] = ce.r
    return r
