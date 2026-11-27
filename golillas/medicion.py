"""Proceso 4 · Medición: ajustar los círculos exterior e interior.

Un buen ajuste mide el círculo nominal aunque la pieza tenga un defecto local:
así lo que se sale del círculo (una muesca, un corte) queda a la vista como residuo.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from .segmentacion import partes


@dataclass
class Circulo:
    cx: float
    cy: float
    r: float

    @property
    def c(self):
        return np.array([self.cx, self.cy])


NAN = Circulo(np.nan, np.nan, np.nan)


def kasa(p):
    x, y = p[:, 0], p[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    a, b, c = np.linalg.lstsq(A, x**2 + y**2, rcond=None)[0]
    cx, cy = a / 2, b / 2
    return Circulo(cx, cy, float(np.sqrt(max(c + cx**2 + cy**2, 0))))


def taubin(p):
    """Ajuste algebraico de Taubin (menos sesgo que Kåsa en arcos)."""
    m = p.mean(axis=0)
    u, v = p[:, 0] - m[0], p[:, 1] - m[1]
    z = u**2 + v**2
    zm = z.mean()
    z0 = (z - zm) / (2 * np.sqrt(zm))
    Z = np.column_stack([z0, u, v])
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    A = Vt[2]
    A0 = A[0] / (2 * np.sqrt(zm))
    A = np.array([A0, A[1], A[2], -zm * A0])
    cx, cy = -A[1] / A[0] / 2 + m[0], -A[2] / A[0] / 2 + m[1]
    r = np.sqrt(A[1] ** 2 + A[2] ** 2 - 4 * A[0] * A[3]) / abs(A[0]) / 2
    return Circulo(cx, cy, float(r))


def _geom(p, ini, perdida="linear"):
    def res(q):
        return np.hypot(p[:, 0] - q[0], p[:, 1] - q[1]) - q[2]

    q = least_squares(res, [ini.cx, ini.cy, ini.r], loss=perdida, f_scale=1.0,
                      method="lm" if perdida == "linear" else "trf").x
    return Circulo(*map(float, q))


def geometrico(p):
    return _geom(p, kasa(p))


def robusto(p):
    """Mínimos cuadrados geométricos con pérdida soft-L1: los puntos raros pesan menos."""
    return _geom(p, kasa(p), "soft_l1")


def ransac(p, iteraciones=300, tol=1.5, semilla=0):
    """RANSAC vectorizado: 300 círculos por 3 puntos al azar, gana el que junta más
    puntos a menos de `tol` px; luego se refina con mínimos cuadrados sobre esos puntos."""
    rng = np.random.default_rng(semilla)
    a, b, c = (p[i] for i in rng.integers(0, len(p), size=(3, iteraciones)))
    A11, A12 = 2 * (b[:, 0] - a[:, 0]), 2 * (b[:, 1] - a[:, 1])
    A21, A22 = 2 * (c[:, 0] - a[:, 0]), 2 * (c[:, 1] - a[:, 1])
    B1 = (b**2).sum(1) - (a**2).sum(1)
    B2 = (c**2).sum(1) - (a**2).sum(1)
    det = A11 * A22 - A12 * A21
    ok = np.abs(det) > 1e-9
    det = np.where(ok, det, 1.0)
    cx, cy = (B1 * A22 - A12 * B2) / det, (A11 * B2 - B1 * A21) / det
    r = np.hypot(a[:, 0] - cx, a[:, 1] - cy)
    d = np.abs(np.hypot(p[None, :, 0] - cx[:, None], p[None, :, 1] - cy[:, None]) - r[:, None]) < tol
    n = np.where(ok, d.sum(1), -1)
    k = int(np.argmax(n))
    mejor = Circulo(float(cx[k]), float(cy[k]), float(r[k]))
    dentro = d[k]
    return _geom(p[dentro], mejor) if dentro.sum() >= 5 else mejor


def elipse(p):
    (cx, cy), (a, b), _ = cv2.fitEllipse(p.astype(np.float32))
    return Circulo(cx, cy, (a + b) / 4)


def min_circulo(p):
    (cx, cy), r = cv2.minEnclosingCircle(p.astype(np.float32))
    return Circulo(cx, cy, r)


AJUSTES = {
    "kasa": kasa,
    "taubin": taubin,
    "geometrico": geometrico,
    "robusto": robusto,
    "ransac": ransac,
    "elipse": elipse,
    "min_circulo": min_circulo,
}


def _momentos(b):
    m = cv2.moments(b, binaryImage=True)
    if m["m00"] == 0:
        return NAN
    return Circulo(m["m10"] / m["m00"], m["m01"] / m["m00"], float(np.sqrt(m["m00"] / np.pi)))


def _hough(g, guia, alt=False):
    if not np.isfinite(guia.r):
        return NAN
    gs = cv2.GaussianBlur(g, (5, 5), 0)
    lo, hi = int(0.85 * guia.r), int(np.ceil(1.15 * guia.r)) + 1
    if alt:
        cs = cv2.HoughCircles(gs, cv2.HOUGH_GRADIENT_ALT, 1.5, 5, param1=150, param2=0.6,
                              minRadius=lo, maxRadius=hi)
    else:
        cs = cv2.HoughCircles(gs, cv2.HOUGH_GRADIENT, 1, 5, param1=100, param2=15,
                              minRadius=lo, maxRadius=hi)
    if cs is None:
        return NAN
    cs = cs.reshape(-1, 3)
    d = np.hypot(cs[:, 0] - guia.cx, cs[:, 1] - guia.cy) + np.abs(cs[:, 2] - guia.r)
    return Circulo(*map(float, cs[int(np.argmin(d))]))


OPCIONES = tuple(AJUSTES) + ("momentos", "hough", "hough_alt")


def medir(g, mascara, pe, pi, metodo):
    """Devuelve (círculo exterior, círculo interior)."""
    if metodo in ("momentos", "hough", "hough_alt"):
        relleno, agujero = partes(mascara)
        ce, ci = _momentos(relleno), _momentos(agujero)
        if metodo == "momentos":
            return ce, ci
        return _hough(g, ce, metodo == "hough_alt"), _hough(g, ci, metodo == "hough_alt")
    f = AJUSTES[metodo]
    return (f(pe) if len(pe) >= 5 else NAN), (f(pi) if len(pi) >= 5 else NAN)
