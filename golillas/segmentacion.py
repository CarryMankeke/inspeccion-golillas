"""Proceso 2 · Segmentación: qué píxeles son golilla y cuáles papel.

Todas las técnicas reciben la imagen preprocesada (uint8) y devuelven una máscara
0/1 del material. Después pasan por la misma limpieza (`limpiar`), para que la
comparación mida la técnica y no la limpieza.
"""

from __future__ import annotations

import cv2
import numpy as np

K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def limpiar(fg: np.ndarray) -> np.ndarray:
    """Se queda con la pieza más grande que no toca el borde y tapa los huecos chicos."""
    fg = (fg > 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return np.zeros_like(fg)
    h, w = fg.shape
    x, y, ww, hh, a = (st[1:, i] for i in range(5))
    toca = (x == 0) | (y == 0) | (x + ww >= w) | (y + hh >= h)
    i = 1 + int(np.argmax(a * np.where(toca, 0.05, 1.0)))
    m = (lab == i).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(1 - m, connectivity=4)
    area = int(m.sum())
    for j in range(1, n2):
        x2, y2, w2, h2, a2 = st2[j]
        if x2 == 0 or y2 == 0 or x2 + w2 >= w or y2 + h2 >= h:
            continue
        if a2 < 0.05 * area:
            m[lab2 == j] = 1
    return m


def _otsu(g):
    return cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def _triangulo(g):
    return cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)[1]


def _adapt_media(g):
    return cv2.adaptiveThreshold(g, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 151, 12)


def _adapt_gauss(g):
    return cv2.adaptiveThreshold(g, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 151, 12)


def _bradley(g):
    """Bradley-Roth, la idea detrás de imbinarize('adaptive') del MATLAB original."""
    k = 2 * (max(g.shape) // 16) + 1
    media = cv2.boxFilter(g.astype(np.float32), -1, (k, k), borderType=cv2.BORDER_REPLICATE)
    return (g.astype(np.float32) < media * 0.85).astype(np.uint8)


def _sauvola(g):
    from skimage.filters import threshold_sauvola

    return (g < threshold_sauvola(g, window_size=151, k=0.2)).astype(np.uint8)


def _kmeans(g):
    z = g.reshape(-1, 1).astype(np.float32)[::7]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, c = cv2.kmeans(z, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return (g < float(c.mean())).astype(np.uint8)


def _canny_relleno(g):
    gs = cv2.GaussianBlur(g, (5, 5), 0)
    alto = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    e = cv2.morphologyEx(cv2.Canny(gs, 0.25 * alto, 0.5 * alto), cv2.MORPH_CLOSE, K5)
    n, lab, st, cen = cv2.connectedComponentsWithStats(1 - (e > 0).astype(np.uint8), connectivity=4)
    h, w = g.shape
    fuera = np.zeros(n, bool)
    for j in range(1, n):
        x, y, ww, hh, _ = st[j]
        fuera[j] = x == 0 or y == 0 or x + ww >= w or y + hh >= h
    relleno = limpiar((~fuera[lab]).astype(np.uint8) & (lab >= 0))
    if relleno.sum() == 0:
        return relleno
    m = cv2.moments(relleno)
    c = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    area = relleno.sum()
    mejor, dmin = None, np.inf
    for j in range(1, n):
        if fuera[j] or not (0.05 * area < st[j, 4] < 0.6 * area):
            continue
        d = np.hypot(*(cen[j] - c))
        if d < dmin:
            mejor, dmin = j, d
    if mejor is not None:
        relleno[cv2.dilate((lab == mejor).astype(np.uint8), K5) > 0] = 0
    return relleno


def _semilla(g):
    return limpiar(_triangulo(cv2.GaussianBlur(g, (5, 5), 0)))


def _grabcut(g):
    t = _semilla(g)
    if t.sum() == 0:
        return t
    x, y, w, h = cv2.boundingRect(t)
    x0, y0 = max(x - 25, 0), max(y - 25, 0)
    x1, y1 = min(x + w + 25, g.shape[1]), min(y + h + 25, g.shape[0])
    roi, tt = g[y0:y1, x0:x1], t[y0:y1, x0:x1]
    mask = np.full(roi.shape, cv2.GC_PR_BGD, np.uint8)
    mask[tt > 0] = cv2.GC_PR_FGD
    mask[cv2.dilate(tt, K5, iterations=3) == 0] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.merge([roi] * 3), mask, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    out = np.zeros_like(g, np.uint8)
    out[y0:y1, x0:x1] = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    return out


def _watershed(g):
    t = _semilla(g)
    if t.sum() == 0:
        return t
    marcas = np.zeros(g.shape, np.int32)
    marcas[cv2.dilate(t, K5, iterations=3) == 0] = 1
    marcas[cv2.erode(t, K5, iterations=2) > 0] = 2
    cv2.watershed(cv2.merge([g] * 3), marcas)
    return (marcas == 2).astype(np.uint8)


def _chan_vese(g):
    from skimage.segmentation import morphological_chan_vese

    s = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    fg = morphological_chan_vese(s, num_iter=60, init_level_set="checkerboard", smoothing=1).astype(bool)
    if s[fg].mean() > s[~fg].mean():
        fg = ~fg
    return cv2.resize(fg.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)


OPCIONES = {
    "otsu": _otsu,
    "triangulo": _triangulo,
    "adapt_media": _adapt_media,
    "adapt_gauss": _adapt_gauss,
    "bradley": _bradley,
    "sauvola": _sauvola,
    "kmeans": _kmeans,
    "canny_relleno": _canny_relleno,
    "grabcut": _grabcut,
    "watershed": _watershed,
    "chan_vese": _chan_vese,
}


def segmentar(g: np.ndarray, metodo: str) -> np.ndarray:
    return limpiar(OPCIONES[metodo](g))


def partes(m: np.ndarray):
    """Separa la máscara del anillo en (relleno exterior, agujero central)."""
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    relleno = np.zeros_like(m)
    if not cnts:
        return relleno, relleno.copy()
    cv2.drawContours(relleno, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
    hueco = (relleno & (1 - m)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(hueco, connectivity=4)
    agujero = np.zeros_like(m)
    if n > 1:
        agujero[lab == 1 + int(np.argmax(st[1:, 4]))] = 1
    return relleno, agujero


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0
