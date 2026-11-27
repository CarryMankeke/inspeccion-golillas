"""Proceso 3 · Bordes: puntos del borde exterior y del borde del agujero.

La máscara ya dice dónde está la golilla; aquí se busca el borde con más precisión
para medir. Cada técnica entrega puntos (x, y) dentro de una banda de ±3 px
alrededor del borde de la máscara, separados en exterior e interior.
"""

from __future__ import annotations

import cv2
import numpy as np

from .segmentacion import partes

K7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def bandas(mascara: np.ndarray):
    relleno, agujero = partes(mascara)

    def banda(b):
        return (cv2.dilate(b, K7) > 0) & ~(cv2.erode(b, K7) > 0)

    return relleno, agujero, banda(relleno), banda(agujero)


def _puntos(E, banda):
    y, x = np.nonzero(E & banda)
    return np.column_stack([x, y]).astype(np.float64)


def _nms(mag, gx, gy):
    """Supresión de no máximos: deja el borde de 1 px de ancho."""
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
    q = (((ang + 22.5) // 45).astype(int)) % 4
    H, W = mag.shape
    p = np.pad(mag, 1)
    c = p[1:-1, 1:-1]

    def v(dy, dx):
        return p[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]

    pares = [((0, 1), (0, -1)), ((1, 1), (-1, -1)), ((1, 0), (-1, 0)), ((1, -1), (-1, 1))]
    keep = np.zeros(mag.shape, bool)
    for k, (a, b) in enumerate(pares):
        keep |= (q == k) & (c >= v(*a)) & (c >= v(*b))
    return mag * keep


def _umbral_otsu(valores):
    v = valores[valores > 0]
    if v.size == 0:
        return np.inf
    v8 = np.clip(v / v.max() * 255, 0, 255).astype(np.uint8)
    t = cv2.threshold(v8.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    return t / 255 * v.max()


def _gradiente(g, kx, ky):
    gf = g.astype(np.float32)
    return cv2.filter2D(gf, -1, kx), cv2.filter2D(gf, -1, ky)


KERNELS = {
    "sobel": (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),),
    "scharr": (np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], np.float32),),
    "prewitt": (np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32),),
}


def _por_gradiente(nombre):
    def f(g, be, bi):
        kx = KERNELS[nombre][0]
        gx, gy = _gradiente(g, kx, kx.T.copy())
        n = _nms(np.hypot(gx, gy), gx, gy)
        t = _umbral_otsu(n[be | bi])
        E = n > t
        return _puntos(E, be), _puntos(E, bi)

    return f


def _log(g, be, bi):
    gs = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.5)
    L = cv2.Laplacian(gs, cv2.CV_32F)
    zc = np.zeros(L.shape, bool)
    zc[:, :-1] |= L[:, :-1] * L[:, 1:] < 0
    zc[:-1, :] |= L[:-1, :] * L[1:, :] < 0
    gx, gy = cv2.Sobel(gs, cv2.CV_32F, 1, 0), cv2.Sobel(gs, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    E = zc & (mag > _umbral_otsu(mag[be | bi]))
    return _puntos(E, be), _puntos(E, bi)


def _canny(g, be, bi):
    E = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 40, 120) > 0
    return _puntos(E, be), _puntos(E, bi)


def _umbrales_banda(gs, be, bi):
    """Umbrales de Canny según el contraste del propio borde: la mitad del percentil 95
    de la magnitud del gradiente (L1, la misma que usa Canny) dentro de la banda.
    Corrección (27 nov): antes se usaba el Otsu de la INTENSIDAD, que no tiene las
    unidades del gradiente; en piezas de bajo contraste se perdía el borde del agujero."""
    gf = gs.astype(np.float32)
    mag = np.abs(cv2.Sobel(gf, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gf, cv2.CV_32F, 0, 1))
    v = mag[be | bi]
    alto = 0.5 * float(np.percentile(v, 95)) if v.size else 100.0
    return 0.4 * alto, alto


def _canny_auto(g, be, bi):
    gs = cv2.GaussianBlur(g, (5, 5), 0)
    E = cv2.Canny(gs, *_umbrales_banda(gs, be, bi)) > 0
    return _puntos(E, be), _puntos(E, bi)


def _contorno(relleno, agujero):
    def c(b):
        cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return max(cs, key=len).reshape(-1, 2).astype(np.float64) if cs else np.zeros((0, 2))

    return c(relleno), c(agujero)


def _subpixel_iso(g, mascara, relleno, agujero, be, bi):
    """Isolínea a mitad de camino entre metal y papel (marching squares, subpíxel)."""
    from skimage.measure import find_contours

    gs = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    metal = np.median(gs[cv2.erode(mascara, K7, iterations=2) > 0]) if mascara.any() else 0
    out = []
    for b, banda in ((relleno, be), (agujero, bi)):
        papel_zona = (cv2.dilate(b, K7, iterations=3) > 0) & ~(cv2.dilate(b, K7) > 0)
        if b is agujero:
            papel_zona = cv2.erode(b, K7) > 0
        papel = np.median(gs[papel_zona]) if papel_zona.any() else 255
        nivel = 0.5 * (metal + papel)
        pts = [c for c in find_contours(gs, nivel) if len(c) > 10]
        P = np.vstack(pts)[:, ::-1] if pts else np.zeros((0, 2))
        if len(P):
            yi = np.clip(np.round(P[:, 1]).astype(int), 0, g.shape[0] - 1)
            xi = np.clip(np.round(P[:, 0]).astype(int), 0, g.shape[1] - 1)
            P = P[banda[yi, xi]]
        out.append(P)
    return out[0], out[1]


def _subpixel_grad(g, be, bi):
    """Canny + ajuste de parábola a la magnitud del gradiente (idea de Devernay)."""
    from scipy.ndimage import map_coordinates

    gs = cv2.GaussianBlur(g, (5, 5), 0)
    E = cv2.Canny(gs, *_umbrales_banda(gs, be, bi)) > 0
    gf = gs.astype(np.float32)
    gx, gy = cv2.Sobel(gf, cv2.CV_32F, 1, 0), cv2.Sobel(gf, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    out = []
    for banda in (be, bi):
        y, x = np.nonzero(E & banda)
        if len(x) == 0:
            out.append(np.zeros((0, 2)))
            continue
        nx, ny = gx[y, x], gy[y, x]
        nn = np.maximum(np.hypot(nx, ny), 1e-6)
        nx, ny = nx / nn, ny / nn
        m0 = mag[y, x]
        mm = map_coordinates(mag, [y - ny, x - nx], order=1)
        mp = map_coordinates(mag, [y + ny, x + nx], order=1)
        den = mm - 2 * m0 + mp
        d = np.where(np.abs(den) > 1e-6, 0.5 * (mm - mp) / den, 0.0)
        d = np.clip(d, -0.5, 0.5)
        out.append(np.column_stack([x + d * nx, y + d * ny]))
    return out[0], out[1]


OPCIONES = (
    "contorno",
    "sobel",
    "scharr",
    "prewitt",
    "log",
    "canny",
    "canny_auto",
    "subpixel_iso",
    "subpixel_grad",
)


def bordes(g: np.ndarray, mascara: np.ndarray, metodo: str):
    """Devuelve (puntos_exteriores, puntos_interiores) como arreglos N×2 de (x, y)."""
    relleno, agujero, be, bi = bandas(mascara)
    if metodo == "contorno":
        return _contorno(relleno, agujero)
    if metodo in KERNELS:
        return _por_gradiente(metodo)(g, be, bi)
    if metodo == "log":
        return _log(g, be, bi)
    if metodo == "canny":
        return _canny(g, be, bi)
    if metodo == "canny_auto":
        return _canny_auto(g, be, bi)
    if metodo == "subpixel_iso":
        return _subpixel_iso(g, mascara, relleno, agujero, be, bi)
    if metodo == "subpixel_grad":
        return _subpixel_grad(g, be, bi)
    raise ValueError(metodo)
