"""Golillas sintéticas con verdad de terreno exacta.

Cada golilla se dibuja a 4 veces la resolución y luego se reduce: así el borde queda
con precisión de fracción de píxel y se conoce el círculo nominal exacto. La imagen
imita las fotos reales: papel con luz desigual, sombra, brillo metálico, rayas finas,
desenfoque y compresión JPEG. Sirve para medir con cientos de casos lo que las
21 fotos reales no alcanzan a mostrar.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ALTO, ANCHO = 640, 480  # igual que las fotos reales (vertical)
SS = 4  # supermuestreo
DEFECTOS = (
    "deformada",
    "muesca_exterior",
    "corte_radial",
    "borde_interior",
    "mordida_exterior",
    "agujero_descentrado",
)


@dataclass
class Sintetica:
    imagen: np.ndarray  # BGR uint8
    mascara: np.ndarray  # uint8 0/1: material de la golilla
    clase: str  # "buena" | "mala"
    defecto: str  # "ninguno" | uno de DEFECTOS
    cx: float  # círculo nominal (px; el centro del píxel i es la coordenada i)
    cy: float
    r_ext: float
    r_int: float


def _fondo(rng: np.random.Generator) -> np.ndarray:
    h, w = ALTO, ANCHO
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = ((xx - w / 2) * np.cos(ang) + (yy - h / 2) * np.sin(ang)) / (0.5 * np.hypot(h, w))
    manchas = rng.normal(0, 1, (h // 20 + 2, w // 20 + 2)).astype(np.float32)
    manchas = cv2.resize(cv2.GaussianBlur(manchas, (0, 0), 1.2), (w, h), interpolation=cv2.INTER_CUBIC)
    g = (
        rng.uniform(170, 225)
        + rng.uniform(5, 35) * grad
        + rng.uniform(2, 6) * manchas
        + rng.normal(0, 1.2, (h, w)).astype(np.float32)
    )
    tinte = rng.uniform(-6, 6, 3).astype(np.float32)
    return g[..., None] + tinte[None, None, :]


def _rotar(X, Y, a):
    return X * np.cos(a) + Y * np.sin(a), -X * np.sin(a) + Y * np.cos(a)


def _forma(rng, defecto, R, r, P, pcx, pcy):
    """Cobertura (0..1) de la golilla y del disco exterior en un parche P×P."""
    t = (np.arange(P * SS, dtype=np.float32) + 0.5) / SS - 0.5
    X = t[None, :] - pcx
    Y = t[:, None] - pcy
    rho = np.hypot(X, Y)
    if defecto == "deformada":
        q = rng.uniform(0.80, 0.94)
        u, v = _rotar(X, Y, rng.uniform(0, np.pi))
        rho_e = np.hypot(u, v / q)
        ext, hueco = rho_e <= R, rho_e <= r
    else:
        hx = hy = 0.0
        if defecto == "agujero_descentrado":
            d, a = rng.uniform(0.10, 0.22) * (R - r), rng.uniform(0, 2 * np.pi)
            hx, hy = d * np.cos(a), d * np.sin(a)
        ext, hueco = rho <= R, np.hypot(X - hx, Y - hy) <= r
    m = ext & ~hueco
    a = rng.uniform(0, 2 * np.pi)
    u, v = _rotar(X, Y, a)
    anillo = R - r
    if defecto == "muesca_exterior":
        w, d = rng.uniform(3, 9), rng.uniform(0.25, 0.6) * anillo
        m &= ~((np.abs(v) <= w / 2) & (u >= R - d))
    elif defecto == "corte_radial":
        for k in range(int(rng.integers(1, 3))):
            u2, v2 = _rotar(X, Y, a + k * np.pi + rng.uniform(-0.4, 0.4))
            w, L = rng.uniform(1.5, 4), rng.uniform(0.5, 1.05) * anillo
            m &= ~((np.abs(v2) <= w / 2) & (u2 >= r - 1) & (u2 <= r + L))
    elif defecto == "borde_interior":
        disco = np.hypot(u - r, v) <= rng.uniform(0.2, 0.4) * anillo
        m = (m | (disco & ext)) if rng.random() < 0.5 else (m & ~disco)
    elif defecto == "mordida_exterior":
        m &= ~(np.hypot(u - R, v) <= rng.uniform(0.25, 0.5) * anillo)

    def reducir(b):
        return b.reshape(P, SS, P, SS).mean(axis=(1, 3)).astype(np.float32)

    return reducir(m), reducir(ext)


def generar(semilla: int, defecto: str | None = None, p_mala: float = 0.5) -> Sintetica:
    """Genera una golilla. Si `defecto` es None, la clase sale al azar con prob. `p_mala`."""
    rng = np.random.default_rng(semilla)
    if defecto is None:
        defecto = str(rng.choice(DEFECTOS)) if rng.random() < p_mala else "ninguno"
    clase = "buena" if defecto == "ninguno" else "mala"
    R = rng.uniform(46, 58)
    r = R * rng.uniform(0.36, 0.46)
    cx = rng.uniform(0.3, 0.7) * ANCHO
    cy = rng.uniform(0.3, 0.7) * ALTO
    P = int(2 * R + 30)
    x0, y0 = int(cx) - P // 2, int(cy) - P // 2
    pcx, pcy = cx - x0, cy - y0
    cob, disco = _forma(rng, defecto, R, r, P, pcx, pcy)

    alfa = np.zeros((ALTO, ANCHO), np.float32)
    alfa[y0 : y0 + P, x0 : x0 + P] = cob
    ext = np.zeros_like(alfa)
    ext[y0 : y0 + P, x0 : x0 + P] = disco

    fondo = _fondo(rng)
    # sombra: el disco desplazado lejos de la luz, difuminado
    luz = rng.uniform(0, 2 * np.pi)
    elevada = defecto == "deformada"
    dist = rng.uniform(4, 10) if elevada else rng.uniform(1.5, 5)
    M = np.float32([[1, 0, -dist * np.cos(luz)], [0, 1, -dist * np.sin(luz)]])
    sombra = cv2.GaussianBlur(cv2.warpAffine(ext, M, (ANCHO, ALTO)), (0, 0), rng.uniform(2, 5))
    k = rng.uniform(0.2, 0.4) if elevada else rng.uniform(0.06, 0.22)
    fondo *= (1 - k * sombra)[..., None]

    # tono del metal: gradiente de luz, grano, borde brillante y brillo especular si está doblada
    yy, xx = np.mgrid[0:ALTO, 0:ANCHO].astype(np.float32)
    Xn, Yn = xx - cx, yy - cy
    metal = (
        rng.uniform(70, 130)
        + rng.uniform(5, 20) * (Xn * np.cos(luz) + Yn * np.sin(luz)) / R
        + rng.normal(0, 3, (ALTO, ANCHO)).astype(np.float32)
    )
    if rng.random() < 0.6:
        rho = np.hypot(Xn, Yn)
        lado = (Xn * np.cos(luz) + Yn * np.sin(luz)) / np.maximum(rho, 1) > 0.3
        metal += rng.uniform(15, 35) * ((rho > R - 1.8) & lado)
    if elevada:
        b = rng.uniform(0, np.pi)
        d = np.abs(Xn * np.cos(b) + Yn * np.sin(b) - rng.uniform(-0.5, 0.5) * R)
        metal += rng.uniform(30, 80) * np.exp(-((d / rng.uniform(3, 8)) ** 2))
    if rng.random() < 0.35:  # rayas finas: también las tienen las buenas
        capa = np.zeros((ALTO, ANCHO), np.float32)
        for _ in range(int(rng.integers(1, 4))):
            rr, aa = rng.uniform(r + 3, R - 3), rng.uniform(0, 2 * np.pi)
            p0 = np.array([cx + rr * np.cos(aa), cy + rr * np.sin(aa)])
            dd = rng.uniform(0, 2 * np.pi)
            p1 = p0 + rng.uniform(5, 18) * np.array([np.cos(dd), np.sin(dd)])
            cv2.line(capa, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
                     float(rng.choice([-1, 1]) * rng.uniform(10, 30)), 1, cv2.LINE_AA)
        metal += capa * (alfa > 0.99)
    metal = metal[..., None] + rng.uniform(-3, 3, 3).astype(np.float32)[None, None, :]

    img = fondo * (1 - alfa[..., None]) + metal * alfa[..., None]
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 0.8))
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(75, 96))])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return Sintetica(img, (alfa >= 0.5).astype(np.uint8), clase, defecto, float(cx), float(cy),
                     float(R), float(r))


def lote(n: int, semilla: int = 0, p_mala: float = 0.5, defecto: str | None = None) -> list[Sintetica]:
    """`n` golillas reproducibles: la i-ésima usa la semilla `semilla*100000 + i`."""
    return [generar(semilla * 100_000 + i, defecto=defecto, p_mala=p_mala) for i in range(n)]
