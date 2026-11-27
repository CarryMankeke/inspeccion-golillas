import numpy as np

from golillas import medicion


def circulo(n=300, cx=120.3, cy=80.7, r=50.2, ruido=0.2, semilla=0):
    rng = np.random.default_rng(semilla)
    a = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)]) + rng.normal(0, ruido, (n, 2))


def test_ajustes_recuperan_el_circulo():
    p = circulo()
    for f in (medicion.kasa, medicion.taubin, medicion.geometrico, medicion.robusto, medicion.ransac):
        c = f(p)
        assert abs(c.r - 50.2) < 0.05, f.__name__
        assert np.hypot(c.cx - 120.3, c.cy - 80.7) < 0.05, f.__name__


def test_ransac_ignora_un_defecto():
    p = circulo()
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 0.6, 120)  # 30 % de puntos metidos 6 px hacia adentro: una "mordida"
    defecto = np.column_stack([120.3 + 44.2 * np.cos(a), 80.7 + 44.2 * np.sin(a)])
    q = np.vstack([p, defecto])
    assert abs(medicion.ransac(q).r - 50.2) < 0.1
    assert abs(medicion.kasa(q).r - 50.2) > 0.5  # el ajuste simple sí se deja arrastrar
