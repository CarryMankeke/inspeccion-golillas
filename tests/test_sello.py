"""Pruebas del sello de los datasets públicos (ADR 0017).

Los archivos falsos no son imágenes: si algo intentara decodificarlos, la prueba fallaría.
"""

import hashlib

import numpy as np
import pytest

from golillas import sello

ESPERADO = dict(entrenamiento=10, buenas=4, malas=3)


def _manta(raiz, tipo="red_washer"):
    base = raiz / "mechanics" / tipo
    for carpeta, nombres in (("train/good", range(10)), ("test/good", range(10, 14)),
                             ("test/scratch", range(14, 16)), ("test/stain", range(16, 17))):
        (base / carpeta).mkdir(parents=True, exist_ok=True)
        for i in nombres:
            (base / carpeta / f"mechanics_{tipo}_{i:05d}.png").write_bytes(f"no es imagen {i}".encode())
            if "good" not in carpeta:
                m = base / "ground_truth" / carpeta.split("/")[1]
                m.mkdir(parents=True, exist_ok=True)
                (m / f"mechanics_{tipo}_{i:05d}.png").write_bytes(f"mascara {i}".encode())
    return raiz


def test_mitades_es_la_permutacion_de_la_semilla_si_cada_pieza_es_su_origen():
    nombres = [f"p{i:03d}" for i in range(578)]
    grupos = sello.mitades(nombres)
    ajustan = {nombres[i] for i in np.random.default_rng(0).permutation(578)[:289]}
    assert {n for n, g in zip(nombres, grupos) if g == "ajusta"} == ajustan
    assert grupos.count("calibra") == 289


def test_mitades_deja_juntas_las_copias_de_una_misma_foto():
    origenes = [f"foto{i // 3}" for i in range(30)]  # 10 fotos con 3 copias aumentadas cada una
    grupos = sello.mitades(origenes)
    por_origen = {}
    for o, g in zip(origenes, grupos):
        por_origen.setdefault(o, set()).add(g)
    assert all(len(g) == 1 for g in por_origen.values())
    assert 12 <= grupos.count("ajusta") <= 18


def test_el_sello_de_manta_parte_las_buenas_y_guarda_las_huellas(tmp_path):
    raiz = _manta(tmp_path / "datos")
    n = sello.sellar_manta("red_washer", raiz, tmp_path / "sellos", esperado=ESPERADO)
    assert n == {"ajusta": 5, "calibra": 5, "prueba": 7}
    prueba = sello.leer("manta_red_washer", "prueba", tmp_path / "sellos")
    assert sorted(r["defecto"] for r in prueba if r["clase"] == "mala") == ["scratch", "scratch", "stain"]
    for r in prueba:
        assert r["sha256"] == hashlib.sha256((raiz / r["archivo"]).read_bytes()).hexdigest()
        assert bool(r["mascara"]) == (r["clase"] == "mala")
    ajusta = sello.leer("manta_red_washer", "ajusta", tmp_path / "sellos")
    assert all(r["clase"] == "buena" and "/train/good/" in r["archivo"] for r in ajusta)
    assert sello.verificar("manta_red_washer", raiz, ["mechanics/red_washer"], tmp_path / "sellos") == n


def test_no_se_sella_dos_veces_ni_con_conteos_distintos(tmp_path):
    raiz = _manta(tmp_path / "datos")
    with pytest.raises(ValueError, match="no calzan"):
        sello.sellar_manta("red_washer", raiz, tmp_path / "sellos", esperado=dict(ESPERADO, malas=4))
    sello.sellar_manta("red_washer", raiz, tmp_path / "sellos", esperado=ESPERADO)
    with pytest.raises(FileExistsError):
        sello.sellar_manta("red_washer", raiz, tmp_path / "sellos", esperado=ESPERADO)


def test_cargar_y_verificar_detectan_cambios(tmp_path):
    raiz = _manta(tmp_path / "datos")
    sello.sellar_manta("red_washer", raiz, tmp_path / "sellos", esperado=ESPERADO)
    extra = raiz / "mechanics" / "red_washer" / "test" / "good" / "intrusa.png"
    extra.write_bytes(b"x")
    with pytest.raises(ValueError, match="sobran 1"):
        sello.verificar("manta_red_washer", raiz, ["mechanics/red_washer"], tmp_path / "sellos")
    extra.unlink()
    r = sello.leer("manta_red_washer", "calibra", tmp_path / "sellos")[0]
    (raiz / r["archivo"]).write_bytes(b"cambiada")
    with pytest.raises(ValueError, match="no calza"):
        sello.cargar("manta_red_washer", "calibra", raiz, tmp_path / "sellos")


def test_una_mala_sin_mascara_detiene_el_sello(tmp_path):
    raiz = _manta(tmp_path / "datos")
    next((raiz / "mechanics" / "red_washer" / "ground_truth" / "stain").iterdir()).unlink()
    with pytest.raises(FileNotFoundError, match="máscara"):
        sello.piezas_manta("red_washer", raiz)
