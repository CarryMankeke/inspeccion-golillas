"""Pruebas de los cargadores del ADR 0017 con un MANTA falso de imágenes de 1.280 × 256."""

import cv2
import numpy as np
import pytest

from golillas import conjuntos, sello

ESPERADO = dict(entrenamiento=6, buenas=2, malas=2)


def _manta(raiz, tipo="red_washer"):
    base = raiz / "mechanics" / tipo
    for carpeta, nombres in (("train/good", range(6)), ("test/good", range(6, 8)), ("test/scratch", range(8, 10))):
        (base / carpeta).mkdir(parents=True, exist_ok=True)
        for i in nombres:
            img = np.zeros((256, 1280, 3), np.uint8)
            for k in range(5):
                img[:, 256 * k : 256 * (k + 1)] = 40 * k + i  # cada vista con su propio gris
            cv2.imwrite(str(base / carpeta / f"mechanics_{tipo}_{i:05d}.png"), img)
            if "good" not in carpeta:
                (base / "ground_truth" / "scratch").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(base / "ground_truth" / "scratch" / f"mechanics_{tipo}_{i:05d}.png"),
                            np.zeros((256, 1280), np.uint8))
    return raiz


def test_manta_entrega_5_vistas_por_pieza_y_respeta_el_sello(tmp_path):
    raiz = _manta(tmp_path / "datos")
    sello.sellar_manta("red_washer", raiz, tmp_path / "sellos", esperado=ESPERADO)
    aj = conjuntos.sellado("manta_red_washer", "ajusta", raiz=raiz, destino=tmp_path / "sellos")
    pr = conjuntos.sellado("manta_red_washer", "prueba", raiz=raiz, destino=tmp_path / "sellos")
    assert len(aj) == 3 and all(len(p.vistas) == 5 and p.vistas[0].shape == (256, 256, 3) for p in aj)
    assert [int(p.vistas[k][0, 0, 0]) - int(p.vistas[0][0, 0, 0]) for p in aj[:1] for k in range(5)] == [0, 40, 80,
                                                                                                         120, 160]
    assert sorted(p.mala for p in pr) == [False, False, True, True]
    assert {p.defecto for p in pr if p.mala} == {"scratch"}


def test_vistas_manta_rechaza_otra_forma():
    with pytest.raises(ValueError, match="5 vistas"):
        conjuntos.vistas_manta(np.zeros((256, 1000, 3), np.uint8))


def test_leer_reduce_las_fotos_grandes(tmp_path):
    cv2.imwrite(str(tmp_path / "g.jpg"), np.zeros((3036, 4024, 3), np.uint8))
    assert max(conjuntos.leer(tmp_path / "g.jpg").shape[:2]) == conjuntos.LADO_MAX


def test_submuestra_es_fija_y_conserva_el_orden():
    a = conjuntos.submuestra(list(range(100)), 10, semilla=3)
    assert a == sorted(a) and a == conjuntos.submuestra(list(range(100)), 10, semilla=3) and len(a) == 10
