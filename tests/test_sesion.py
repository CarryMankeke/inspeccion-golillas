"""Pruebas de la lista de tomas y del sellado de la v5 (ADR 0016)."""

import csv
import hashlib
from collections import Counter

import pytest
from PIL import Image

from golillas import sesion


def test_la_lista_tiene_los_grupos_del_adr():
    lista = sesion.lista_tomas()
    cuenta = Counter((t["grupo"], t["clase"], t["defecto"]) for t in lista)
    assert len(lista) == 180
    assert cuenta[("ajusta", "buena", "ninguno")] == 20
    assert cuenta[("calibra", "buena", "ninguno")] == 100
    assert cuenta[("prueba", "buena", "ninguno")] == 30
    assert sum(n for (g, c, _), n in cuenta.items() if g == "prueba" and c == "mala") == 30
    assert sum(cuenta[("prueba", "mala", d)] for d in sesion.EXIGIDAS) == 18
    assert [t["toma"] for t in lista] == list(range(1, 181))
    assert Counter(t["bloque"] for t in lista) == {b: 10 for b in range(1, 19)}


def test_la_lista_guardada_y_la_guia_son_las_de_la_semilla():
    lista = sesion.lista_tomas()
    assert sesion.leer_lista() == lista
    assert sesion.tabla_markdown(lista) in sesion.GUIA.read_text(encoding="utf-8")


def _sesion(carpeta, n=198, dia="2026:12:05", tamano=(30, 40)):
    """n fotos chicas con hora de cámara creciente; los nombres dan la vuelta (IMG_9950… → IMG_0000…)."""
    carpeta.mkdir()
    for i in range(n):
        ex = Image.Exif()
        ex[sesion.IFD_EXIF] = {sesion.FECHA_ORIGINAL: f"{dia} 15:{i // 60:02d}:{i % 60:02d}",
                               sesion.SUBSEGUNDOS: "250"}
        Image.new("RGB", tamano, (i, 200, 200)).save(carpeta / f"IMG_{(9950 + i) % 10000:04d}.JPG", exif=ex)
    return carpeta


def _leer(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_el_sello_calza_las_fotos_con_la_lista_por_la_hora_de_la_camara(tmp_path):
    fotos = _sesion(tmp_path / "fotos")
    n = sesion.sellar(fotos, tmp_path / "sello", sesion.lista_tomas(), hoja=tmp_path / "hoja.jpg")
    assert n == {"ajusta": 20, "calibra": 100, "prueba": 60, "separadores": 18}
    orden = sorted(fotos.iterdir(), key=lambda p: (int(p.stem[4:]) - 9950) % 10000)  # orden de la hora
    seps = _leer(tmp_path / "sello" / "sello_separadores.csv")
    assert [s["archivo"] for s in seps] == [orden[11 * k].name for k in range(18)]
    prueba = _leer(tmp_path / "sello" / "sello_prueba.csv")
    for r in prueba:
        t = int(r["toma"])
        foto = orden[(t - 1) // 10 * 11 + (t - 1) % 10 + 1]
        assert r["archivo"] == foto.name
        assert r["sha256"] == hashlib.sha256(foto.read_bytes()).hexdigest()
    assert prueba[0]["fecha"].endswith(".250")
    assert (tmp_path / "hoja.jpg").exists()
    assert len(sesion.cargar_grupo("prueba", fotos, tmp_path / "sello")) == 60


def test_el_sello_se_detiene_si_falta_una_foto(tmp_path):
    with pytest.raises(ValueError, match="se esperaban 198"):
        sesion.sellar(_sesion(tmp_path / "fotos", n=197), tmp_path / "sello", sesion.lista_tomas(), hoja=None)


def test_el_sello_se_detiene_con_fotos_de_dos_dias(tmp_path):
    fotos = _sesion(tmp_path / "fotos")
    ex = Image.Exif()
    ex[sesion.IFD_EXIF] = {sesion.FECHA_ORIGINAL: "2026:12:06 09:00:00"}
    otra = sorted(fotos.iterdir())[0]
    Image.new("RGB", (30, 40)).save(otra, exif=ex)
    with pytest.raises(ValueError, match="más de un día"):
        sesion.sellar(fotos, tmp_path / "sello", sesion.lista_tomas(), hoja=None)


def test_el_sello_no_acepta_heic_ni_se_repite(tmp_path):
    fotos = _sesion(tmp_path / "fotos")
    (fotos / "IMG_0100.HEIC").write_bytes(b"x")
    with pytest.raises(ValueError, match="no son JPG"):
        sesion.sellar(fotos, tmp_path / "sello", sesion.lista_tomas(), hoja=None)
    (fotos / "IMG_0100.HEIC").unlink()
    sesion.sellar(fotos, tmp_path / "sello", sesion.lista_tomas(), hoja=None)
    with pytest.raises(FileExistsError):
        sesion.sellar(fotos, tmp_path / "sello", sesion.lista_tomas(), hoja=None)


def test_cargar_grupo_se_detiene_si_una_foto_cambio_despues_del_sello(tmp_path):
    fotos = _sesion(tmp_path / "fotos")
    sesion.sellar(fotos, tmp_path / "sello", sesion.lista_tomas(), hoja=None)
    r = _leer(tmp_path / "sello" / "sello_ajusta.csv")[0]
    (fotos / r["archivo"]).write_bytes(b"otra cosa")
    with pytest.raises(ValueError, match="no calza"):
        sesion.cargar_grupo("ajusta", fotos, tmp_path / "sello")


def test_reducir_lleva_el_lado_corto_a_480():
    import numpy as np

    assert sesion.reducir(np.zeros((4032, 3024, 3), np.uint8)).shape == (640, 480, 3)
    assert sesion.reducir(np.zeros((640, 480, 3), np.uint8)).shape == (640, 480, 3)
