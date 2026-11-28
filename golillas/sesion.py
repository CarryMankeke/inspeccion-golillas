"""Sesión de fotos propias de la v5 (ADR 0016): la lista de tomas y el sellado.

La lista se arma al azar con semilla fija ANTES de tomar ninguna foto. Dice qué golilla va en
cada toma y a qué grupo pertenece: ajusta, calibra o prueba. Camilo solo la sigue, y como
buenas y malas van intercaladas, un cambio de luz durante la tarde cae parejo en todos los grupos.

El sellado se corre después de la sesión y ANTES de que cualquier script mire los píxeles.
Ordena las fotos por la hora de la cámara, las calza con la lista (cada fila de 10 tomas parte
con una foto a un papelito separador) y guarda, por grupo, un CSV con la huella SHA-256 de
cada foto. Esos CSV van en un commit propio. Aquí solo se leen metadatos (EXIF) y bytes. Las
únicas imágenes que se decodifican son las de los papelitos separadores, para la hoja de control.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "propio_v5"
LISTA_CSV = DATOS / "lista_tomas.csv"
GUIA = RAIZ / "docs" / "sesion-fotos-v5.md"

SEMILLA = 0
BUENAS = {"ajusta": 20, "calibra": 100, "prueba": 30}
MALAS = {"cortada": 6, "doblada": 6, "muesca_exterior": 3, "muesca_interior": 3, "otra_medida": 6, "rayada": 6}
EXIGIDAS = ("cortada", "doblada", "muesca_exterior", "muesca_interior")  # requisito 1 del ADR 0016
POR_FILA = 10
CAMPOS = ("toma", "bloque", "grupo", "clase", "defecto")

# Fijados en el ADR 0016 antes de tomar las fotos
LADO_CORTO = 480  # px: cada foto se reduce a 480 × 640, como las 21 fotos originales
RADIO_EQ = (40.0, 60.0)  # px: rango aceptado de la mediana del radio en las 20 que ajustan
MIDE_MIN = 18  # de las 20 que ajustan, cuántas tiene que medir la cadena clásica
ALFA = 0.03  # α/2 = 1,5 % por parte en la fusión

ETIQUETA = {"ninguno": "buena", "cortada": "**CORTADA**", "doblada": "**DOBLADA**",
            "muesca_exterior": "**MUESCA AFUERA**", "muesca_interior": "**MUESCA AGUJERO**",
            "otra_medida": "**OTRA MEDIDA**", "rayada": "**RAYADA ↑**"}
INICIO, FIN = "<!-- lista:inicio -->", "<!-- lista:fin -->"
EXTENSIONES = {".jpg", ".jpeg"}
IFD_EXIF, FECHA_ORIGINAL, SUBSEGUNDOS, FECHA = 0x8769, 0x9003, 0x9291, 0x0132


# ---------------------------------------------------------------- lista de tomas
def lista_tomas(semilla: int = SEMILLA) -> list[dict]:
    """Las 180 tomas en orden al azar: 150 buenas (20 ajustan, 100 calibran, 30 de prueba) y 30 malas de prueba."""
    piezas = [(g, "buena", "ninguno") for g, n in BUENAS.items() for _ in range(n)]
    piezas += [("prueba", "mala", d) for d, n in MALAS.items() for _ in range(n)]
    orden = np.random.default_rng(semilla).permutation(len(piezas))
    return [dict(toma=t, bloque=(t - 1) // POR_FILA + 1, grupo=g, clase=c, defecto=d)
            for t, (g, c, d) in enumerate((piezas[i] for i in orden), start=1)]


def tabla_markdown(lista: list[dict]) -> str:
    """Una fila por bloque de 10 tomas, para imprimir y marcar con lápiz."""
    filas = ["| Fila | " + " | ".join(str(i) for i in range(1, POR_FILA + 1)) + " |", "|---" * (POR_FILA + 1) + "|"]
    for b in sorted({t["bloque"] for t in lista}):
        celdas = " | ".join(ETIQUETA[t["defecto"]] for t in lista if t["bloque"] == b)
        filas.append(f"| **B{b:02d}** | {celdas} |")
    return "\n".join(filas)


def escribir_lista(lista: list[dict] | None = None, csv_destino: Path = LISTA_CSV, guia: Path = GUIA) -> None:
    """Guarda la lista en CSV y la copia en la guía de Camilo, entre las marcas de inicio y fin."""
    lista = lista or lista_tomas()
    csv_destino.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_destino, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, CAMPOS)
        w.writeheader()
        w.writerows(lista)
    texto = guia.read_text(encoding="utf-8")
    a, b = texto.index(INICIO) + len(INICIO), texto.index(FIN)
    guia.write_text(texto[:a] + "\n" + tabla_markdown(lista) + "\n" + texto[b:], encoding="utf-8")


def leer_lista(ruta: Path = LISTA_CSV) -> list[dict]:
    with open(ruta, newline="", encoding="utf-8") as f:
        return [dict(r, toma=int(r["toma"]), bloque=int(r["bloque"])) for r in csv.DictReader(f)]


# ---------------------------------------------------------------- sellado
def huella(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def fecha_camara(ruta: Path) -> tuple[str, float, tuple[int, int]]:
    """(fecha y hora de la cámara, subsegundos, tamaño). Lee solo la cabecera y el EXIF, no los píxeles."""
    from PIL import Image

    with Image.open(ruta) as im:
        ex = im.getexif()
        sub = ex.get_ifd(IFD_EXIF)
        fecha = sub.get(FECHA_ORIGINAL) or ex.get(FECHA)
        ss = str(sub.get(SUBSEGUNDOS, "") or "").strip()
        tam = im.size
    if not fecha:
        raise ValueError(f"{ruta.name}: no trae la hora de la cámara (EXIF)")
    return str(fecha).strip(), float(f"0.{ss}") if ss.isdigit() else 0.0, tam


def sellar(carpeta: Path, destino: Path = DATOS, lista: list[dict] | None = None,
           hoja: Path | None = RAIZ / "cache" / "sello_separadores.jpg") -> dict[str, int]:
    """Calza las fotos con la lista por la hora de la cámara y escribe un CSV por grupo con su SHA-256.

    Cada bloque son 1 foto al papelito separador y sus 10 tomas, en ese orden. Se detiene si falta o
    sobra una foto, si hay fotos de más de un día o de distinto tamaño, si alguna no es JPG o si
    ya hay un sello. La hoja de control muestra solo los separadores: tienen que leerse B01…B18.
    """
    carpeta, destino = Path(carpeta), Path(destino)
    lista = lista or leer_lista()
    salidas = [destino / f"sello_{g}.csv" for g in (*BUENAS, "separadores")]
    if any(p.exists() for p in salidas):
        raise FileExistsError("ya hay un sello: no se vuelve a sellar")
    fotos = sorted(p for p in carpeta.iterdir() if p.is_file() and not p.name.startswith("."))
    no_jpg = [p.name for p in fotos if p.suffix.lower() not in EXTENSIONES]
    if no_jpg:
        raise ValueError(f"no son JPG: {no_jpg[:5]}… (en el iPhone: Ajustes › Cámara › Formatos › Más compatible)")
    meta = sorted(((p, *fecha_camara(p)) for p in fotos), key=lambda m: (m[1], m[2], m[0].name))
    bloques = sorted({t["bloque"] for t in lista})
    if len(meta) != len(lista) + len(bloques):
        raise ValueError(f"hay {len(meta)} fotos y se esperaban {len(lista) + len(bloques)} "
                         f"({len(lista)} tomas y {len(bloques)} separadores)")
    dias = sorted({m[1][:10] for m in meta})
    if len(dias) != 1:
        raise ValueError(f"las fotos son de más de un día: {dias}")
    tamanos = Counter(m[3] for m in meta)
    if len(tamanos) != 1:
        raise ValueError(f"las fotos tienen distinto tamaño: {dict(tamanos)}")

    def fila(m):
        return dict(archivo=m[0].name, sha256=huella(m[0]), fecha=f"{m[1]}.{round(m[2] * 1000):03d}")

    filas, seps, i = {g: [] for g in BUENAS}, [], 0
    for b in bloques:
        seps.append(dict(bloque=b, **fila(meta[i])))
        i += 1
        for t in (t for t in lista if t["bloque"] == b):
            filas[t["grupo"]].append({**t, **fila(meta[i])})
            i += 1
    destino.mkdir(parents=True, exist_ok=True)
    for g, r in filas.items():
        _escribir(destino / f"sello_{g}.csv", r, (*CAMPOS, "archivo", "sha256", "fecha"))
    _escribir(destino / "sello_separadores.csv", seps, ("bloque", "archivo", "sha256", "fecha"))
    if hoja is not None:
        hoja_de_control([carpeta / s["archivo"] for s in seps], hoja)
    return {g: len(r) for g, r in filas.items()} | {"separadores": len(seps)}


def _escribir(ruta: Path, filas: list[dict], campos: tuple[str, ...]) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, campos)
        w.writeheader()
        w.writerows(filas)


def hoja_de_control(rutas: list[Path], salida: Path, ancho: int = 200) -> Path:
    """Miniaturas de los separadores, en orden, con el bloque esperado debajo de cada una."""
    from PIL import Image, ImageDraw, ImageOps

    alto, col = int(ancho * 4 / 3), 6
    lienzo = Image.new("RGB", (col * ancho, -(-len(rutas) // col) * (alto + 20)), "white")
    dibujo = ImageDraw.Draw(lienzo)
    for k, r in enumerate(rutas):
        with Image.open(r) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((ancho, alto))
            x, y = (k % col) * ancho, (k // col) * (alto + 20)
            lienzo.paste(im, (x, y))
            dibujo.text((x + 4, y + alto + 4), f"esperado B{k + 1:02d} · {r.name}", fill="black")
    salida.parent.mkdir(parents=True, exist_ok=True)
    lienzo.save(salida, quality=85)
    return salida


def cargar_grupo(grupo: str, carpeta: Path, destino: Path = DATOS) -> list[dict]:
    """Filas selladas de un grupo, con la ruta de cada foto. Se detiene si una huella no calza."""
    with open(destino / f"sello_{grupo}.csv", newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    for r in filas:
        r["ruta"] = Path(carpeta) / r["archivo"]
        if huella(r["ruta"]) != r["sha256"]:
            raise ValueError(f"{r['archivo']}: la huella no calza con el sello")
    return filas


def reducir(img: np.ndarray) -> np.ndarray:
    """Lleva la foto a 480 px de lado corto (INTER_AREA), como las 21 fotos originales de 480 × 640."""
    import cv2

    alto, ancho = img.shape[:2]
    f = LADO_CORTO / min(alto, ancho)
    if f >= 1:
        return img
    return cv2.resize(img, (round(ancho * f), round(alto * f)), interpolation=cv2.INTER_AREA)
