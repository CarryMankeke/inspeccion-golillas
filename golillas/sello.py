"""Sello de los datasets públicos del ADR 0017: partición y huellas, sin decodificar ninguna imagen.

El sello se arma ANTES de que cualquier programa mire los píxeles de un set de prueba. Solo lee
nombres de archivo, tamaños en bytes y etiquetas (las carpetas en MANTA, el JSON COCO en Roboflow).
Por cada conjunto deja tres CSV en data/sellos/<conjunto>/:

- ajusta.csv y calibra.csv: las buenas de entrenamiento, ordenadas por nombre, barajadas con
  np.random.default_rng(0) y partidas en dos mitades (la primera ajusta y la segunda calibra).
  Las piezas que comparten origen (copias aumentadas de una misma foto en Roboflow) quedan juntas.
- prueba.csv: la partición de prueba del dataset, tal cual viene.

Cada fila trae la ruta relativa, los bytes y la SHA-256 del archivo (y de su máscara, si tiene).
Los CSV van en un commit propio. Después, `cargar` entrega las filas de un grupo y se detiene si
una huella no calza, y `verificar` revisa además que no sobre ni falte ningún archivo.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from golillas.sesion import huella

RAIZ = Path(__file__).resolve().parents[1]
PUBLICO = RAIZ / "data" / "publico"
SELLOS = RAIZ / "data" / "sellos"
SEMILLA = 0
GRUPOS = ("ajusta", "calibra", "prueba")
CAMPOS = ("pieza", "grupo", "clase", "defecto", "archivo", "bytes", "sha256",
          "mascara", "mascara_bytes", "mascara_sha256", "origen", "caja")

MANTA = "manta_tiny_mechanics"
MANTA_TIPOS = {  # conteos de distribution.md (figshare), fijados en el ADR 0017
    "red_washer": dict(entrenamiento=578, buenas=148, malas=74),
    "yellow_green_washer": dict(entrenamiento=584, buenas=144, malas=72),
}


# ---------------------------------------------------------------- partición
def mitades(origenes: list[str], semilla: int = SEMILLA) -> list[str]:
    """«ajusta» o «calibra» para cada pieza de entrenamiento, en el mismo orden de entrada.

    Los orígenes distintos se ordenan, se barajan con default_rng(semilla) y se asignan a «ajusta»
    hasta cubrir la mitad de las piezas (redondeando hacia arriba); el resto calibra. Si cada pieza
    es su propio origen, como en MANTA, queda exacto: ajustan las de permutation(n)[:n/2].
    """
    unicos = sorted(set(origenes))
    cuenta = Counter(origenes)
    mitad = -(-len(origenes) // 2)
    grupo, acumulado = {}, 0
    for i in np.random.default_rng(semilla).permutation(len(unicos)):
        o = unicos[i]
        grupo[o] = "ajusta" if acumulado < mitad else "calibra"
        acumulado += cuenta[o]
    return [grupo[o] for o in origenes]


# ---------------------------------------------------------------- MANTA Tiny
def piezas_manta(tipo: str, raiz: Path = PUBLICO / MANTA) -> list[dict]:
    """Las piezas de una golilla de MANTA Tiny, leyendo solo carpetas y nombres de archivo.

    Cada PNG es una pieza con sus 5 vistas. Entrenamiento: train/good. Prueba: test/<defecto>, con
    «good» como buena y la máscara de cada mala en ground_truth/<defecto>/<mismo nombre>.
    """
    raiz = Path(raiz)
    base = raiz / "mechanics" / tipo

    def pngs(carpeta: Path) -> list[Path]:
        return sorted((p for p in carpeta.iterdir() if p.is_file() and p.suffix == ".png"), key=lambda p: p.name)

    def rel(p: Path) -> str:
        return p.relative_to(raiz).as_posix()

    filas = [dict(pieza=p.stem, particion="entrenamiento", clase="buena", defecto="ninguno",
                  archivo=rel(p), mascara="", origen=p.stem, caja="") for p in pngs(base / "train" / "good")]
    for d in sorted((q for q in (base / "test").iterdir() if q.is_dir()), key=lambda q: q.name):
        mala = d.name != "good"
        for p in pngs(d):
            m = base / "ground_truth" / d.name / p.name
            if mala and not m.is_file():
                raise FileNotFoundError(f"{rel(p)}: falta su máscara en ground_truth/{d.name}/")
            filas.append(dict(pieza=p.stem, particion="prueba", clase="mala" if mala else "buena",
                              defecto=d.name if mala else "ninguno", archivo=rel(p),
                              mascara=rel(m) if mala else "", origen=p.stem, caja=""))
    return filas


def sellar_manta(tipo: str, raiz: Path = PUBLICO / MANTA, destino: Path = SELLOS,
                 esperado: dict | None = None) -> dict[str, int]:
    """Sella una golilla de MANTA Tiny. Se detiene si los conteos no calzan con distribution.md."""
    filas = piezas_manta(tipo, raiz)
    esperado = esperado or MANTA_TIPOS[tipo]
    real = dict(entrenamiento=sum(f["particion"] == "entrenamiento" for f in filas),
                buenas=sum(f["particion"] == "prueba" and f["clase"] == "buena" for f in filas),
                malas=sum(f["particion"] == "prueba" and f["clase"] == "mala" for f in filas))
    if real != esperado:
        raise ValueError(f"{tipo}: los conteos {real} no calzan con los publicados {esperado}")
    return sellar(f"manta_{tipo}", filas, raiz, destino)


# ---------------------------------------------------------------- sello genérico
def sellar(conjunto: str, filas: list[dict], raiz: Path, destino: Path = SELLOS,
           semilla: int = SEMILLA) -> dict[str, int]:
    """Parte las buenas de entrenamiento en ajusta y calibra y escribe un CSV por grupo con las huellas.

    `filas` trae pieza, particion («entrenamiento» o «prueba»), clase, defecto, archivo y mascara
    (rutas relativas a `raiz`), origen y caja. Solo buenas pueden ajustar o calibrar.
    """
    raiz, carpeta = Path(raiz), Path(destino) / conjunto
    if carpeta.exists() and any(carpeta.iterdir()):
        raise FileExistsError(f"{conjunto}: ya hay un sello; no se vuelve a sellar")
    repetidas = [p for p, n in Counter(f["pieza"] for f in filas).items() if n > 1]
    if repetidas:
        raise ValueError(f"{conjunto}: piezas repetidas {repetidas[:5]}")
    entrenamiento = [f for f in filas if f["particion"] == "entrenamiento"]
    if any(f["clase"] != "buena" for f in entrenamiento):
        raise ValueError(f"{conjunto}: solo las buenas ajustan y calibran")
    if any(f["particion"] not in ("entrenamiento", "prueba") for f in filas):
        raise ValueError(f"{conjunto}: partición desconocida")
    grupos = dict(zip((f["pieza"] for f in entrenamiento), mitades([f["origen"] for f in entrenamiento], semilla)))

    huellas: dict[str, tuple[int, str]] = {}  # un archivo puede tener varias piezas (cajas de Roboflow)

    def bytes_y_huella(rel: str) -> tuple[int | str, str]:
        if not rel:
            return "", ""
        if rel not in huellas:
            ruta = raiz / rel
            huellas[rel] = (ruta.stat().st_size, huella(ruta))  # lee bytes, no decodifica
        return huellas[rel]

    salida = {g: [] for g in GRUPOS}
    for f in filas:
        g = grupos.get(f["pieza"], "prueba")
        b, h = bytes_y_huella(f["archivo"])
        mb, mh = bytes_y_huella(f.get("mascara", ""))
        salida[g].append(dict(pieza=f["pieza"], grupo=g, clase=f["clase"], defecto=f["defecto"],
                              archivo=f["archivo"], bytes=b, sha256=h, mascara=f.get("mascara", ""),
                              mascara_bytes=mb, mascara_sha256=mh, origen=f["origen"], caja=f.get("caja", "")))
    carpeta.mkdir(parents=True, exist_ok=True)
    for g, r in salida.items():
        with open(carpeta / f"{g}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, CAMPOS, lineterminator="\n")
            w.writeheader()
            w.writerows(r)
    return {g: len(r) for g, r in salida.items()}


def leer(conjunto: str, grupo: str, destino: Path = SELLOS) -> list[dict]:
    with open(Path(destino) / conjunto / f"{grupo}.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cargar(conjunto: str, grupo: str, raiz: Path, destino: Path = SELLOS) -> list[dict]:
    """Filas selladas de un grupo, con `ruta` (y `ruta_mascara`). Se detiene si un archivo cambió."""
    raiz, vistos = Path(raiz), {}
    filas = leer(conjunto, grupo, destino)
    for r in filas:
        for campo, b, h in (("archivo", "bytes", "sha256"), ("mascara", "mascara_bytes", "mascara_sha256")):
            if not r[campo]:
                continue
            ruta = raiz / r[campo]
            if r[campo] not in vistos:
                if not ruta.is_file() or ruta.stat().st_size != int(r[b]) or huella(ruta) != r[h]:
                    raise ValueError(f"{conjunto}/{grupo}: {r[campo]} no calza con el sello")
                vistos[r[campo]] = True
            r["ruta" if campo == "archivo" else "ruta_mascara"] = ruta
    return filas


def verificar(conjunto: str, raiz: Path, carpetas: list[str], destino: Path = SELLOS) -> dict[str, int]:
    """Comprueba las huellas de los tres grupos y que en `carpetas` no sobre ni falte ningún archivo."""
    raiz = Path(raiz)
    n = {g: len(cargar(conjunto, g, raiz, destino)) for g in GRUPOS}
    selladas = {r[c] for g in GRUPOS for r in leer(conjunto, g, destino) for c in ("archivo", "mascara") if r[c]}
    hay = {p.relative_to(raiz).as_posix() for c in carpetas for p in (raiz / c).rglob("*")
           if p.is_file() and p.name != ".DS_Store"}
    if hay != selladas:
        sobran, faltan = sorted(hay - selladas), sorted(selladas - hay)
        raise ValueError(f"{conjunto}: sobran {len(sobran)} {sobran[:3]} y faltan {len(faltan)} {faltan[:3]}")
    return n
