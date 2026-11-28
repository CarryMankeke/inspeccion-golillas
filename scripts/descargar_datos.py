"""Descarga los datasets públicos que usa el proyecto. Las imágenes NO se suben al repo.

Uso:  python scripts/descargar_datos.py edge_impulse_flat_washers
      python scripts/descargar_datos.py --lista

Cada fuente queda en data/publico/<nombre>/ (ignorado por git). El script comprueba el
tamaño y la huella del archivo (SHA-256, o el MD5 que publica Zenodo). Si la huella aún no
está fijada en FUENTES, la imprime para fijarla en el commit siguiente: desde ahí,
cualquier cambio del archivo en el servidor detiene la descarga. Las licencias y
atribuciones están en data/ATRIBUCIONES.md.

Roboflow y Kaggle piden cuenta: se bajan a mano (Roboflow en formato COCO; Kaggle con su
conector o su CLI) y se descomprimen en data/publico/<nombre>/. Para esas fuentes,
`--verificar` calcula la huella del árbol (SHA-256 de la lista ordenada «sha256  ./ruta» de
cada archivo, como `sha256sum`) y la compara con la fijada:
      python scripts/descargar_datos.py --verificar roboflow_massmep_washers kaggle_simplified_washers
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "data" / "publico"

FUENTES = {
    "edge_impulse_flat_washers": dict(
        url="https://cdn.edgeimpulse.com/datasets/Visual+Anomaly+Detection+-+Flat+washers.zip",
        bytes=37_261_072,
        sha256="4a9c9b8bb291decd941aca2055c87760ffe38cbcd5cd9dcd109684cfc9db16fb",  # fijada el 27 nov 2026
        licencia="BSD 3-Clause Clear",
        fuente="https://docs.edgeimpulse.com/datasets/image/flat-washers",
    ),
    "csem_misd_washers": dict(
        url="https://zenodo.org/api/records/5513769/files/washer.tar.gz/content",
        bytes=888_092_869,
        md5="9097ae81e3ef62b778470b79902552f8",  # publicado por Zenodo
        sha256="642d4c37dff093b1182ea9211411561d3b3370878ffbdd422d8cba809c832f91",  # fijada el 27 nov 2026
        licencia="CC BY-NC-ND 4.0 (sin modificar ni redistribuir)",
        fuente="https://zenodo.org/records/5513769",
        formato="tar.gz",
    ),
    "manta_tiny_mechanics": dict(
        # MANTA Tiny (CVPR 2025): trae red_washer y yellow_green_washer, más otras 9 piezas (ADR 0017).
        # Bajada el 28 nov 2026 con el OK de Camilo. Las golillas: red_washer (578 / 148 + 74) y
        # yellow_green_washer (584 / 144 + 72), como distribution.md. Las 9 piezas que sobran se borran al sellar.
        url="https://ndownloader.figshare.com/files/50971446",
        bytes=3_197_059_395,
        md5="8b01e374f83bf02390d50b3cb4ad96f9",  # publicado por figshare (API, leída el 28 nov 2026)
        sha256="57d5a326cd24119cfa5406a7c267931c6840f9bd1f3c71c1599cf119290048bc",  # fijada el 28 nov 2026
        licencia="CC BY-NC-SA 4.0 (página del proyecto; figshare marca CC BY 4.0: se respeta la más restrictiva)",
        fuente="https://figshare.com/articles/dataset/manta-tiny/27934938",
        formato="tar.gz",
    ),
    "roboflow_massmep_washers": dict(
        manual=True,  # pide cuenta: se exporta a mano en formato COCO (Camilo, 27 nov 2026)
        url="https://universe.roboflow.com/massmep/washer-defects-3usoj",
        archivos=192,
        arbol_sha256="b55de72e78b4a448f5900eac3fe6dbe23ca7ab2731048d742f37264278b2c7fb",  # fijada el 27 nov 2026
        licencia="CC BY 4.0",
        fuente="https://universe.roboflow.com/massmep/washer-defects-3usoj",
    ),
    "kaggle_simplified_washers": dict(
        manual=True,  # pide cuenta: se baja con el conector o la CLI de Kaggle (27 nov 2026, versión 5)
        url="https://www.kaggle.com/datasets/thibdx/simplified-washers-for-anomaly-detection",
        zip_bytes=31_801_003,
        zip_sha256="83297b801eb3e5dc5fd970a3dbba55dd4ecb9d5c0dae1de904148a0206e5afbd",
        archivos=2403,
        arbol_sha256="c3311c1a11395eea10b6cf60e993952fedf8481ce3ba089d85a641361e280cc6",  # fijada el 27 nov 2026
        licencia="CC BY-SA 4.0",
        fuente="https://www.kaggle.com/datasets/thibdx/simplified-washers-for-anomaly-detection",
    ),
}


def huella(ruta: Path, algoritmo: str = "sha256") -> str:
    h = hashlib.new(algoritmo)
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def huella_arbol(carpeta: Path) -> tuple[str, int]:
    """SHA-256 de la lista ordenada «sha256  ./ruta» (igual que find | sort | sha256sum)."""
    rutas = sorted((r for r in carpeta.rglob("*") if r.is_file() and r.name != ".DS_Store"),
                   key=lambda r: str(r.relative_to(carpeta)).encode())
    lista = "".join(f"{huella(r)}  ./{r.relative_to(carpeta).as_posix()}\n" for r in rutas)
    return hashlib.sha256(lista.encode()).hexdigest(), len(rutas)


def verificar(nombre: str) -> None:
    f, carpeta = FUENTES[nombre], DESTINO / nombre
    if not carpeta.exists():
        sys.exit(f"{nombre}: no está en {carpeta}. Bájalo a mano desde {f['url']}.")
    h, n = huella_arbol(carpeta)
    if n != f["archivos"] or h != f["arbol_sha256"]:
        sys.exit(f"{nombre}: {n} archivos, huella {h}. No calza con la fijada; revisa la copia.")
    print(f"{nombre}: {n} archivos, huella del árbol correcta")


def descargar(nombre: str, forzar: bool = False) -> Path:
    f = FUENTES[nombre]
    if f.get("manual"):
        verificar(nombre)
        return DESTINO / nombre
    carpeta = DESTINO / nombre
    if carpeta.exists() and not forzar:
        print(f"{nombre}: ya está en {carpeta} (usa --forzar para bajarlo de nuevo)")
        return carpeta
    DESTINO.mkdir(parents=True, exist_ok=True)
    zip_ = DESTINO / f"{nombre}.{f.get('formato', 'zip')}"
    print(f"{nombre}: bajando {f['bytes'] / 1e6:.1f} MB desde {f['url']}  ({f['licencia']})")
    urllib.request.urlretrieve(f["url"], zip_)
    tam = zip_.stat().st_size
    if tam != f["bytes"]:
        sys.exit(f"tamaño distinto: {tam} bytes, se esperaban {f['bytes']}. Revisa la fuente antes de seguir.")
    if f.get("md5") and huella(zip_, "md5") != f["md5"]:
        sys.exit("MD5 distinto al publicado por la fuente. Revisa antes de seguir.")
    h = huella(zip_)
    if f["sha256"] is None:
        print(f"huella SHA-256 (fíjala en FUENTES): {h}")
    elif h != f["sha256"]:
        sys.exit(f"huella distinta: {h}. El archivo cambió en el servidor; revisa la fuente.")
    if f.get("formato") == "tar.gz":
        with tarfile.open(zip_) as t:
            t.extractall(carpeta, filter="data")  # el filtro rechaza rutas absolutas, ".." y enlaces raros
    else:
        with zipfile.ZipFile(zip_) as z:
            for m in z.namelist():  # sin rutas absolutas ni "..": el zip es dato externo
                if m.startswith("/") or ".." in Path(m).parts:
                    sys.exit(f"ruta sospechosa dentro del zip: {m}")
            z.extractall(carpeta)
    zip_.unlink()
    print(f"{nombre}: listo en {carpeta}")
    return carpeta


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("nombres", nargs="*")
    p.add_argument("--lista", action="store_true")
    p.add_argument("--forzar", action="store_true")
    p.add_argument("--verificar", action="store_true", help="solo comprueba la huella de una copia manual")
    a = p.parse_args()
    if a.lista or not a.nombres:
        for n, f in FUENTES.items():
            tam = "a mano" if f.get("manual") else f"{f['bytes'] / 1e6:.1f} MB"
            print(f"{n}: {tam} · {f['licencia']} · {f['fuente']}")
        return
    if a.verificar:
        for n in a.nombres:
            verificar(n)
        return
    for n in a.nombres:
        descargar(n, a.forzar)


if __name__ == "__main__":
    main()
