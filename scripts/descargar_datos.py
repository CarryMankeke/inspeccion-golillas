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
        # yellow_green_washer (584 / 144 + 72), como distribution.md. Las 9 piezas que sobran se borraron al sellar.
        url="https://ndownloader.figshare.com/files/50971446",
        bytes=3_197_059_395,
        md5="8b01e374f83bf02390d50b3cb4ad96f9",  # publicado por figshare (API, leída el 28 nov 2026)
        sha256="57d5a326cd24119cfa5406a7c267931c6840f9bd1f3c71c1599cf119290048bc",  # fijada el 28 nov 2026
        licencia="CC BY-NC-SA 4.0 (página del proyecto; figshare marca CC BY 4.0: se respeta la más restrictiva)",
        fuente="https://figshare.com/articles/dataset/manta-tiny/27934938",
        formato="tar.gz",
        # Tras sellar (commit 30ac891) se borraron las 9 piezas que sobran; quedan las dos golillas:
        archivos=1746,
        arbol_sha256="998c68885336efc8c348b5ed0563739a0c41cb9fa86375f268b224789e02b205",  # fijada el 28 nov 2026
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
    # Los tres de Roboflow del ADR 0017: la versión más reciente sin aumentos, exportada a mano con la
    # cuenta de Camilo el 28 nov 2026 (COCO; carpetas por clase en el de clasificación). Se descomprimen con
    # `--zip <nombre> <archivo.zip>`, que comprueba el zip y deja la huella del árbol.
    "roboflow_saksit_metal_washer": dict(
        manual=True,
        url="https://universe.roboflow.com/saksit-zdivw/metal-washer/dataset/4",  # v4, 313 fotos (194/60/59)
        zip_bytes=9_962_256,
        zip_sha256="60abd6f36ae470e8d5c351486525b007b15305bde575cd8f52f1fa91d3b07e6c",
        archivos=318,
        arbol_sha256="ddf2e98392cb16210c3be2aa252805b963f21443bdbb991fef32807c26824d09",  # fijada el 28 nov 2026
        licencia="Dominio público",
        fuente="https://universe.roboflow.com/saksit-zdivw/metal-washer",
    ),
    "roboflow_renuka_washer_defect": dict(
        manual=True,
        url="https://universe.roboflow.com/renuka-p/washer-defect-detection-ap9hk/dataset/3",  # v3, 151 (60/45/46)
        zip_bytes=141_203_887,
        zip_sha256="4cc9e9c7807eb9f1906f82ead63f08864572749d8edeac94cf3392c9715692e1",
        archivos=156,
        arbol_sha256="1efcb92040d737a2b0b8543218eb8a2059ab47933c974bc0e0c2732eb7437b37",  # fijada el 28 nov 2026
        licencia="CC BY 4.0",
        fuente="https://universe.roboflow.com/renuka-p/washer-defect-detection-ap9hk",
    ),
    "roboflow_washer_classifier": dict(
        manual=True,
        # v3, 443 fotos (377/54/12); es de clasificación, así que se exporta en carpetas por clase
        url="https://universe.roboflow.com/cap-inspection-cioib/washer-classifier-fohnr/dataset/3",
        zip_bytes=22_437_138,
        zip_sha256="48e4022fa6a3c882b1ee9e666e801003dd9ef867f76aea21b13e9123889615bf",
        archivos=445,
        arbol_sha256="79b44acd79440f154fd6a115392f98f897969b8ade2a9823b8907ec6637253f5",  # fijada el 28 nov 2026
        licencia="CC BY 4.0",
        fuente="https://universe.roboflow.com/cap-inspection-cioib/washer-classifier-fohnr",
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
    if f["arbol_sha256"] is None:
        print(f"{nombre}: {n} archivos, huella del árbol {h} (fíjala en FUENTES)")
        return
    if n != f["archivos"] or h != f["arbol_sha256"]:
        sys.exit(f"{nombre}: {n} archivos, huella {h}. No calza con la fijada; revisa la copia.")
    print(f"{nombre}: {n} archivos, huella del árbol correcta")


def desde_zip(nombre: str, zip_: Path) -> None:
    """Descomprime un zip exportado a mano en data/publico/<nombre>/, después de comprobar su tamaño y su huella."""
    f, carpeta = FUENTES[nombre], DESTINO / nombre
    if carpeta.exists():
        sys.exit(f"{nombre}: ya está en {carpeta}; bórralo a mano si quieres descomprimirlo de nuevo")
    if zip_.stat().st_size != f["zip_bytes"] or huella(zip_) != f["zip_sha256"]:
        sys.exit(f"{zip_.name}: el tamaño o la huella no calzan con los fijados; no es el zip exportado")
    with zipfile.ZipFile(zip_) as z:
        for m in z.namelist():  # sin rutas absolutas ni "..": el zip es dato externo
            if m.startswith("/") or ".." in Path(m).parts:
                sys.exit(f"ruta sospechosa dentro del zip: {m}")
        z.extractall(carpeta)
    verificar(nombre)


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
    p.add_argument("--zip", type=Path, help="zip exportado a mano para descomprimir en data/publico/<nombre>/")
    a = p.parse_args()
    if a.zip:
        desde_zip(a.nombres[0], a.zip)
        return
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
