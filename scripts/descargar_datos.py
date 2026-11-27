"""Descarga los datasets públicos que usa el proyecto. Las imágenes NO se suben al repo.

Uso:  python scripts/descargar_datos.py edge_impulse_flat_washers
      python scripts/descargar_datos.py --lista

Cada fuente queda en data/publico/<nombre>/ (ignorado por git). El script comprueba el
tamaño y la huella del archivo (SHA-256, o el MD5 que publica Zenodo). Si la huella aún no
está fijada en FUENTES, la imprime para fijarla en el commit siguiente: desde ahí,
cualquier cambio del archivo en el servidor detiene la descarga. Las licencias y
atribuciones están en data/ATRIBUCIONES.md. Roboflow y Kaggle piden cuenta: esos se bajan
a mano (ver docs/adr/0013-datos-publicos.md).
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
        sha256=None,  # se fija tras la primera descarga
        licencia="BSD 3-Clause Clear",
        fuente="https://docs.edgeimpulse.com/datasets/image/flat-washers",
    ),
    "csem_misd_washers": dict(
        url="https://zenodo.org/api/records/5513769/files/washer.tar.gz/content",
        bytes=888_092_869,
        md5="9097ae81e3ef62b778470b79902552f8",  # publicado por Zenodo
        sha256=None,
        licencia="CC BY-NC-ND 4.0 (sin modificar ni redistribuir)",
        fuente="https://zenodo.org/records/5513769",
        formato="tar.gz",
    ),
}


def huella(ruta: Path, algoritmo: str = "sha256") -> str:
    h = hashlib.new(algoritmo)
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def descargar(nombre: str, forzar: bool = False) -> Path:
    f = FUENTES[nombre]
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
    a = p.parse_args()
    if a.lista or not a.nombres:
        for n, f in FUENTES.items():
            print(f"{n}: {f['bytes'] / 1e6:.1f} MB · {f['licencia']} · {f['fuente']}")
        return
    for n in a.nombres:
        descargar(n, a.forzar)


if __name__ == "__main__":
    main()
