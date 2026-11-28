"""Sesión de fotos propias de la v5 (ADR 0016): la lista de tomas y el sellado.

Uso:  python scripts/sesion_v5.py lista
          Arma la lista al azar (semilla 0) y la escribe en data/propio_v5/lista_tomas.csv y en la
          tabla de docs/sesion-fotos-v5.md. Se corre una vez, antes de la sesión.
      python scripts/sesion_v5.py sellar data/propio_v5/originales
          Después de la sesión y antes de que ningún script mire las fotos: las calza con la lista
          por la hora de la cámara y escribe data/propio_v5/sello_<grupo>.csv con la huella SHA-256
          de cada foto. La hoja de control (cache/sello_separadores.jpg) muestra solo los papelitos
          separadores: tienen que leerse B01…B18 en orden. Los CSV van en un commit propio.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from golillas import sesion  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="orden", required=True)
    sub.add_parser("lista", help="arma la lista de tomas (semilla 0)")
    s = sub.add_parser("sellar", help="sella las fotos de la sesión por grupo")
    s.add_argument("carpeta", type=Path)
    a = p.parse_args()
    if a.orden == "lista":
        sesion.escribir_lista()
        print(f"lista escrita en {sesion.LISTA_CSV.relative_to(sesion.RAIZ)} y {sesion.GUIA.relative_to(sesion.RAIZ)}")
    else:
        n = sesion.sellar(a.carpeta)
        print("sello escrito:", n, "· revisa la hoja de control: cache/sello_separadores.jpg")


if __name__ == "__main__":
    main()
