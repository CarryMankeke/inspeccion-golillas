"""Sella los datasets públicos que deciden el ADR 0017, sin decodificar ninguna imagen.

Uso:  python scripts/sellar.py manta
          Parte las buenas de entrenamiento de red_washer y yellow_green_washer en ajusta y
          calibra (semilla 0) y escribe data/sellos/manta_<tipo>/{ajusta,calibra,prueba}.csv
          con la SHA-256 de cada archivo y de cada máscara. Los CSV van en un commit propio.
      python scripts/sellar.py verificar
          Revisa las huellas de todo lo sellado y que no sobre ni falte ningún archivo.
"""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from golillas import sello  # noqa: E402

CARPETAS = {f"manta_{t}": (sello.PUBLICO / sello.MANTA, [f"mechanics/{t}"]) for t in sello.MANTA_TIPOS}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("orden", choices=("manta", "verificar"))
    a = p.parse_args()
    if a.orden == "manta":
        for t in sello.MANTA_TIPOS:
            print(f"manta_{t}:", sello.sellar_manta(t))
    for c, (raiz, carpetas) in CARPETAS.items():
        if not (sello.SELLOS / c).exists():
            continue
        n = sello.verificar(c, raiz, carpetas)
        csvs = " · ".join(f"{g} {hashlib.sha256((sello.SELLOS / c / f'{g}.csv').read_bytes()).hexdigest()[:12]}"
                          for g in sello.GRUPOS)
        print(f"{c}: {n}, huellas correctas; SHA-256 de los CSV: {csvs}")


if __name__ == "__main__":
    main()
