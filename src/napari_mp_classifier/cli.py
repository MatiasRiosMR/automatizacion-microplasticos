"""Interfaz de línea de comandos de napari-mp-classifier.

Estado: **Fase 3** (esqueleto). Comando previsto::

    napari-mp-classifier classify muestra.tif \\
        --calibracion calibracion.csv --salida resultados/

Hoy expone solo ``version`` y deja el ``parser`` armado para las Fases 2-3.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="napari-mp-classifier",
        description="Clasificación de microplásticos por phasores espectral + FLIM.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="comando")
    p_clf = sub.add_parser("classify", help="Clasificar partículas de una muestra (Fase 3).")
    p_clf.add_argument("muestra", help="Imagen de la muestra (.tif/.sdt/.czi).")
    p_clf.add_argument("--calibracion", required=True, help="CSV de calibración de los 6 polímeros.")
    p_clf.add_argument("--salida", required=True, help="Carpeta de resultados.")
    p_clf.add_argument(
        "--estrategia", default="centroide", choices=["centroide", "knn", "gmm"]
    )
    p_clf.add_argument("--umbral-no-clasificable", type=float, default=3.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la CLI. Devuelve el código de salida del proceso."""
    parser = _construir_parser()
    args = parser.parse_args(argv)

    if args.comando is None:
        parser.print_help()
        return 0

    if args.comando == "classify":
        print(
            "El comando 'classify' se implementa en la Fase 3 "
            "(pipeline segmentación → features → fusión → clasificación → reporte).",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
