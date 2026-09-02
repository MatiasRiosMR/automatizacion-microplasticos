"""Interfaz de línea de comandos de napari-mp-classifier.

Comando principal::

    napari-mp-classifier classify muestra.npz \\
        --calibracion calibracion.csv --salida resultados/

- ``muestra.npz``: archivo NumPy con los canales de la muestra como arrays 2D. Claves
  reconocidas: ``intensidad`` (obligatoria) y ``g_flim`` / ``s_flim`` / ``g_esp`` /
  ``s_esp`` (al menos un par). La modalidad se deduce de los pares presentes.
- ``calibracion.csv``: una fila por medición de calibración, con las columnas de phasor
  correspondientes y una columna ``polimero``. Se usa para la firma de referencia y,
  con ``--estrategia knn``, para entrenar el clasificador.

La lectura de ``.sdt`` / ``.czi`` crudos pasa por ``io_crudo`` y se habilita cuando el
equipo entregue datos de ejemplo (ver ``docs/PREGUNTAS_DATOS.md``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__

_CANALES = ("intensidad", "g_flim", "s_flim", "g_esp", "s_esp")
_COLUMNAS_MODALIDAD = {
    "fusion": ["g_flim", "s_flim", "g_esp", "s_esp"],
    "flim": ["g_flim", "s_flim"],
    "espectral": ["g_esp", "s_esp"],
}


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="napari-mp-classifier",
        description="Clasificación de microplásticos por phasores espectral + FLIM.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="comando")
    p_clf = sub.add_parser("classify", help="Clasificar las partículas de una muestra.")
    p_clf.add_argument("muestra", help="Archivo .npz con los canales de la muestra.")
    p_clf.add_argument("--calibracion", required=True,
                       help="CSV de mediciones de calibración (con columna 'polimero').")
    p_clf.add_argument("--salida", required=True, help="Carpeta de resultados.")
    p_clf.add_argument("--modalidad", default="auto",
                       choices=["auto", "fusion", "flim", "espectral"],
                       help="Modalidad de clasificación. 'auto' la deduce de los canales.")
    p_clf.add_argument("--estrategia", default="knn",
                       choices=["centroide", "knn", "gmm"])
    p_clf.add_argument("--confianza", type=float, default=0.99,
                       help="Nivel de confianza de la regla 'no_clasificable' (0-1). "
                            "0 la desactiva.")
    p_clf.add_argument("--metodo-segmentacion", default="umbral",
                       choices=["umbral", "kmeans"])
    p_clf.add_argument("--escala-um-px", type=float, default=None,
                       help="Tamaño de píxel en µm (para area_um2).")
    p_clf.add_argument("--sin-separar-contacto", action="store_true",
                       help="No aplicar watershed para separar partículas en contacto.")
    return parser


def _cargar_muestra(ruta: str) -> dict[str, np.ndarray]:
    datos = np.load(ruta)
    canales = {c: np.asarray(datos[c], dtype=float) for c in _CANALES if c in datos.files}
    if "intensidad" not in canales:
        raise ValueError(f"{ruta} no tiene el canal 'intensidad'.")
    return canales


def _modalidad_de_canales(canales: dict[str, np.ndarray]) -> str:
    tiene_flim = "g_flim" in canales and "s_flim" in canales
    tiene_esp = "g_esp" in canales and "s_esp" in canales
    if tiene_flim and tiene_esp:
        return "fusion"
    if tiene_flim:
        return "flim"
    if tiene_esp:
        return "espectral"
    raise ValueError("La muestra no tiene ningún par de phasor (g/s) completo.")


def _classify(args: argparse.Namespace) -> int:
    from .calibracion import Calibracion
    from .pipeline import analizar_muestra
    from .reportes import generar_reporte

    canales = _cargar_muestra(args.muestra)
    modalidad = args.modalidad if args.modalidad != "auto" else _modalidad_de_canales(canales)
    columnas = _COLUMNAS_MODALIDAD[modalidad]

    df_cal = pd.read_csv(args.calibracion)
    faltan = [c for c in [*columnas, "polimero"] if c not in df_cal.columns]
    if faltan:
        print(f"El CSV de calibración no tiene las columnas {faltan}.", file=sys.stderr)
        return 2
    calibracion = Calibracion.desde_dataframe(df_cal, columnas=columnas)
    mediciones = (
        (df_cal[columnas].to_numpy(), df_cal["polimero"].to_numpy())
        if args.estrategia == "knn"
        else None
    )

    resultado = analizar_muestra(
        canales, calibracion,
        estrategia=args.estrategia,
        confianza=args.confianza if args.confianza > 0 else None,
        metodo_segmentacion=args.metodo_segmentacion,
        separar_contacto=not args.sin_separar_contacto,
        escala_um_px=args.escala_um_px,
        mediciones_calibracion=mediciones,
    )

    salida = Path(args.salida)
    rutas = generar_reporte(resultado, salida, canales=canales,
                            titulo=Path(args.muestra).stem)

    print(f"{resultado.n_rois} ROIs clasificadas (modalidad {modalidad}).")
    for etiqueta, n in resultado.conteo_por_polimero().items():
        print(f"  {etiqueta:>16}: {int(n)}")
    print(f"\nReporte en: {salida}")
    for clave, ruta in rutas.items():
        print(f"  - {clave}: {ruta.relative_to(salida)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la CLI. Devuelve el código de salida del proceso."""
    parser = _construir_parser()
    args = parser.parse_args(argv)

    if args.comando is None:
        parser.print_help()
        return 0
    if args.comando == "classify":
        try:
            return _classify(args)
        except (ValueError, FileNotFoundError, KeyError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
