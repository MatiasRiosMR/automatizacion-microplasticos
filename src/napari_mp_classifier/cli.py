"""Interfaz de línea de comandos de napari-mp-classifier.

Comando principal::

    napari-mp-classifier classify muestra.npz \\
        --calibracion calibracion.csv --salida resultados/

- ``muestra.npz``: archivo NumPy con los canales de la muestra como arrays 2D. Claves
  reconocidas: ``intensidad`` (obligatoria) y ``g_flim`` / ``s_flim`` / ``g_esp`` /
  ``s_esp`` (al menos un par). La modalidad se deduce de los pares presentes.
- ``calibracion.csv``: una fila por medición de calibración, con las columnas de phasor
  correspondientes y una columna ``polimero``. Se usa para la firma de referencia y,
  con ``--estrategia knn``/``gmm``, para entrenar el clasificador. También se acepta un
  ``.json`` guardado con :meth:`Calibracion.guardar_json` (solo ``centroide``/``gmm``).

Además del CSV de asignaciones, las métricas y las figuras, escribe el **informe por
muestra** ``informe.html`` (+ ``informe.pdf`` si hay Chrome/Chromium); ``--sin-informe``
lo omite. ``--modalidad`` restringe el análisis a FLIM o espectral aunque el ``.npz``
traiga los cuatro canales.

La lectura de ``.sdt`` / ``.czi`` crudos se hace con :mod:`napari_mp_classifier.io_crudo`.
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
    p_clf.add_argument("--estrategia", default="centroide",
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
    p_clf.add_argument("--nombre", default=None,
                       help="Nombre de la muestra para el informe (por defecto, el del archivo).")
    p_clf.add_argument("--sin-informe", action="store_true",
                       help="No generar el informe HTML/PDF por muestra.")
    p_clf.add_argument("--sin-pdf", action="store_true",
                       help="Generar el informe HTML pero no el PDF.")
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
    disponible = _modalidad_de_canales(canales)
    modalidad = args.modalidad if args.modalidad != "auto" else disponible
    columnas = _COLUMNAS_MODALIDAD[modalidad]
    faltan_canales = [c for c in columnas if c not in canales]
    if faltan_canales:
        print(f"La muestra no tiene los canales {faltan_canales} para la modalidad "
              f"'{modalidad}'.", file=sys.stderr)
        return 2
    # Se conservan solo los canales de la modalidad elegida (el pipeline la deduce de ellos).
    canales = {c: v for c, v in canales.items() if c == "intensidad" or c in columnas}

    mediciones = None
    if str(args.calibracion).lower().endswith(".json"):
        calibracion = Calibracion.cargar_json(args.calibracion)
        if list(calibracion.columnas) != columnas:
            print(f"La calibración tiene columnas {calibracion.columnas}; la modalidad "
                  f"'{modalidad}' necesita {columnas}.", file=sys.stderr)
            return 2
        if args.estrategia == "knn":
            print("Con una calibración .json usá --estrategia centroide o gmm (knn necesita "
                  "las mediciones individuales en CSV).", file=sys.stderr)
            return 2
    else:
        df_cal = pd.read_csv(args.calibracion)
        faltan = [c for c in [*columnas, "polimero"] if c not in df_cal.columns]
        if faltan:
            print(f"El CSV de calibración no tiene las columnas {faltan}.", file=sys.stderr)
            return 2
        calibracion = Calibracion.desde_dataframe(df_cal, columnas=columnas)
        if args.estrategia in ("knn", "gmm"):
            mediciones = (df_cal[columnas].to_numpy(), df_cal["polimero"].to_numpy())

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
    if not args.sin_informe:
        from .informe_html import generar_informe_html

        rutas["informe"] = generar_informe_html(
            resultado, canales, salida / "informe.html",
            nombre_muestra=args.nombre or Path(args.muestra).stem, archivo=args.muestra,
            escala_um_px=args.escala_um_px, exportar_pdf=not args.sin_pdf,
        )
        if (salida / "informe.pdf").exists():
            rutas["informe_pdf"] = salida / "informe.pdf"

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
