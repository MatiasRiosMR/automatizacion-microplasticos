"""Informes por muestra (HTML + PDF) sobre imágenes sintéticas.

Genera dos ejemplos con :func:`napari_mp_classifier.informe_html.generar_informe_html`:

- **M-01** — control de composición conocida (incluye la sección de verificación).
- **M-02** — muestra «ambiental» sintética, algo más envejecida que la calibración y sin
  verdad de terreno (el caso habitual del investigador).

Uso::

    python ejemplos/demo_informe.py

Salida en ``ejemplos/salida_demo_informes/`` (el PDF requiere Chrome/Chromium).
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tests"))

import numpy as np
from datos_sinteticos import _columnas, generar_calibracion, generar_imagen_muestra

from napari_mp_classifier import Calibracion, analizar_muestra
from napari_mp_classifier.informe_html import generar_informe_html

SALIDA = RAIZ / "ejemplos" / "salida_demo_informes"
ESCALA_DEMO = 0.18  # µm/píxel, valor del demo (no medido)

CASOS = [
    ("M-01", "M-01 · control con composición conocida", 11, 0.0, True,
     "Muestra sintética de control: 4 partículas de cada polímero + materia orgánica."),
    ("M-02", "M-02 · muestra ambiental (sintética)", 3, 0.2, False,
     "Muestra ambiental simulada, algo más envejecida que la calibración. Sin verdad de terreno."),
]


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    columnas = _columnas("fusion")
    df = generar_calibracion("fusion", n_por_polimero=60, sigma=0.02, semilla=0)
    calibracion = Calibracion.desde_dataframe(
        df, columnas=columnas, metadatos={"origen": "sintética (tests/datos_sinteticos.py)"})
    mediciones = (df[columnas].to_numpy(), df["polimero"].to_numpy())

    for codigo, nombre, semilla, envejecimiento, con_verdad, descripcion in CASOS:
        canales, verdad = generar_imagen_muestra(semilla=semilla,
                                                 grado_envejecimiento=envejecimiento)
        archivo = SALIDA / f"{codigo}.npz"
        np.savez_compressed(archivo, **canales)
        resultado = analizar_muestra(
            canales, calibracion, estrategia="knn", mediciones_calibracion=mediciones,
            escala_um_px=ESCALA_DEMO, verdad=verdad if con_verdad else None, semilla=semilla,
        )
        ruta = generar_informe_html(
            resultado, canales, SALIDA / f"informe_{codigo}.html", nombre_muestra=nombre,
            archivo=archivo, descripcion=descripcion, nota_escala=" (valor del demo, no medido)",
            calibracion_flim="datos sintéticos, ya calibrados",
        )
        conteo = resultado.conteo_por_polimero().to_dict()
        print(f"{codigo}: {resultado.n_rois} partículas {conteo} → {ruta.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
