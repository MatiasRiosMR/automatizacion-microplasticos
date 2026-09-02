"""Demo de la Fase 4: el plugin de napari sobre una muestra sintética.

Abre un visor de napari, agrega las capas de la muestra, corre el widget
``WidgetClasificador`` y engancha el ``PhasorPlotWidget`` (back-projection). Guarda
capturas en ``ejemplos/salida_demo_fase4/``.

Uso (necesita el extra ``[napari]`` y un entorno con Qt; en headless funciona con
``QT_QPA_PLATFORM=offscreen``)::

    conda run -n napari-mp-env python ejemplos/demo_fase4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tests"))

import numpy as np

SALIDA = RAIZ / "ejemplos" / "salida_demo_fase4"


def main() -> None:
    import os

    import napari
    from datos_sinteticos import generar_calibracion, generar_imagen_muestra

    from napari_mp_classifier.napari_integracion._phasor_plot import PhasorPlotWidget
    from napari_mp_classifier.napari_integracion._widget import WidgetClasificador
    from napari_mp_classifier.reportes import figura_segmentacion, guardar_figura

    SALIDA.mkdir(parents=True, exist_ok=True)
    canales, _verdad = generar_imagen_muestra(semilla=7)
    ruta_cal = SALIDA / "calibracion.csv"
    generar_calibracion("fusion", n_por_polimero=60, semilla=0).to_csv(ruta_cal, index=False)

    viewer = napari.Viewer(show=False)
    for nombre, arr in canales.items():
        viewer.add_image(np.nan_to_num(arr), name=nombre, visible=(nombre == "intensidad"))

    clf = WidgetClasificador(viewer)
    clf._intensidad.value = viewer.layers["intensidad"]
    for canal in ("g_flim", "s_flim", "g_esp", "s_esp"):
        clf._canales[canal].value = canal
    clf._calibracion.value = ruta_cal
    clf._ejecutar()
    print(clf._estado.value)

    viewer.window.add_dock_widget(clf, name="Clasificar microplásticos", area="right")
    plot = PhasorPlotWidget(viewer)
    viewer.window.add_dock_widget(plot, name="Diagrama de phasores", area="right")

    capa = viewer.layers["clasificación MP"]
    resultado = capa.metadata["resultado"]
    print(resultado.conteo_por_polimero().to_dict())

    # back-projection: seleccionar una ROI en el visor resalta su punto en el phasor plot.
    primer_label = int(resultado.features.index[0])
    capa.selected_label = primer_label
    plot._on_seleccion_capa()
    print(f"back-projection: ROI {primer_label} resaltada -> {plot._info.text()}")

    # Captura del visor y overlay: necesitan un contexto gráfico real (fallan en headless).
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        print("(headless: se omiten las capturas; el plugin queda verificado por los tests)")
    else:
        etiquetas = {
            int(l): p for l, p in zip(resultado.features.index, resultado.features["polimero_predicho"])
        }
        fig = figura_segmentacion(
            canales["intensidad"], resultado.labels, etiquetas_por_label=etiquetas,
            titulo="Capa 'clasificación MP' — ROIs por polímero predicho",
        )
        guardar_figura(fig, SALIDA / "clasificacion_overlay", formatos=("png",))
        plot._fig.savefig(SALIDA / "phasor_plot.png", dpi=150, bbox_inches="tight")
        viewer.screenshot(str(SALIDA / "visor_napari.png"), canvas_only=False, scale=2)
        print(f"Capturas en {SALIDA}")

    print(f"Salida en {SALIDA}")
    viewer.close()


if __name__ == "__main__":
    main()
