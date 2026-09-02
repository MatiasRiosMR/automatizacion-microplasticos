"""Tests del plugin de napari (Fase 4).

Se saltan enteros si ``napari`` no está instalado (entorno sin el extra ``[napari]``).
Necesitan Qt (``pytest-qt`` aporta el manejo del event loop; napari, la fixture
``make_napari_viewer``).
"""

import numpy as np
import pandas as pd
import pytest

napari = pytest.importorskip("napari")

from datos_sinteticos import generar_calibracion, generar_imagen_muestra


@pytest.fixture
def muestra_en_capas(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    canales, verdad = generar_imagen_muestra(semilla=3)
    for nombre, arr in canales.items():
        viewer.add_image(np.nan_to_num(arr), name=nombre)

    ruta_cal = tmp_path / "cal.csv"
    generar_calibracion("fusion", n_por_polimero=60, semilla=0).to_csv(ruta_cal, index=False)
    return viewer, canales, verdad, ruta_cal


def test_manifest_valido():
    from importlib.resources import files

    from npe2 import PluginManifest

    ruta = files("napari_mp_classifier.napari_integracion") / "napari.yaml"
    manifest = PluginManifest.from_file(str(ruta))
    assert manifest.name == "napari-mp-classifier"
    assert len(manifest.contributions.widgets) == 2


def test_widget_clasificador_construye(make_napari_viewer):
    from napari_mp_classifier.napari_integracion._widget import WidgetClasificador

    viewer = make_napari_viewer()
    widget = WidgetClasificador(viewer)
    assert widget._estrategia.value == "knn"


def test_flujo_completo_agrega_capa_clasificada(muestra_en_capas):
    from napari_mp_classifier.napari_integracion._widget import WidgetClasificador

    viewer, _canales, _verdad, ruta_cal = muestra_en_capas
    widget = WidgetClasificador(viewer)
    widget._intensidad.value = viewer.layers["intensidad"]
    for canal in ("g_flim", "s_flim", "g_esp", "s_esp"):
        widget._canales[canal].value = canal
    widget._calibracion.value = ruta_cal

    widget._ejecutar()

    assert "clasificación MP" in viewer.layers
    capa = viewer.layers["clasificación MP"]
    assert capa.data.max() > 5
    assert "polimero_predicho" in capa.features.columns
    assert "resultado" in capa.metadata
    assert "ROIs" in widget._estado.value


def test_flujo_sin_calibracion_muestra_error(muestra_en_capas):
    from napari_mp_classifier.napari_integracion._widget import WidgetClasificador

    viewer, _canales, _verdad, _ruta = muestra_en_capas
    widget = WidgetClasificador(viewer)
    widget._intensidad.value = viewer.layers["intensidad"]
    widget._canales["g_flim"].value = "g_flim"
    widget._canales["s_flim"].value = "s_flim"
    widget._calibracion.value = "no_existe.csv"

    widget._ejecutar()
    assert widget._estado.value.startswith("⚠")
    assert "clasificación MP" not in viewer.layers


def test_phasor_plot_back_projection(muestra_en_capas, qtbot):
    from napari_mp_classifier.napari_integracion._phasor_plot import PhasorPlotWidget
    from napari_mp_classifier.napari_integracion._widget import WidgetClasificador

    viewer, _canales, _verdad, ruta_cal = muestra_en_capas
    clf = WidgetClasificador(viewer)
    clf._intensidad.value = viewer.layers["intensidad"]
    for canal in ("g_flim", "s_flim", "g_esp", "s_esp"):
        clf._canales[canal].value = canal
    clf._calibracion.value = ruta_cal
    clf._ejecutar()

    plot = PhasorPlotWidget(viewer)
    qtbot.addWidget(plot)
    assert plot._resultado is not None
    assert plot._puntos_scatter is not None

    # visor -> phasor plot: seleccionar una ROI resalta su punto.
    capa = viewer.layers["clasificación MP"]
    primer_label = int(capa.features["label"].iloc[0])
    capa.selected_label = primer_label
    plot._on_seleccion_capa()
    assert plot._resaltado.get_offsets().shape[0] == 1

    # phasor plot -> visor: simular un pick sobre el primer punto.
    class _Ev:
        artist = plot._puntos_scatter
        ind = (0,)

    plot._on_pick(_Ev())
    assert int(capa.selected_label) == int(plot._resultado.features.index[0])


def test_colores_por_label():
    from napari_mp_classifier.napari_integracion._widget import colores_por_label

    feats = pd.DataFrame(
        {"polimero_predicho": ["PET", "no_clasificable", "PS"]}, index=[1, 2, 3]
    )
    colores = colores_por_label(feats)
    assert set(colores) == {1, 2, 3}
    assert all(len(c) == 4 for c in colores.values())
